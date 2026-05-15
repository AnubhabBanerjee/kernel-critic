"""
Generator node: turn an OptimizationStrategy + analysis into CUDA C++ source.

Two paths exist:
- First pass: no previous attempt in state → emit a fresh kernel from the
  strategy fields.
- Repair pass: state has both `source_code` (previous attempt) and
  `compiler_error` (nvcc log). The model gets the broken source, the error,
  and the always-captured nvcc diagnostic, and is asked to fix it.

We also implement a defensive code-fence extractor so we don't accidentally
hand `nvcc` a Python/PyTorch block the model emitted as "test code".
"""
import re
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from src.config import get_llm


# ---------------------------------------------------------------------------
# Code fence extraction
# ---------------------------------------------------------------------------
# We capture both the fence's language tag (group 1) and body (group 2) so we
# can prefer fences explicitly labelled `cuda`/`cu`. The earlier version that
# returned the longest body would happily hand back a Python driver script
# if the model decided to be "helpful".
_FENCE_RE = re.compile(
    r"```([a-zA-Z0-9_+#-]*)\s*\n(.*?)```",
    re.DOTALL,
)

# Heuristic gates: drop fences that are too tiny to be a real kernel, and
# refuse absurdly large blobs (usually pasted docs).
_MIN_KERNEL_CHARS = 80
_MAX_KERNEL_CHARS = 500_000

# Language tags we consider "this is CUDA".
_CUDA_LANGS = frozenset({"cuda", "cu"})


def _has_device_entrypoints(body: str) -> bool:
    """Cheap sanity check that a fence actually contains device code."""
    return "__global__" in body or "__device__" in body


def extract_cuda_source(raw: str) -> str:
    """
    Pull the most plausible CUDA body out of an LLM response.
    Tier strategy (highest precedence first):
      A) Fence tagged ``cuda``/``cu`` AND contains __global__/__device__ AND
         length is in [MIN, MAX].
      B) Fence tagged ``cuda``/``cu`` AND length in range (model labelled it
         correctly but skipped the entrypoint heuristic).
      C) Any fence containing __global__/__device__ AND length in range
         (model used ``cpp`` instead of ``cuda``).
    Among ties we pick the SHORTEST body — long "helpful" wrappers tend to be
    wrong; a tight ~few-hundred-line kernel is usually what we want.
    """
    if not raw:
        return ""

    blocks = _FENCE_RE.findall(raw)  # [(lang, body), ...]
    if not blocks:
        # No fences at all → return the stripped string and let nvcc judge.
        return raw.strip()

    # Normalize: lowercase the language tag, strip the body. Skip empties.
    bodies = [
        (lang.strip().lower(), body.strip())
        for lang, body in blocks
        if body.strip()
    ]
    if not bodies:
        return raw.strip()

    def in_len(s: str) -> bool:
        return _MIN_KERNEL_CHARS <= len(s) <= _MAX_KERNEL_CHARS

    tier_a = [
        body for lang, body in bodies
        if lang in _CUDA_LANGS and _has_device_entrypoints(body) and in_len(body)
    ]
    tier_b = [
        body for lang, body in bodies
        if lang in _CUDA_LANGS and in_len(body)
    ]
    tier_c = [
        body for lang, body in bodies
        if _has_device_entrypoints(body) and in_len(body)
    ]

    for group in (tier_a, tier_b, tier_c):
        if group:
            return min(group, key=len)

    # No qualifying fence at all — fall back to the legacy "longest body"
    # behavior rather than returning nothing. nvcc will tell us if it's bad.
    stripped = [b for _, b in bodies]
    return max(stripped, key=len) if stripped else raw.strip()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# We split into FIRST_PASS vs REPAIR. Both share the "Hardware Constraints"
# block (so the model sees the same strategy contract either way), but the
# repair prompt also gets the previous source, the compiler error, and the
# captured ptxas diagnostic.

_FIRST_PASS_SYSTEM = (
    "You are a Senior CUDA Developer. Implement a high-performance C++ kernel.\n\n"
    "Hardware Constraints to Implement:\n"
    "- Grid Dimensions: {grid_size}\n"
    "- Block Dimensions: {block_size}\n"
    "- Memory Strategy: {mem_strategy}\n"
    "- Registers per Thread: {registers_per_thread}\n"
    "- Shared Memory (bytes per block): {shared_memory_bytes}\n"
    "- Vectorization Width: {vectorization_width}\n"
    "- Precision: {precision}\n"
    "- Compiler Flags: {compiler_flags}\n\n"
    # The execution-config block is a hard guardrail against the "make errors
    # go away by setting blockDim=(1,1,1)" failure mode.
    "Execution configuration bounds (do not violate these just to silence errors):\n"
    "- Prefer staying close to the suggested grid_size and block_size unless the compiler log proves they are invalid.\n"
    "- For typical 1D kernels, blockDim.x must be a positive multiple of 32; avoid degenerate configs like (1,1,1) unless the log explicitly requires it.\n"
    "- Total threads per block (blockDim.x*blockDim.y*blockDim.z) should be at least 32 and typically >= 128 for reasonable occupancy unless the task forbids it.\n\n"
    "Requirements:\n"
    "- Use 'extern \"C\"' for potential compatibility with JIT compilers.\n"
    "- Include robust error handling for CUDA API calls.\n"
    "- Ensure the implementation matches the technical analysis provided.\n"
    "- Return ONLY CUDA C++ source. No prose, no Markdown fences.\n"
    # The two "exactly one fence" lines work together with extract_cuda_source
    # to make multi-block / wrong-language output extremely unlikely.
    "- Emit EXACTLY ONE Markdown code fence for the CUDA source, with language tag `cuda` (i.e. opening line ```cuda).\n"
    "- Do not include any other fenced code blocks (no Python driver, no shell).\n\n"
)

_FIRST_PASS_HUMAN = "Task: {task}\n\nTechnical Analysis:\n{analysis}"


_REPAIR_SYSTEM = (
    "You are a Senior CUDA Developer. Your previous kernel FAILED to compile under nvcc.\n"
    "Produce a corrected version that compiles cleanly. You may deviate from the "
    "originally suggested grid/block dimensions if they are the cause of the failure, "
    "but preserve the overall optimization intent.\n\n"
    "Originally suggested constraints (treat as guidance, not hard requirements on this retry):\n"
    "- Grid Dimensions: {grid_size}\n"
    "- Block Dimensions: {block_size}\n"
    "- Memory Strategy: {mem_strategy}\n"
    "- Registers per Thread: {registers_per_thread}\n"
    "- Shared Memory (bytes per block): {shared_memory_bytes}\n"
    "- Vectorization Width: {vectorization_width}\n"
    "- Precision: {precision}\n"
    "- Compiler Flags: {compiler_flags}\n\n"
    "Execution configuration bounds (do not violate these just to silence errors):\n"
    "- Prefer staying close to the suggested grid_size and block_size unless the compiler log proves they are invalid.\n"
    "- For typical 1D kernels, blockDim.x must be a positive multiple of 32; avoid degenerate configs like (1,1,1) unless the log explicitly requires it.\n"
    "- Total threads per block (blockDim.x*blockDim.y*blockDim.z) should be at least 32 and typically >= 128 for reasonable occupancy unless the task forbids it.\n\n"
    "Hard requirements:\n"
    "- The output must compile with `nvcc -c`.\n"
    "- Do not reintroduce any of the errors listed in the compiler log.\n"
    "- Return ONLY CUDA C++ source. No prose, no Markdown fences.\n"
    "- Emit EXACTLY ONE Markdown code fence for the CUDA source, with language tag `cuda` (i.e. opening line ```cuda).\n\n"
)

_REPAIR_HUMAN = (
    "Task: {task}\n\n"
    "Technical Analysis:\n{analysis}\n\n"
    # The previous broken source — useful so the model can do a *minimal* fix
    # rather than rewriting from scratch each retry.
    "Previous (broken) source on attempt #{attempt}:\n"
    "----- BEGIN PREVIOUS SOURCE -----\n{prev_code}\n----- END PREVIOUS SOURCE -----\n\n"
    # The actual nvcc errors that caused the failure.
    "nvcc error log:\n"
    "----- BEGIN COMPILER LOG -----\n{prev_error}\n----- END COMPILER LOG -----\n\n"
    # The ALWAYS-captured ptxas diagnostic. Even on the first repair, this
    # gives the model resource-usage context (registers, smem, spills) that
    # plain error messages don't include.
    "Last nvcc / ptxas diagnostic (resource usage, spills, register pressure):\n"
    "----- BEGIN NVCC DIAGNOSTIC -----\n{nvcc_log}\n----- END NVCC DIAGNOSTIC -----\n\n"
    "Fix every error reported above. Also use the ptxas resource lines (registers, "
    "shared memory, spill stores/loads) to adjust __launch_bounds__, register usage, "
    "or shared memory when they indicate spilling or excessive smem pressure. "
    "Output the full corrected kernel."
)


# ---------------------------------------------------------------------------
# Node entrypoint
# ---------------------------------------------------------------------------

def generator_node(state: AgentState) -> dict:
    """
    Translate `state["strategy"]` + analysis (+ optional repair context) into
    CUDA C++ source and stash it in `state["source_code"]`. Repair detection: we treat 
    the call as a repair pass iff BOTH a previous `source_code` and a previous `compiler_error` 
    exist in state. That avoids accidentally going into repair mode on the first pass when 
    source_code is None.
    """
    strategy = state["strategy"]
    analysis = state["analysis"]

    prev_code = state.get("source_code")
    prev_error = state.get("compiler_error")
    is_repair = bool(prev_code and prev_error)

    # Temperature heuristic:
    # - First pass: 0.0 → deterministic, predictable baseline.
    # - Repair: small positive value so the model can break out of a fix that
    #   keeps producing the same broken code. Without this, temperature 0 +
    #   identical inputs reliably reproduces the same wrong patch.
    llm = get_llm(temperature=0.2 if is_repair else 0.0)

    if is_repair:
        # Use the repair templates; payload must satisfy every {placeholder}
        # the templates reference, otherwise LangChain raises at invoke time.
        prompt = ChatPromptTemplate.from_messages([
            ("system", _REPAIR_SYSTEM),
            ("human", _REPAIR_HUMAN),
        ])
        payload = {
            "grid_size": strategy.grid_size,
            "block_size": strategy.block_size,
            # MemoryConfig is a nested model — dump to JSON so the model sees
            # a readable structured block, not Python's default repr.
            "mem_strategy": strategy.memory_strategy.model_dump_json(indent=2),
            "registers_per_thread": strategy.registers_per_thread,
            "shared_memory_bytes": strategy.shared_memory_bytes,
            "vectorization_width": strategy.vectorization_width,
            # Enum → its string value, so the model sees "FP32" not "PrecisionType.FP32".
            "precision": strategy.precision.value,
            # Join compiler_flags into a single readable line; nvcc args are
            # validated and re-split in critic.py, this is purely cosmetic
            # for the prompt.
            "compiler_flags": " ".join(strategy.compiler_flags),
            "task": state["task"],
            "analysis": analysis,
            "prev_code": prev_code,
            "prev_error": prev_error,
            "attempt": state.get("retry_count", 0),
            # The ALWAYS-on ptxas log. If the very first compile failed before
            # we ever ran ptxas, fall back to a placeholder so the template
            # variable is satisfied.
            "nvcc_log": state.get("nvcc_diagnostic_log") or "(no diagnostic captured yet)",
        }
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", _FIRST_PASS_SYSTEM),
            ("human", _FIRST_PASS_HUMAN),
        ])
        payload = {
            "grid_size": strategy.grid_size,
            "block_size": strategy.block_size,
            "mem_strategy": strategy.memory_strategy.model_dump_json(indent=2),
            "registers_per_thread": strategy.registers_per_thread,
            "shared_memory_bytes": strategy.shared_memory_bytes,
            "vectorization_width": strategy.vectorization_width,
            "precision": strategy.precision.value,
            "compiler_flags": " ".join(strategy.compiler_flags),
            "task": state["task"],
            "analysis": analysis,
        }

    # The chain is just: format prompt → call LLM. We do NOT use structured
    # output here because the LLM needs to emit raw CUDA, not a JSON object.
    chain = prompt | llm
    response = chain.invoke(payload)

    # Defensive extraction: strip prose / pick the right fence.
    cleaned = extract_cuda_source(response.content)
    return {"source_code": cleaned}