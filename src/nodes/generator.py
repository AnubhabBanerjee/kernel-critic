import re
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from src.config import get_llm


# Regex that captures fenced code blocks of the form ```<lang>\n<body>\n```
# We accept any language tag (cpp, c++, cuda, c, or none) and pick the largest body.
_FENCE_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+#-]+)?\s*\n(.*?)```",
    re.DOTALL,
)


def extract_cuda_source(raw: str) -> str:
    """
    Robustly pull CUDA C++ out of an LLM response.

    The Generator's downstream consumer is `nvcc`, which cannot tolerate
    Markdown fences or prose. LLMs almost always wrap code in ```cpp ... ```
    even when explicitly told not to, so we normalize here.

    Strategy:
      1. If the response contains fenced blocks, return the longest one.
      2. Otherwise return the response stripped of leading/trailing whitespace.
    """
    if not raw:
        return ""
    blocks = _FENCE_RE.findall(raw)
    if blocks:
        return max(blocks, key=len).strip()
    return raw.strip()


# --- Prompts -----------------------------------------------------------------

_FIRST_PASS_SYSTEM = (
    "You are a Senior CUDA Developer. Implement a high-performance C++ kernel.\n\n"
    "Hardware Constraints to Implement:\n"
    "- Grid Dimensions: {grid_size}\n"
    "- Block Dimensions: {block_size}\n"
    "- Memory Strategy: {mem_strategy}\n\n"
    "Requirements:\n"
    "- Use 'extern \"C\"' for potential compatibility with JIT compilers.\n"
    "- Include robust error handling for CUDA API calls.\n"
    "- Ensure the implementation matches the technical analysis provided.\n"
    "- Return ONLY CUDA C++ source. No prose, no Markdown fences."
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
    "- Memory Strategy: {mem_strategy}\n\n"
    "Hard requirements:\n"
    "- The output must compile with `nvcc -c`.\n"
    "- Do not reintroduce any of the errors listed in the compiler log.\n"
    "- Return ONLY CUDA C++ source. No prose, no Markdown fences."
)

_REPAIR_HUMAN = (
    "Task: {task}\n\n"
    "Technical Analysis:\n{analysis}\n\n"
    "Previous (broken) source on attempt #{attempt}:\n"
    "----- BEGIN PREVIOUS SOURCE -----\n{prev_code}\n----- END PREVIOUS SOURCE -----\n\n"
    "nvcc error log:\n"
    "----- BEGIN COMPILER LOG -----\n{prev_error}\n----- END COMPILER LOG -----\n\n"
    "Fix every error reported above. Output the full corrected kernel."
)


def generator_node(state: AgentState) -> dict:
    """
    Code Generation Stage: translates the optimization strategy and technical
    analysis into compilable CUDA C++ source.

    Reflective behavior: if a previous attempt exists in state, we run the
    repair prompt with the prior source code and the nvcc error log injected.
    This is what makes the Critic -> Generator edge an actual self-healing
    loop rather than a deterministic re-roll.
    """
    strategy = state["strategy"]
    analysis = state["analysis"]

    prev_code = state.get("source_code")
    prev_error = state.get("compiler_error")
    is_repair = bool(prev_code and prev_error)

    llm = get_llm(temperature=0.0)

    if is_repair:
        prompt = ChatPromptTemplate.from_messages([
            ("system", _REPAIR_SYSTEM),
            ("human", _REPAIR_HUMAN),
        ])
        payload = {
            "grid_size": strategy.grid_size,
            "block_size": strategy.block_size,
            "mem_strategy": strategy.memory_strategy,
            "task": state["task"],
            "analysis": analysis,
            "prev_code": prev_code,
            "prev_error": prev_error,
            "attempt": state.get("retry_count", 0),
        }
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", _FIRST_PASS_SYSTEM),
            ("human", _FIRST_PASS_HUMAN),
        ])
        payload = {
            "grid_size": strategy.grid_size,
            "block_size": strategy.block_size,
            "mem_strategy": strategy.memory_strategy,
            "task": state["task"],
            "analysis": analysis,
        }

    chain = prompt | llm
    response = chain.invoke(payload)

    cleaned = extract_cuda_source(response.content)

    return {"source_code": cleaned}
