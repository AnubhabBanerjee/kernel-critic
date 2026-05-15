"""
Critic node: compile the generated CUDA source and capture diagnostic output.

Why this node matters
---------------------
A green compile (nvcc returncode == 0) only proves the code is *syntactically* valid. 
It says nothing about whether the kernel uses the GPU well. To get a cheap signal about runtime behavior 
*without actually running on a GPU*, we always pass `-Xptxas -v` to nvcc. ptxas (the PTX assembler) prints per-kernel
resource usage — registers, static shared memory, spill stores/loads — which
the Generator's repair prompt can then react to.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from src.state import AgentState


# Truncate nvcc/ptxas dumps before storing them on AgentState. ptxas can be
# chatty on multi-kernel files; keeping the log bounded prevents AgentState
# (and LLM prompts) from blowing up.
_MAX_DIAG_CHARS = 64_000

# The pair of argv tokens we want nvcc to always see. They MUST be two
# separate strings: nvcc parses `-Xptxas` and its argument independently.
_REQUIRED_PTXAS_PAIR = ("-Xptxas", "-v")


def _ensure_ptxas_verbose(flags: List[str]) -> List[str]:
    """
    Make sure ``-Xptxas -v`` appears as two consecutive elements in ``flags``.

    We scan adjacent pairs instead of doing ``"-v" in flags`` so a stray
    ``-v`` somewhere else (e.g. as another flag's value) does not silence the
    addition. If the pair is already present, we leave the list untouched —
    duplicate ``-Xptxas -v`` would just clutter the log.
    """
    out = list(flags)  # copy so we don't mutate the strategy's list
    for i in range(len(out) - 1):
        if out[i] == _REQUIRED_PTXAS_PAIR[0] and out[i + 1] == _REQUIRED_PTXAS_PAIR[1]:
            return out  # already present, nothing to do
    out.extend(_REQUIRED_PTXAS_PAIR)
    return out


def _truncate(text: str, limit: int = _MAX_DIAG_CHARS) -> str:
    """
    Keep the most informative portions of a long log.

    We slice both ends — the header (banner / first errors) and the tail
    (ptxas summary lines tend to come near the end) — and drop the middle.
    This is more useful than a hard cut at `limit`.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    head = text[:half]
    tail = text[-half:]
    omitted = len(text) - limit
    return f"{head}\n... [diagnostic truncated, {omitted} chars omitted] ...\n{tail}"


def critic_node(state: AgentState) -> dict:
    """
    Validation Stage of the agent loop.

    Steps:
      1. Write the latest source_code to a temp .cu file.
      2. Build the nvcc argv: base flags + strategy.compiler_flags
         (validated upstream) + the mandatory `-Xptxas -v` pair.
      3. Run nvcc with stderr+stdout captured.
      4. Persist BOTH:
           - compiler_error: only set on failure, drives the repair prompt's
             "fix this" path.
           - nvcc_diagnostic_log: ALWAYS set when there is any output, so the
             Generator can react to ptxas resource lines even on a green
             compile.
      5. Clean up temp files and bump retry_count.

    Note: `retry_count` is incremented even on success here. `graph.should_continue`
    only cares about retry_count when compilation failed, so this is harmless,
    but be aware if you ever read retry_count from elsewhere.
    """

    # 1. Isolated temp paths: avoid fixed names like temp_kernel.cu so concurrent
    # graph runs (e.g. multiple workers) do not overwrite each other's files.
    source_path: Optional[str] = None
    object_path: Optional[str] = None
    diag_log: Optional[str] = None
    success = False
    error_msg: Optional[str] = None

    try:
        # NamedTemporaryFile(delete=False): we close the handle, then nvcc reads
        # the path in a subprocess — delete=True would remove the file too early.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".cu",
            prefix="kernel_critic_",
            delete=False,
            encoding="utf-8",
        ) as src_file:
            src_file.write(state["source_code"])
            source_path = src_file.name

        # Pair a unique .o with the same basename (same directory as the .cu).
        object_path = str(Path(source_path).with_suffix(".o"))

        # 2. Build the nvcc command.
        strategy = state.get("strategy")
        # Strategy can be None in theory (e.g. test harnesses); guard for it
        # rather than assuming the graph always populated it.
        extra: List[str] = list(strategy.compiler_flags) if strategy is not None else []
        # Always force the diagnostic flag pair, idempotently.
        extra = _ensure_ptxas_verbose(extra)

        # argv form (NOT shell=True) keeps us safe from shell metacharacters
        # inside compiler_flags. Each element is passed as-is.
        cmd = ["nvcc", "-c", source_path, "-o", object_path, *extra]

        # 3. Run nvcc. check=False because we want to inspect returncode
        #    ourselves rather than raise on non-zero exits.
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        # 4. Build the diagnostic log. We combine stdout + stderr because
        #    ptxas's `-v` summary historically lands on stderr, but error
        #    spew can land on either depending on toolchain version.
        success = result.returncode == 0
        combined_parts = [s for s in (result.stdout, result.stderr) if s]
        combined = "\n".join(combined_parts).strip()
        diag_log = _truncate(combined) if combined else None

        # compiler_error is what the Generator's repair prompt treats as "the
        # thing to fix". We only populate it on failure so that a successful
        # compile does not look like an error to downstream code.
        error_msg = diag_log if not success else None

    except FileNotFoundError:
        # nvcc is missing entirely (common in CI/dev boxes without CUDA).
        # We fail loudly via the state so the user can debug, rather than
        # silently looping.
        success = False
        error_msg = "Critical: nvcc compiler not found in the local environment."
        diag_log = error_msg

    finally:
        # 5. Cleanup: remove temp artifacts even if nvcc fails or raises mid-run.
        for path in (source_path, object_path):
            if path and os.path.isfile(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    # Return only the keys we want to update; LangGraph merges this into the
    # rest of AgentState.
    return {
        "compilation_success": success,
        "compiler_error": error_msg,
        "nvcc_diagnostic_log": diag_log,
        "retry_count": state["retry_count"] + 1,
    }
