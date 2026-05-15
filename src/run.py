"""
CLI entrypoint for KernelCritic.

Responsibilities:
  1. Load environment (.env), pick a runtime mode (openai / internal / local).
  2. Run pre-flight checks so we fail fast instead of mid-graph.
  3. Build the initial AgentState and invoke the LangGraph app.
  4. Present the final source code and (when available) the ptxas diagnostic.
"""

import os
import sys

import requests
from dotenv import load_dotenv

from src.graph import app


# Load .env exactly once at import time so every sub-module under src/ sees
# the same environment regardless of import order.
load_dotenv()


def generate_runtime_viz():
    """
    Render the compiled LangGraph as a PNG for human inspection.

    This is optional; we hide it behind a y/n prompt at the end of a
    successful run so day-to-day usage stays terminal-only.
    """
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("task_execution_flow.png", "wb") as f:
            f.write(png_data)
        print("✅ Visualization successfully saved as 'task_execution_flow.png'.")
    except Exception as e:
        # Mermaid rendering depends on optional system packages; degrade
        # gracefully instead of crashing the whole run.
        print(f"❌ Visualization failed: {e}. (Ensure mermaid dependencies are installed.)")


def main():
    """Interactive entrypoint. One question in, one kernel out."""

    # 1. Pick the active runtime. Default to 'local' so a developer with
    #    Ollama running locally can `python run.py` without any env vars.
    active_env = os.getenv("ACTIVE_ENV", "local").lower()
    print(f"🔍 Initializing KernelCritic (Mode: {active_env})...")

    # 2. Mode-specific pre-flight checks. These short-circuit before we
    #    pay for any LLM call.
    if active_env == "openai":
        # Hosted OpenAI — we only need an API key.
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("❌ Error: ACTIVE_ENV is 'openai' but OPENAI_API_KEY is missing from .env")

    elif active_env == "internal":
        # Internal proxy/gateway — needs both URL and key.
        required = ["INTERNAL_API_KEY", "INTERNAL_BASE_URL"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            sys.exit(f"❌ Error: Internal environment missing required keys: {', '.join(missing)}")
        print(f"✅ Secure gateway verified: {os.getenv('INTERNAL_BASE_URL')}")

    elif active_env == "local":
        # Local Ollama / vLLM. We do a TCP health check so the user gets a
        # clear "is your server running?" message instead of a cryptic
        # LangChain timeout deep in the graph.
        base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
        # Strip '/v1' so we ping the server root, which most engines expose.
        health_check_url = base_url.replace('/v1', '').rstrip('/')
        try:
            response = requests.get(health_check_url, timeout=3)
            # 2xx and 3xx are fine; some local engines redirect on root.
            if response.status_code >= 500:
                raise ConnectionError
            print(f"✅ Local inference engine detected at {base_url}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(f"❌ Error: Could not connect to local engine at {base_url}")
            print("👉 Check if Ollama ('ollama serve') or vLLM is running on your machine.")
            sys.exit(1)

    else:
        sys.exit(f"❌ Error: Unknown ACTIVE_ENV '{active_env}'. Expected: [local, openai, internal]")

    # 3. Interactive prompt.
    print("--- KernelCritic: CUDA Optimization Agent ---\n")
    user_task = input("\nDescribe the CUDA kernel or optimization task: ")

    # 4. Build the initial AgentState. We only seed the keys whose absence
    #    would crash a node; the rest are filled in by planner/analyzer/etc.
    #    Note: nvcc_diagnostic_log starts as None and is set by the critic on
    #    every compile (success or failure).
    initial_state = {
        "task": user_task,
        "retry_count": 0,
        "compilation_success": False,
        "nvcc_diagnostic_log": None,
    }

    print("\n[KernelCritic is thinking...]\n")

    # 5. Run the graph end-to-end. `app.invoke` blocks until the graph
    #    reaches END (success, max retries, or non-CUDA short-circuit).
    final_output = app.invoke(initial_state)

    # 6. Present results to the user.
    if final_output["compilation_success"]:
        print("\n✅ Optimization Successful. Compiled Source Code:")
        print("-" * 40)
        print(final_output["source_code"])
        print("-" * 40)

        # If the critic captured ptxas resource usage, show it. It is one of
        # the few cheap signals about *quality* (registers / smem / spills),
        # not just correctness.
        diag = final_output.get("nvcc_diagnostic_log")
        if diag:
            print("\n--- nvcc / ptxas diagnostic ---")
            print(diag)
            print("-" * 40)

        # Optional architectural audit.
        viz_choice = input("\nDo you want to get a visualization of how your task was executed? (y/n): ").lower()
        if viz_choice == 'y':
            generate_runtime_viz()
    else:
        # Either nvcc kept failing for 5 attempts, or the planner short-
        # circuited a non-CUDA request. Surface the latest compiler log so
        # the user can see what went wrong.
        print("\n❌ Optimization Failed after 5 retries.")
        print(f"Last Compiler Error: {final_output.get('compiler_error')}")


if __name__ == "__main__":
    main()