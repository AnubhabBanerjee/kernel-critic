import os
from dotenv import load_dotenv
from src.graph import app
import sys
import requests

# 1. Centralized Environment Loading
# This ensures every sub-module in src/ can access API keys.
load_dotenv()

def generate_runtime_viz():
    """
    Generates and saves a PNG of the graph architecture.
    This provides an on-demand audit of the agent's logic.
    """
    try:
        # Extract the graph binary from the compiled LangGraph app
        png_data = app.get_graph().draw_mermaid_png()
        with open("task_execution_flow.png", "wb") as f:
            f.write(png_data)
        print("✅ Visualization successfully saved as 'task_execution_flow.png'.")
    except Exception as e:
        print(f"❌ Visualization failed: {e}. (Ensure mermaid dependencies are installed.)")

def main():
    """
    User-facing interface for the KernelCritic agent.
    """
    # 1. Environment Routing
    active_env = os.getenv("ACTIVE_ENV", "local").lower()

    # 2. Pre-flight Validation Checks
    print(f"🔍 Initializing KernelCritic (Mode: {active_env})...")

    # --- OpenAI Cloud Mode ---
    if active_env == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("❌ Error: ACTIVE_ENV is 'openai' but OPENAI_API_KEY is missing from .env")

    # --- Internal Proxy Mode ---
    elif active_env == "internal":
        required = ["INTERNAL_API_KEY", "INTERNAL_BASE_URL"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            sys.exit(f"❌ Error: Internal environment missing required keys: {', '.join(missing)}")
        print(f"✅ Secure gateway verified: {os.getenv('INTERNAL_BASE_URL')}")

    # --- Local Hardware Mode (Ollama/vLLM) ---
    elif active_env == "local":
        base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
        # Health check: Strip '/v1' to ping the root server
        health_check_url = base_url.replace('/v1', '').rstrip('/')
        
        try:
            # Ping the local engine with a 3-second timeout
            response = requests.get(health_check_url, timeout=3)
            # Accept success or redirects; some local engines vary on root response
            if response.status_code >= 500:
                raise ConnectionError
            print(f"✅ Local inference engine detected at {base_url}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(f"❌ Error: Could not connect to local engine at {base_url}")
            print("👉 Check if Ollama ('ollama serve') or vLLM is running on your machine.")
            sys.exit(1)
    
    else:
        sys.exit(f"❌ Error: Unknown ACTIVE_ENV '{active_env}'. Expected: [local, openai, internal]")
    
    # 3. Interactive Loop
    print("--- KernelCritic: CUDA Optimization Agent ---\n")
    user_task = input("\nDescribe the CUDA kernel or optimization task: ")

    initial_state = {
        "task": user_task,
        "retry_count": 0,
        "compilation_success": False
    }

    print("\n[KernelCritic is thinking...]\n")

    # 2. Initializing the Global State
    # We populate the required fields to start the graph.
    initial_state = {
        "task": user_task,
        "retry_count": 0,
        "compilation_success": False
    }

    # 3. Invoking the Agentic Workflow
    # This runs the Planner -> Analyzer -> Optimizer -> Generator -> Critic.
    print("\n[KernelCritic is thinking...]")
    final_output = app.invoke(initial_state)

    # 4. Final Result Presentation
    if final_output["compilation_success"]:
        print("\n✅ Optimization Successful. Compiled Source Code:")
        print("-" * 40)
        print(final_output["source_code"])
        print("-" * 40)

        # 5. Dynamic Visualization Logic
        # Offers the user a chance to audit the execution flow on demand.
        viz_choice = input("\nDo you want to get a visualization of how your task was executed? (y/n): ").lower()
        if viz_choice == 'y':
            generate_runtime_viz()
    else:
        print("\n❌ Optimization Failed after 5 retries.")
        print(f"Last Compiler Error: {final_output.get('compiler_error')}")

if __name__ == "__main__":
    main()