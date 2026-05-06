import subprocess
import os
from src.state import AgentState

def critic_node(state: AgentState) -> dict:
    """
    Validation Stage: Attempts to compile the generated CUDA source code using the NVIDIA 
    compiler (nvcc) and logs errors for the next iterative loop.
    """
    
    # 1. Setup Temporary File
    # We write the source_code from the state to a physical file for nvcc to read.
    source_path = "temp_kernel.cu"
    with open(source_path, "w") as f:
        f.write(state["source_code"])
    
    # 2. Execution of nvcc
    # We attempt to compile the code without linking (-c) to check for 
    # syntax and hardware-alignment errors.
    try:
        result = subprocess.run(
            ["nvcc", "-c", source_path, "-o", "temp_kernel.o"],
            capture_output=True,
            text=True,
            check=False # We handle the return code manually
        )
        
        # 3. Analyze Results
        # If returncode is 0, the kernel is valid and ready for execution.
        success = result.returncode == 0
        error_msg = result.stderr if not success else None
        
    except FileNotFoundError:
        # Graceful fallback if nvcc is not installed in the environment
        success = False
        error_msg = "Critical: nvcc compiler not found in the local environment."
    
    # 4. Clean up
    if os.path.exists(source_path): os.remove(source_path)
    if os.path.exists("temp_kernel.o"): os.remove("temp_kernel.o")

    # 5. State Update
    # We increment the retry_count and update success/error fields.
    return {
        "compilation_success": success,
        "compiler_error": error_msg,
        "retry_count": state["retry_count"] + 1
    }