from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState, OptimizationStrategy
from src.config import get_llm

def optimizer_node(state: AgentState) -> dict:
    """
    Optimization Strategy Stage: Converts technical analysis into a structured hardware configuration 
    using Pydantic for schema enforcement.
    """
    
    # 1. Initialize the LLM with deterministic settings
    # We use temperature 0.0 because optimization requires mathematical precision, not creative variety.
    llm = get_llm(temperature=0.0)
    
    # 2. Bind the LLM to a specific Pydantic schema. This forces the model to return an OptimizationStrategy object 
    # rather than unstructured text.
    structured_llm = llm.with_structured_output(OptimizationStrategy)
    
    # 3. Define the Expert Strategy Prompt
    # This focuses on hardware-level constraints like power-of-two block sizes. The prompt provides the Technical Analysis and RAG Context together. This forces the optimizer 
    # to align its "decisions" with both the specific user problem and the official NVIDIA best practices
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a CUDA Performance Engineer. Produce a single optimization strategy object "
            "that matches the required JSON schema exactly.\n\n"
            "Follow this checklist internally before you finalize every field:\n"
            "1) Bottleneck: set bottleneck_type to exactly one of: MemoryBound, ComputeBound, "
            "InstructionBound, SharedMemoryConflict, Other — consistent with the analysis.\n"
            "2) Launch: choose grid_size and block_size (each 3 integers) that favor occupancy and "
            "coalesced access for this task; prefer conventional shapes (e.g., x dimension "
            "multiples of 32 warps) unless the analysis demands otherwise.\n"
            "3) Registers / SRAM / vectors: set registers_per_thread, shared_memory_bytes, and "
            "vectorization_width (1=scalar, 2=float2-style, 4=float4-style) so they agree with "
            "the same launch and memory plan (no contradictory budgets).\n"
            "4) Memory strategy object (memory_strategy): set USE_SHARED_MEMORY, USE_READ_ONLY_CACHE, "
            "TILING_FACTOR as a pair of ints, and PADDING_REQUIRED so they align with "
            "shared_memory_bytes and the tiling story in the analysis.\n"
            "5) Precision: set precision to exactly one schema enum value among FP32, FP16, BF16, "
            "TF32, FP64, INT32, INT16, INT8 — match the dominant compute/storage types implied by "
            "the task (including Tensor-Core-friendly paths when the math is matrix-like and the "
            "analysis supports reduced precision).\n"
            "6) Compiler flags: set compiler_flags as a JSON array of separate nvcc flag tokens "
            "(e.g. [\"-use_fast_math\"] or [\"--ptxas-options=-v\"]); use only flags you would "
            "reasonably pass to nvcc for this kernel, not shell metacharacters or unrelated tools.\n\n"
            "Ground your choices in the provided Technical Analysis, Documentation Context, and "
            "Original Task. Prefer internally consistent numbers across all fields."
        )),
        ("human", (
            "Technical Analysis: {analysis}\n\n"
            "Documentation Context: {context}\n\n"
            "Original Task: {task}"
        )),
    ])
    
    # 4. Construct and invoke the chain
    chain = prompt | structured_llm
    strategy = chain.invoke({
        "analysis": state["analysis"],
        "context": state["rag_context"],
        "task": state["task"]
    })
    
    # 5. State Update
    # Populates the 'strategy' field identified in the AgentState diagram.
    return {"strategy": strategy}