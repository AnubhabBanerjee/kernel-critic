from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from src.config import get_llm
from src.tools.retriever import get_relevant_cuda_docs

def analyzer_node(state: AgentState) -> dict:
    """
    Technical Analysis Stage:
    Performs a deep-dive analysis of the CUDA task by grounding the request 
    in official documentation to identify architectural constraints.
    """
    
    # 1. Hardware-Aware Retrieval
    # We call the tool we just finished to get the "Ground Truth"
    docs_context = get_relevant_cuda_docs(state["task"])
    
    # 2. Initialize the LLM
    # We use a slightly higher temperature (0.1) to allow for 
    # sophisticated technical reasoning and pattern recognition.
    llm = get_llm(temperature=0.1)
    
    # 3. Expert-Level Prompting
    # This prompt forces the model to think like a Senior Performance Engineer
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior NVIDIA CUDA Architect. Your goal is to analyze code for "
            "maximum hardware utilization. Focus on:\n"
            "- Memory Hierarchy: Global memory coalescing and shared memory bank conflicts.\n"
            "- Execution Configuration: Warp divergence and occupancy metrics.\n"
            "- Compute Throughput: ILP (Instruction Level Parallelism) and loop unrolling.\n\n"
            "Use the following documentation context to ground your analysis:\n{context}"
        )),
        ("human", "Analyze the following task for CUDA implementation or optimization: {task}")
    ])
    
    # 4. Execution
    chain = prompt | llm
    response = chain.invoke({
        "context": docs_context,
        "task": state["task"]
    })
    
    # 5. State Update
    # Based on the AgentState image, we fill both fields here.
    return {
        "rag_context": docs_context,
        "analysis": response.content
    }