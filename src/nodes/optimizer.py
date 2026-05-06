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
            "You are a CUDA Performance Engineer. Define a concrete optimization strategy "
            "based on the provided analysis. Ensure grid and block dimensions "
            "maximize SM occupancy and memory coalescing."
        )),
        ("human", (
            "Technical Analysis: {analysis}\n\n"
            "Documentation Context: {context}\n\n"
            "Original Task: {task}"
        ))
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