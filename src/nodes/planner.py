"""
The goal of KernelCritic is to act as an autonomous, fault-tolerant software engineering agent. The Planner serves as the system's Edge Gateway and Resource Guard.

1. Compute Conservation (Defensive Engineering)
GPU time is an expensive resource, particularly if the agent eventually scales on an enterprise cluster. If a user inputs a standard Python question or absolute garbage text, you do not want the system to execute vector database searches, extensive RAG generation, and nvcc subprocess invocations. The Planner short-circuits the graph immediately for irrelevant queries, saving heavy compute.

2. Deterministic Graph Routing
Agentic workflows fail when control-flow logic relies on parsing conversational text. If an LLM replies, "Yes, I believe this is a CUDA issue," writing a standard regex to parse that intent is fragile. By forcing the LLM to output a strict True or False boolean via Pydantic, LangGraph's conditional edges have a mathematically definitive binary flag to evaluate. This ensures the graph routes perfectly every single time.

3. Architectural Separation of Concerns
Instead of asking one massive LLM prompt to "Decide if this is CUDA, analyze it, and write the code," the Planner handles only classification. This modularity makes the system easier to test, benchmark, and debug.
"""
# Pydantic to enforce data validation. BaseModel creates the schema, and Field allows you to attach instructions directly to the variables.
from pydantic import BaseModel, Field
# LangChain's tool for separating the system persona instructions from the user's raw input.
from langchain_core.prompts import ChatPromptTemplate
# Imports the typed dictionary that defines the data passing between LangGraph nodes.
from src.state import AgentState
# Imports custom factory function to dynamically load the LLM without hardcoding the API endpoints.
from src.config import get_llm


# this class defines the precise data structure the LLM must generate
class PlannerDecision(BaseModel):
    # The description string acts as an internal prompt, explicitly telling the model the criteria for returning True or False.
    is_cuda: bool = Field(description="True if the prompt requires CUDA C++ or GPU performance optimization, False otherwise.")

# It reads the current global state and outputs a dictionary of updates.
def planner_node(state: AgentState) -> dict:
    # Instantiates the LLM client. temperature=0 is critical here; it removes all randomness to guarantee deterministic routing decisions.
    llm = get_llm(temperature=0)
    # Modifies the LLM client to reject text generation and strictly output a JSON object matching the PlannerDecision schema.
    structured_llm = llm.with_structured_output(PlannerDecision)
    
    # The "system" message assigns the role, and the "human" message dynamically injects the task variable from the current state.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a routing assistant. Determine if the following task requires CUDA C++ or GPU performance optimization."),
        ("human", "{task}")
    ])
    
    # pipe the formatted prompt directly into the schema-enforced LLM.
    chain = prompt | structured_llm
    # Triggers the API call. It evaluates the user's input and returns a validated PlannerDecision object.
    result = chain.invoke({"task": state["task"]})
    
    # Returns the boolean value mapped to the "is_cuda" key
    return {"is_cuda": result.is_cuda}