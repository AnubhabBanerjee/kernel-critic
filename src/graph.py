from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes.planner import planner_node
from src.nodes.analyzer import analyzer_node
from src.nodes.optimizer import optimizer_node
from src.nodes.generator import generator_node
from src.nodes.critic import critic_node

# 1. Define the Conditional Routing Logic
def should_continue(state: AgentState):
    """
    Determines if the agent should retry code generation or stop.
    """
    # If compilation succeeded, we are done.
    if state["compilation_success"]:
        return END
    
    # If it failed but we haven't hit the retry limit (e.g., 5), try again.
    if state["retry_count"] < 5:
        return "generator"
    
    # Otherwise, stop (even if it failed) to prevent infinite loops.
    return END

# 2. Initialize the Graph
workflow = StateGraph(AgentState)

# 3. Add Nodes to the Graph
workflow.add_node("planner", planner_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("optimizer", optimizer_node)
workflow.add_node("generator", generator_node)
workflow.add_node("critic", critic_node)

# 4. Define the Edges (The Flow)
workflow.set_entry_point("planner")

# Conditional Edge from Planner: Only analyze if the task is CUDA-related
workflow.add_conditional_edges(
    "planner",
    lambda state: "analyzer" if state["is_cuda"] else END
)

workflow.add_edge("analyzer", "optimizer")
workflow.add_edge("optimizer", "generator")
workflow.add_edge("generator", "critic")

# Conditional Edge from Critic: The Loopback Mechanism
workflow.add_conditional_edges(
    "critic",
    should_continue
)

# 5. Compile the Graph
app = workflow.compile()