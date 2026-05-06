from typing import TypedDict, Tuple, Optional
from pydantic import BaseModel, Field

# defined here globally so that it can be accessed both by the optimizer and the generator
class OptimizationStrategy(BaseModel):
    """
    The strict Pydantic schema used by the Optimizer node.
    """
    bottleneck_type: str = Field(description="The primary performance bottleneck, e.g., 'Global Memory Latency'")
    block_size: Tuple[int, int, int] = Field(description="Optimal CUDA block dimensions, e.g., [256, 1, 1]")
    grid_size: Tuple[int, int, int] = Field(description="Optimal CUDA grid dimensions, e.g., [1024, 1, 1]")
    memory_strategy: str = Field(description="Specific instructions for shared memory, memory coalescing, etc.")

# global agent state defined
class AgentState(TypedDict):
    """
    The global memory object passed between all LangGraph nodes.
    """
    task: str                                  # The original user request
    is_cuda: Optional[bool]                    # Set by Planner
    rag_context: Optional[str]                 # Set by Retriever/Analyzer
    analysis: Optional[str]                    # Set by Analyzer
    strategy: Optional[OptimizationStrategy]   # Set by Optimizer
    source_code: Optional[str]                 # Set by Generator
    compilation_success: Optional[bool]        # Set by Critic
    compiler_error: Optional[str]              # Set by Critic
    retry_count: int                           # Maintained by Critic