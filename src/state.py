from typing import TypedDict, Tuple, Optional, List
from pydantic import BaseModel, Field
from enum import Enum

class BottleneckType(Enum):
    """
    The type of performance bottleneck.
    """
    MEMORY_BOUND = "MemoryBound"
    COMPUTE_BOUND = "ComputeBound"
    INSTRUCTION_BOUND = "InstructionBound"
    SHARED_MEMORY_CONFLICT = "SharedMemoryConflict"
    OTHER = "Other"

class MemoryConfig(BaseModel):
    """
    The memory configuration for the optimization strategy.
    """
    USE_SHARED_MEMORY: bool
    USE_READ_ONLY_CACHE: bool
    TILING_FACTOR: Tuple[int,int]
    PADDING_REQUIRED: bool


class PrecisionType(Enum):
    """
    Numeric precision / math path for the kernel (Tensor Core eligibility depends on GPU + ops)
    """

    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    TF32 = "TF32"
    FP64 = "FP64"
    INT32 = "INT32"
    INT16 = "INT16"
    INT8 = "INT8"


# defined here globally so that it can be accessed both by the optimizer and the generator
class OptimizationStrategy(BaseModel):
    """
    The strict Pydantic schema used by the Optimizer node.
    """
    bottleneck_type: BottleneckType = Field(description="One of: MemoryBound, ComputeBound, InstructionBound, SharedMemoryConflict, Other.")
    block_size: Tuple[int, int, int] = Field(description="Optimal CUDA block dimensions, e.g., [256, 1, 1]")
    grid_size: Tuple[int, int, int] = Field(description="Optimal CUDA grid dimensions, e.g., [1024, 1, 1]")
    memory_strategy: MemoryConfig = Field(description="Specific instructions for shared memory, memory coalescing, etc.")
    registers_per_thread: int = Field(description="Registers per thread to target for occupancy and to limit register spilling (maxrregcount-style guidance).")
    shared_memory_bytes: int = Field(description="Shared memory to use per block, in bytes (static shared memory budget).")
    vectorization_width: int = Field(description="Memory vectorization width: 1 = scalar, 2 = float2-style, 4 = float4-style coalesced global access.")
    precision: PrecisionType = Field(description="Precision of the kernel: 'fp32', 'fp16', 'bf16', etc.")
    compiler_flags: List[str] = Field(description="Compiler flags to use for the kernel, e.g., -use_fast_math, -Xptxas -v.")

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
    nvcc_diagnostic_log: Optional[str]         # set by Critic after every nvcc run
    retry_count: int                           # Maintained by Critic