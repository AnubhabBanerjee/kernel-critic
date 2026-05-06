**Copyright (c) 2026 Anubhab Banerjee (AnubhabBanerjee/kernel-critic)**
**All rights reserved. No part of this repository may be used, redistributed, or modified in any form or by any means without the prior written permission of the author.**

---

# 🚀 KernelCritic: Agentic CUDA Optimization

> Self-healing CUDA codegen with an `nvcc`-grounded reflective loop.

**KernelCritic** is an AI-driven orchestration framework that automates the generation and validation of high-performance CUDA kernels. It uses a **Reflective Agent** architecture built on LangGraph: every kernel is planned, RAG-grounded against official NVIDIA documentation, generated, and then gated through the real `nvcc` compiler. When compilation fails, the error log is fed back into the generator so the next attempt is informed by the previous failure — not a deterministic re-roll.

## 🧠 System Architecture

KernelCritic operates as a state-aware graph with a feedback edge from validation back to generation:

1. **Planner** — Classifies the task. Non-CUDA prompts are short-circuited so the system doesn't waste GPU/LLM cycles on irrelevant queries. Uses Pydantic structured output for deterministic boolean routing.
2. **Analyzer** — RAG-grounds the task against a local ChromaDB index of NVIDIA documentation, then produces a senior-engineer-level analysis focused on memory hierarchy, warp divergence, and ILP.
3. **Optimizer** — Converts the analysis into a strict `OptimizationStrategy` (block size, grid size, memory plan) via Pydantic schema enforcement. No free-form text — the downstream nodes consume typed fields.
4. **Generator** — Synthesizes CUDA C++. On a retry, it receives the previous (broken) source and the full `nvcc` error log, and is instructed to fix every reported error. Output is post-processed to strip Markdown fences before reaching the compiler.
5. **Critic (the compile gate)** — Invokes `nvcc -c` on the generated source. On success, the loop terminates. On failure, `compiler_error` and the source are written back into shared state and the graph re-enters the Generator. Bounded by a retry budget to prevent infinite loops.

The graph is **not** a DAG: the `Critic → Generator` feedback edge is intentional and is what makes the system "self-healing" rather than one-shot.

You can dump the live graph as a PNG after a successful run:

![KernelCritic execution flow](task_execution_flow.png)

## 🛠️ Multi-Backend Infrastructure

Designed for portability across local, corporate, and cloud environments via a single `ACTIVE_ENV` switch:

* **Local** — OpenAI-compatible local inference (vLLM, Ollama, LM Studio) on consumer GPGPU hardware. Tested with a DeepSeek-R1 distill (e.g. `DeepSeek-R1-Distill-Qwen-7B`) on an 8 GB-class card; the full DeepSeek-R1 requires a much larger setup.
* **Internal** — Routes through a corporate OpenAI-compatible gateway (LiteLLM, vLLM proxy, etc.). Chat and embeddings endpoints are configured independently.
* **OpenAI Cloud** — Direct integration with OpenAI's flagship chat and embedding models for high-reasoning tasks.

All three tiers share the same agent code. Switching backends is a single `.env` change; no code edits.

## ✅ Prerequisites

* Python 3.10+
* **NVIDIA CUDA Toolkit** with `nvcc` on `PATH` — required by the Critic node.
  Verify with `nvcc --version`. Without it, the agent will return a graceful error and exit.
* For `local` mode: a running OpenAI-compatible inference server (vLLM, Ollama, or LM Studio).

## ⚙️ Installation

```bash
git clone <your-repo-url> kernel-critic
cd kernel-critic

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — pick ACTIVE_ENV and fill in the matching tier.
```

## 📂 Knowledge Base Setup (BYOD)

To respect upstream IP, this repository does **not** redistribute proprietary documentation.

1. Drop your own NVIDIA references (e.g. *CUDA C++ Best Practices Guide*, *PTX ISA*, vendor SM specs) into `data/` as PDFs.
2. Build the local vector index:

```bash
python3 ingest.py
```

The pipeline chunks the PDFs, embeds them via the embeddings backend you selected in `.env`, and persists a ChromaDB index to `./db`. The Analyzer node retrieves the top-k most relevant chunks per query.

## 🚀 Execution

Run the interactive agent — it performs pre-flight connectivity checks against the active backend before invoking the graph:

```bash
python3 run.py
```

You'll be prompted for a CUDA task. See [`examples.txt`](examples.txt) for five test prompts spanning baseline kernels, shared-memory tiling, warp-level reductions, occupancy tuning, and constant-memory optimization.

## 🎬 Example Run

```text
$ python3 run.py
Initializing KernelCritic (Mode: local)...
Local inference engine detected at http://localhost:8000/v1
--- KernelCritic: CUDA Optimization Agent ---

Describe the CUDA kernel or optimization task: Write a tiled matrix multiply
using shared memory, handle non-multiple-of-tile dimensions.

[KernelCritic is thinking...]
  planner   -> is_cuda=True
  analyzer  -> retrieved 3 doc chunks
  optimizer -> block=(16,16,1)  grid=(N/16,N/16,1)  memory=shared-tile
  generator -> 412 lines
  critic    -> nvcc -c FAIL: identifier "TILE_SIZE" is undefined
  generator -> repairing with compiler log...
  critic    -> nvcc -c OK

Optimization Successful. Compiled Source Code:
----------------------------------------
extern "C" __global__ void matmul_tiled(...) { ... }
----------------------------------------
Do you want to get a visualization of how your task was executed? (y/n):
```

## 📁 Project Layout

```text
kernel-critic/
├── run.py                 # CLI entry point + pre-flight checks
├── ingest.py              # Build the ChromaDB index from data/
├── requirements.txt
├── .env.example           # Reference configuration
├── data/                  # Drop your NVIDIA PDFs here (gitignored)
├── db/                    # Persisted ChromaDB index (gitignored)
└── src/
    ├── graph.py           # LangGraph wiring + retry edge
    ├── state.py           # Typed AgentState + OptimizationStrategy schema
    ├── config.py          # Tier-aware LLM factory
    ├── nodes/
    │   ├── planner.py     # CUDA / non-CUDA classifier
    │   ├── analyzer.py    # RAG-grounded technical analysis
    │   ├── optimizer.py   # Pydantic-enforced strategy
    │   ├── generator.py   # Reflective code synthesis (repair-aware)
    │   └── critic.py      # nvcc compile gate
    └── tools/
        └── retriever.py   # ChromaDB similarity search
```

## 🛣️ Roadmap (Coming in the Next Release)

Active work-in-progress items already on the bench:

* **Correctness + performance gate.** Extending the Critic beyond `nvcc -c` with a host-side correctness harness and an `ncu`-driven performance pass, so a kernel must be both numerically correct and meet a roofline target before the loop terminates.
* **Strategy-aware retries.** Routing persistent failures back to the Optimizer (not just the Generator), so a bad block/grid choice can be re-derived rather than patched around.
* **Sandboxed compilation.** Moving the `nvcc` invocation into an isolated `tempfile`/container sandbox so the agent can be safely run on untrusted inputs and in parallel.
* **Tool-using sub-agents.** Adding live SM-spec lookups and `cuobjdump` / `ncu` telemetry as agent tools, so optimization decisions are grounded in measured behavior, not just static documentation.
* **Reproducible benchmark suite.** Promoting `examples.txt` into a proper eval harness reporting pass-rate, retries-to-success, and roofline utilization across a fixed set of kernels.

## 🙏 Acknowledgments

Built on [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [ChromaDB](https://github.com/chroma-core/chroma). NVIDIA documentation is the property of NVIDIA Corporation and is not redistributed by this project.
