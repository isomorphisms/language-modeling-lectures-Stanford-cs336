# Systems and scaling: making training and inference fit

The same model can be cheap or impractical depending on sequence length, accelerator memory, network topology, and how its work is divided. These lectures treat systems choices as part of the model design rather than as a final deployment detail.

## 04 — Attention alternatives and mixture-of-experts

Source: [lecture_04.pdf](../lecture_04.pdf)

Standard attention forms scores between many pairs of positions, so its attention matrix grows quadratically with sequence length. Linear-attention and sparse-attention proposals change that computation by restricting or reorganizing which positions interact. Their value depends on more than asymptotic notation: a theoretically smaller computation can still be slower if it prevents efficient dense kernels or has poor memory access.

Mixture-of-experts (MoE) changes the feed-forward part of a Transformer. A router selects a small number of experts for each token, so a model can hold many parameters while activating only a fraction per token. The core trade is between conditional capacity and new systems problems. Routing must avoid sending too many tokens to one expert, token dispatch creates communication, and an expert's nominal parameters do not equal the active FLOPs of a token.

The lecture uses modern designs, including DeepSeek variants, to show that architectural changes interact. Fine-grained or shared experts modify routing behavior; multi-head latent attention (MLA) reorganizes attention state; and multi-token prediction (MTP) changes the training objective. Evaluate these as a coupled recipe with clear active-parameter, total-parameter, memory, and communication accounting.

## 05 — GPU systems and memory traffic

Source: [lecture_05.pdf](../lecture_05.pdf)

GPUs obtain high throughput by running many lightweight threads while hiding latency with other ready work. A kernel must make useful work available in this hierarchy of threads, warps, blocks, registers, shared memory, and global memory. The lecture pairs that hierarchy with the roofline model: a kernel may be limited by arithmetic units or by moving values to and from memory.

Coalesced accesses let neighboring threads read nearby addresses efficiently. Tiling loads a reusable piece of a larger operand into fast storage, then performs several operations with it before returning to global memory. Mixed precision can increase arithmetic throughput and reduce storage, but it changes rounding behavior and requires deliberate treatment of numerically sensitive reductions and updates. Divergent control flow wastes parallel lanes when threads in a warp follow different branches.

Training memory must cover more than the parameters. The lecture separates parameter storage, gradients, optimizer state, and activations, then relates these quantities to sequence length and batch size. That ledger is the practical starting point for selecting model shapes and parallelism strategies.

## 06 — Kernels and performance measurement

Source: [lecture_06.py](../lecture_06.py)

Lecture 6 moves from hardware principles to executable kernels. The examples compare naïve, framework-provided, and compiled GELU implementations; they then build Triton versions of GELU, softmax, row summation, and matrix multiplication followed by ReLU. The sequence matters: first establish a correct reference, then measure, then change the implementation.

Kernel fusion can avoid writing an intermediate tensor to global memory only to read it again in the next operation. Softmax shows why numerical reasoning cannot be separated from speed: stable softmax subtracts a row maximum before exponentiating, which prevents avoidable overflow while preserving the probability distribution. Tiled matrix multiplication illustrates reuse; the best tile size depends on shape, hardware resources, and register/shared-memory pressure.

Timing needs warm-up, repeated trials, and synchronization because accelerator launches are asynchronous. Profiling should answer a specific question such as “which operator dominates step time?” rather than merely produce an attractive trace. The lecture's equality checks reinforce the right order: an optimization is useful only after it agrees with the reference within an appropriate tolerance.

## 07 — Collective communication and basic parallelism

Source: [lecture_07.py](../lecture_07.py)

Multiple accelerators need collective operations to keep their views of training consistent. All-reduce combines values across ranks and gives every rank the result. Reduce-scatter combines values and gives each rank only its shard. All-gather assembles shards so every rank can reconstruct the whole. These operations have different byte movement and synchronization patterns; choosing one is part of the algorithm.

Data parallelism puts a copy of the model on each rank and splits examples across ranks. Each rank computes local gradients, then an all-reduce makes the update agree. Tensor parallelism splits a layer itself across ranks, introducing communication inside the layer. Pipeline parallelism gives consecutive layer ranges to different stages and uses microbatches to keep stages busy. The lecture's distributed examples make the rank, world size, initialization, and synchronization explicit, which helps prevent vague claims that a model “uses several GPUs.”

## 08 — Training parallelism strategies

Source: [lecture_08.pdf](../lecture_08.pdf)

The parallelism lecture expands the previous patterns into a design space. Data parallelism is often the base. Fully sharded approaches partition parameters, gradients, and optimizer state rather than replicating all of them. Tensor parallelism is valuable within a fast interconnect domain. Pipeline parallelism extends capacity across stages but introduces idle bubbles unless the microbatch schedule is chosen carefully. Expert parallelism distributes MoE experts, and context or sequence parallelism attacks memory and work associated with long sequences.

There is no universally best decomposition. The useful question is: which tensor is too large or too expensive, and which communication path is available? The answer should include the topology: communication within one node and communication across nodes can have very different costs. Combining methods increases capacity but also compounds scheduling, failure, and debugging complexity.

## 09 — Scaling laws

Source: [lecture_09.pdf](../lecture_09.pdf)

Scaling laws are empirical descriptions of how a specified family of training runs behaves as compute, parameter count, data, or another resource increases. They are tools for interpolation and careful extrapolation, not guarantees that all capabilities grow smoothly. A fitted loss curve can guide a budget decision only when its data distribution, architecture family, optimizer recipe, and evaluation protocol remain relevant.

The lecture studies the trade between model size, token count, and compute. IsoFLOP-style comparisons hold a compute budget fixed and ask how to divide it between a larger model trained on fewer tokens and a smaller model trained on more. The answer depends on the training setup and changes with data quality and model architecture. MoE complicates naive parameter-count stories because total parameters and active compute per token diverge.

## 10 — Inference systems

Source: [lecture_10.py](../lecture_10.py)

Inference has two distinct phases. **Prefill** processes a prompt and writes key-value (KV) states for its tokens. **Decode** produces later tokens one at a time, repeatedly reading the growing KV cache. The first phase can use substantial parallel work across prompt positions; the second is often limited by memory movement and per-token latency. Treating them as one average workload hides the bottleneck.

The KV cache avoids recomputing prior keys and values, but it consumes memory proportional to sequence length, layers, batch size, and key/value representation. GQA reduces that cache by sharing key/value heads. Quantization reduces representation precision to relieve memory and sometimes bandwidth pressure, while requiring accuracy checks. The lecture also covers pruning, speculative sampling, continuous batching, and paged attention. Each one changes a different constraint: speculation tries to accept several future tokens after a cheap draft; continuous batching keeps the device busy as requests arrive and finish at different times; paged attention organizes cache memory in blocks to reduce fragmentation and simplify request growth.

Report latency, throughput, prompt length, generated length, batch policy, hardware, and quality together. A throughput number without workload shape is not an inference result.

## 11 — Training and scaling recipes

Source: [lecture_11.pdf](../lecture_11.pdf)

Large training runs magnify small recipe errors. Initialization, optimizer choice, learning-rate schedule, global batch size, gradient clipping, precision, and data mixture all affect whether a run is stable and whether it reaches the expected loss. The lecture discusses warmup–stable–decay (WSD) schedules and scaling studies as ways to make these choices observable rather than inherited by folklore.

An IsoFLOP study is useful only if configurations are comparable: matched implementation quality, enough training to observe the relevant behavior, and a stated held-out metric. MoE introduces additional questions about router loss, expert load, and active versus total computation. The examples of scaling blow-ups are a warning against assuming that an apparently modest extension is safe; monitor the actual numerical and optimization signals that precede failure.

## Systems checklist

- Start from a per-step ledger: FLOPs, bytes moved, activation memory, parameter/optimizer memory, and communication bytes.
- Benchmark a full representative workload after measuring an isolated kernel. The dominant operation can move when the workload changes.
- Name the parallelism axis and the collective it requires; “distributed” is not a performance model.
- For scale studies, preserve configuration, data mixture, tokenizer, software version, and evaluation set alongside every curve.
- Separate total parameters, active parameters, and inference-cache size whenever MoE or GQA is involved.
