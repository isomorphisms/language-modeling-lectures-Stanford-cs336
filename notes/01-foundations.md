# Foundations: text, tensors, and the Transformer

This section covers the first question in language modeling: what concrete computation is trained when a system is asked to predict text? The answer begins before the neural network. Text is converted into tokens, tokens become tensors, tensors pass through a Transformer, and gradients update its parameters.

## 01 — Course map and tokenization

Source: [lecture_01.py](../lecture_01.py)

An autoregressive language model repeatedly answers a simple question: given the tokens already seen, which token should come next? That description is compact, but the choice of tokens changes the problem. A tokenizer is the interface between a string and the model's finite vocabulary. It determines the length of a prompt, the units at which probabilities are assigned, and the sequences for which the model spends its context window.

The lecture builds tokenizers from increasingly general representations: characters, UTF-8 bytes, words, and byte-pair encoding (BPE). BPE starts with a base vocabulary and repeatedly merges a frequent adjacent pair. The procedure is easy to state, but its implementation needs a deterministic policy for ties and a clear convention for applying learned merges at inference time. The `train_bpe` and `count_adjacent_pairs` functions make that mechanism inspectable instead of treating a tokenizer as a black box.

The compression ratio is useful because it makes a tokenizer's consequence visible: fewer tokens per string can make a fixed context window cover more text. It is not a complete quality measure. Compression can vary sharply by language, script, spelling, code, whitespace, or uncommon names. A tokenizer choice therefore belongs in both model evaluation and data analysis.

**Checks to keep:**

- Tokenize the same short examples with the candidate tokenizer and inspect the pieces, not only the token count.
- Test encode/decode round trips on Unicode, whitespace, and code-like text.
- Record the tokenizer version with checkpoints and evaluation results. A checkpoint without its tokenizer cannot be interpreted reliably.

## 02 — Tensors, gradients, and the training loop

Source: [lecture_02.py](../lecture_02.py)

The model only sees tensors: typed, shaped blocks of numbers placed in CPU or accelerator memory. Shape answers what an operation means; strides and layout influence how the operation is carried out. The tensor sections distinguish a high-level tensor expression from the bytes moved to perform it. That distinction is the bridge from a correct model to a fast one.

The lecture uses FLOPs to count arithmetic and arithmetic intensity to compare arithmetic with memory traffic. For an operation, arithmetic intensity is the number of arithmetic operations performed per byte transferred. A large matrix multiplication can reuse values enough times to be compute-bound, while an elementwise activation often moves data with comparatively little arithmetic and becomes bandwidth-bound. The roofline model gives a disciplined way to reason about that difference: achievable performance is constrained by both peak arithmetic throughput and memory bandwidth.

Autodiff changes the cost model of training. The forward pass produces activations, the backward pass computes gradients through the computation graph, and the optimizer transforms gradients into parameter updates. The total training-memory footprint therefore includes parameters, gradients, optimizer state, and saved activations; it is not captured by parameter count alone. The lecture's deep-network and optimizer examples place these pieces in one visible loop.

Gradient accumulation makes several small microbatches contribute to one update. It can let a run emulate a larger effective batch when the whole batch will not fit in memory, though it changes the timing of optimizer steps and can interact with randomness and normalization. Activation checkpointing trades additional forward computation for reduced activation storage by saving selected states and recomputing the others during backward.

**Checks to keep:**

- Write down tensor shapes at every boundary, especially after rearranges and reductions.
- Account for forward, backward, and optimizer-state memory before calling a model “too large.”
- Benchmark after warm-up and synchronize accelerator work before measuring elapsed time.
- Compare a memory-saving configuration against a baseline for both tokens per second and final model quality.

## 03 — Transformer architecture and stability

Source: [lecture_03.pdf](../lecture_03.pdf)

A decoder-only Transformer alternates two ways of mixing information. Attention lets each position retrieve information from earlier positions. The feed-forward block transforms each position independently after that retrieval. Residual paths preserve a route for information and gradients across many layers, while normalization keeps the scale of the signals manageable.

For one attention head, hidden states are projected to queries, keys, and values. A query at position `i` scores every permitted key `j`; the causal mask removes future positions; softmax turns the remaining scores into weights; and the weighted sum of values becomes the position's retrieved information. Scaling the dot product by the square root of the key width keeps scores from becoming needlessly extreme as the width changes. The important picture is not the formula alone: attention is a content-addressed read from the prefix, performed independently at every position and head.

The lecture compares design choices that often travel together in modern models: LayerNorm or RMSNorm, ordinary or gated MLPs, serial or parallel placement of attention and MLP sublayers, and positional schemes such as RoPE. These are not decorative variants. They alter optimization behavior, representation capacity, and the shape of the implementation. Rotary position embeddings, for example, encode relative position through rotations applied to query and key components rather than by simply adding a position vector to the token representation.

Model dimensions are coupled decisions. Increasing width, depth, head count, vocabulary size, or context length changes parameter count, activation memory, attention cost, and the numerical scale of intermediate values. Grouped-query attention (GQA) shares key/value heads among several query heads. It changes the attention representation modestly in exchange for a smaller key-value cache at inference time, a trade-off developed further in Lecture 10.

**Checks to keep:**

- State whether a change affects training compute, inference cache size, activation memory, or all three.
- Verify causal masking with a tiny sequence where a future token would make a leak obvious.
- Treat stable loss as an observation to diagnose: inspect learning rate, initialization, normalization, precision, and data before assuming architecture alone is responsible.
- Keep a small reference implementation whose attention outputs can be compared to optimized kernels.
