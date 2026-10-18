# CS336 course notes

These notes are a working guide to the Spring 2026 materials on this repository's `main` branch. They explain how the lectures fit together, state the engineering questions that each one answers, and point back to the lecture source for the details and examples.

The lecture files remain authoritative. In particular, these notes do not try to merge the Spring 2026 material with the separate Spring 2025 course branch: lecture order, examples, and recommendations can change between editions.

## Read the course as one system

A language model is not only a Transformer function. It is a chain of representations, optimization procedures, systems choices, data decisions, and measurements. A change in any part of the chain can alter the result that appears to be a property of “the model.”

| Lectures | Question to carry forward | Notes |
| --- | --- | --- |
| 01–03 | What function is trained, and how is text represented? | [Foundations](01-foundations.md) |
| 04–08 | How can that function fit and run efficiently across accelerators? | [Systems and scaling](02-systems-and-scaling.md) |
| 09–11 | How should a training run be sized, tuned, and interpreted? | [Systems and scaling](02-systems-and-scaling.md) |
| 12–16 | What behavior is being measured and optimized after pre-training? | [Evaluation and post-training](03-evaluation-and-post-training.md) |
| 17 | How does the text model become a model of images and text together? | [Multimodal models](04-multimodal.md) |

## Lecture map

| Lecture | Main concern | Primary source |
| --- | --- | --- |
| 01 | Course map and tokenization | [lecture_01.py](../lecture_01.py) |
| 02 | Tensors, autodiff, and the training loop | [lecture_02.py](../lecture_02.py) |
| 03 | Transformer architecture and stability | [lecture_03.pdf](../lecture_03.pdf) |
| 04 | Attention alternatives and mixture-of-experts | [lecture_04.pdf](../lecture_04.pdf) |
| 05 | GPU systems and memory traffic | [lecture_05.pdf](../lecture_05.pdf) |
| 06 | Kernel programming and performance measurement | [lecture_06.py](../lecture_06.py) |
| 07 | Collective communication and parallelism | [lecture_07.py](../lecture_07.py) |
| 08 | Training parallelism strategies | [lecture_08.pdf](../lecture_08.pdf) |
| 09 | Scaling laws | [lecture_09.pdf](../lecture_09.pdf) |
| 10 | Inference systems | [lecture_10.py](../lecture_10.py) |
| 11 | Training and scaling recipes | [lecture_11.pdf](../lecture_11.pdf) |
| 12 | Evaluation | [lecture_12.py](../lecture_12.py) |
| 13 | Data sources | [lecture_13.py](../lecture_13.py) |
| 14 | Data processing and mixing | [lecture_14.py](../lecture_14.py) |
| 15 | Instruction tuning and preference optimization | [lecture_15.pdf](../lecture_15.pdf) |
| 16 | Reinforcement learning from verifiable rewards | [lecture_16.pdf](../lecture_16.pdf) |
| 17 | Multimodal language models | [lecture_17.py](../lecture_17.py) |

## A useful discipline for every lecture

When evaluating an idea in this course, keep four questions separate:

1. **What changes?** Name the representation, computation, data distribution, or policy that is different.
2. **Why should it help?** State the proposed mechanism: less memory traffic, a better statistical signal, more useful capacity, lower latency, and so on.
3. **What does it cost?** Include compute, memory, communication, implementation complexity, data provenance, and evaluator labor.
4. **What measurement could disprove the claim?** Decide the task, metric, baseline, and held-out data before treating an improvement as real.

That discipline prevents several common confusions: a lower training loss can be mistaken for better product behavior; a faster kernel can be mistaken for a faster training run; and a benchmark gain can be mistaken for a general capability gain.

## Notation used in the notes

`L` is sequence length, `d` is a hidden width, and `B` is batch size when those names are useful. A next-token model reads a token prefix and produces a score for every possible next token. Applying softmax turns those scores into a probability distribution; training then increases the probability placed on the next token observed in the data. The notation matters only because it lets us say exactly which quantity grows with sequence length, which data are cached, and which tensors must be communicated.

Each thematic note closes with checks that are meant to be performed against a real run or implementation, rather than memorized as slogans.
