# Multimodal models: connecting images to language

Source: [lecture_17.py](../lecture_17.py)

An image-language model has to solve a representation problem before it can answer a visual question. A vision encoder turns an image into a sequence or grid of feature vectors. A connector projects, resamples, or otherwise adapts those vectors to the language model's hidden space. The language model then treats the visual representation as part of the context from which it predicts text or other tokens.

## The common pattern

CLIP and SigLIP illustrate contrastive ways to align images and text. Rather than generating a caption token by token, the system learns representations that make matched image-text pairs compatible and mismatched pairs less compatible. The loss construction changes the learning signal: the notes in the lecture distinguish the normalized contrastive formulation used by CLIP from the pairwise sigmoid formulation used by SigLIP.

LLaVA-style models combine a pretrained vision encoder with a projection into the language model and train the resulting system on image-text instruction data. Later systems change the visual token budget, image resolution handling, connector, data recipe, or the degree to which the architecture treats different modalities jointly. The lecture traces these choices through LLaVA-OneVision, Qwen-VL, Qwen2-VL, Qwen3-VL, and Chameleon.

The important engineering question is where the modalities meet. Injecting visual features as a prefix gives the text model visual context but preserves a distinct image encoder. A unified token approach can use a shared generative interface for image and text, but it changes the vocabulary, training data, compute distribution, and failure surface. “Multimodal” is therefore a family of designs, not one architecture.

## What to measure

Visual benchmarks should test the intended use: document understanding, charts, OCR, grounded question answering, spatial reasoning, multi-image comparison, or image generation. A score can be inflated by textual clues in a dataset, contaminated examples, a narrow answer format, or a task that does not require the image. Include counterfactual tests in which the image is removed, replaced, cropped, or contradicted by the prompt; they make it possible to see whether the answer actually depends on visual evidence.

Track the image encoder, connector, language model, image preprocessing, token budget, training data mixture, and evaluation protocol as separate versions. A change in image resolution or tiling can change both accuracy and latency even when the language model weights are untouched.

## Multimodal checklist

- Specify the visual representation: patch or image features, number of visual tokens, and how they enter the language model.
- Test whether answers change appropriately when the image content changes.
- Measure image preprocessing, prefill cost, KV-cache effect, and memory alongside answer quality.
- Keep source and rights information for image-text data as carefully as for text-only data.
- Report which tasks were evaluated and which visual capabilities were not tested.
