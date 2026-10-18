# Evaluation and post-training: deciding what behavior is worth optimizing

Pre-training gives a model a broad next-token objective. It does not, by itself, settle how a model should answer users, call tools, follow policies, solve a coding task, or behave under distribution shift. These lectures make the target behavior and the evidence for it explicit.

## 12 — Evaluation

Source: [lecture_12.py](../lecture_12.py)

An evaluation begins with a decision. “Is this model good?” is too broad until it is tied to a user, task, failure cost, and comparison. Perplexity measures how much probability a model assigns to held-out next tokens; it is a useful pre-training signal, but it is not a general score for helpfulness, tool use, mathematical reasoning, or safety.

The lecture surveys exam-style, chat, agentic, pure-reasoning, and safety benchmarks. Their diversity is a reminder that benchmark format shapes what is observed. A multiple-choice result may reflect answer-order or prompt choices. An agentic result also depends on the tool environment, stopping rules, and verifier. A safety result depends on the taxonomy of failures and the annotators' judgment. Realistic evaluation seeks a task distribution that resembles use; valid evaluation measures the claimed property rather than a shortcut.

Benchmark contamination, prompt sensitivity, evaluator bias, and narrow task coverage are not reasons to abandon evaluation. They are reasons to state what an evaluation can establish and to supplement it with targeted tests, human review, and operational monitoring.

## 13 — Data sources

Source: [lecture_13.py](../lecture_13.py)

Training corpora are assembled from sources with very different structure, quality, permissions, and social context. The lecture examines Common Crawl, Wikipedia, GitHub, arXiv, books, web-text collections, code collections, and large open datasets such as the Pile, C4, Dolma, DCLM, Nemotron-CC, the Stack, and the Common Pile. A source name is not enough to describe a dataset: the collection date, extraction rule, filtering, language coverage, licensing posture, and deduplication policy all matter.

Copyright and provenance belong in the engineering record. They affect whether material can be used, redistributed, audited, removed, or described honestly. They also affect representation: a corpus that is convenient to collect can systematically underrepresent languages, genres, or communities. The historical survey in the lecture makes clear that data choices have changed alongside model scale and access constraints.

For a real corpus, maintain source-level metadata and a clear path from retained document back to its origin and transformations. This is necessary for debugging model behavior as well as for governance.

## 14 — Data processing and mixing

Source: [lecture_14.py](../lecture_14.py)

Raw data becomes model input through transformations, filters, deduplication, and mixture selection. A transformation might normalize markup or remove boilerplate. A filter might reject low-quality pages, unwanted language, personally identifying content, or a particular document form. Every such operation creates false positives and false negatives, so it should be evaluated on samples and tracked in statistics rather than described as a magical cleaning step.

Deduplication makes the problem more concrete. Exact hashing catches identical records. Near-duplicate methods such as MinHash and locality-sensitive hashing compare approximations to token-set similarity so that repeated or lightly modified documents can be found at scale. These methods have thresholds, shingling choices, collision behavior, and computational costs. They change both training efficiency and the risk that evaluation material has close relatives in training data.

Mixture weights decide how often each source appears during training. They are an objective decision, not just an ingestion detail: raising the share of code, mathematics, conversational text, or a particular language changes what the model is optimized to predict. Post-training data needs a separately stated purpose and quality bar because it can overwrite, narrow, or reshape the general behavior learned during pre-training.

## 15 — Instruction tuning and preference optimization

Source: [lecture_15.pdf](../lecture_15.pdf)

Supervised fine-tuning (SFT) teaches a model to imitate selected responses for prompts. Preference-based methods use comparisons, ratings, or a learned reward signal to prefer one response over another. Both procedures replace the broad pre-training distribution with a much narrower and more consequential training signal. The source of prompts, responses, and judgments therefore determines the behavior that is reinforced.

The lecture discusses RLHF and direct preference optimization (DPO), along with practical concerns about response length, evaluator interfaces, and annotator labor. A preference label is evidence about a task under a rubric; it is not a universal statement of truth. If a reward process favors length, fluency, refusal, or a recognizable template, optimization can amplify that feature even when it is not the intended outcome. The resulting overoptimization or mode collapse is a model-behavior problem, not merely a metric issue.

DPO is useful to understand as a direct way to increase the relative likelihood of preferred responses while remaining close to a reference model. Its objective still inherits the limitations of the preference data and the comparison process. Examine examples, disagreement, and behavior under distribution shift alongside aggregate win rates.

## 16 — Reinforcement learning from verifiable rewards

Source: [lecture_16.pdf](../lecture_16.pdf)

Reinforcement learning from verifiable rewards (RLVR) uses tasks where an outcome can be checked by a program, a test suite, a mathematical verifier, or another explicit procedure. That creates a sharper signal than an open-ended style preference for some domains, especially programming and formal or numerical tasks. It does not remove the problem of task selection: a verifier checks the property it implements, and a policy can still find shortcuts in an incomplete environment.

The lecture discusses policy gradients, group-relative policy optimization (GRPO), advantage estimation, length normalization, reasoning bootstrapping, and data curation. In GRPO-style reasoning, several completions for a prompt can be compared within a group; reward relative to that group gives a learning signal. The detail matters because group composition, reward scale, output length, and filtering all alter what the model is encouraged to produce.

Reasoning improvements can coexist with regressions in general conversation, factuality, style, or other domains. Track the complete evaluation profile and preserve a capable reference policy. Strong verification is valuable, but it is neither a substitute for clear task design nor proof of broad intelligence.

## Evaluation and post-training checklist

- Start from the decision and the failure modes, then choose metrics and tasks.
- Version prompts, tools, model settings, scorers, and datasets with the result.
- Inspect representative failures and disagreements before treating an aggregate score as a product decision.
- Preserve data lineage through filtering, deduplication, and mixing; report removal rates and known blind spots.
- Treat human judgments and automated verifiers as measurement systems with scope, error, and incentives.
- Compare post-trained models to the pre-trained or SFT reference across capabilities that may be unintentionally changed.
