# LLM Domain Adaptation — Full Fine-Tuning vs LoRA

Adapting a compact causal language model (**DistilGPT2**) to the formal register of
government policy reports, and comparing **full fine-tuning** against a
**parameter-efficient LoRA adapter** on cost, parameter count, and output quality.

---

## Overview

Out of the box, a general-purpose language model is "surprised" by the dense,
procedural language of formal policy documents. This project measures that gap and
then closes it two different ways:

1. **Full fine-tuning** — update all ~82M weights of DistilGPT2.
2. **LoRA (Low-Rank Adaptation)** — freeze the base model and train a small set of
   low-rank adapter weights (~147K parameters) instead.

The model is fine-tuned on one policy document (the UN Declaration on the Rights of
Indigenous Peoples) and evaluated on **both** that document *and* a second, unseen
related report on First Nations economic self-determination — so the results capture
both in-domain adaptation and transfer to nearby text.

## Key result

> The LoRA adapter recovered most of the domain adaptation while training just
> **0.18% of the model's parameters** — a textbook illustration of the
> cost-vs-quality trade-off in PEFT.

Full fine-tuning produced the largest perplexity reductions and the smoothest training
curve, but at the cost of updating every weight. LoRA captured most of the stylistic
lift for a tiny fraction of the trainable parameters, with bumpier convergence and
smaller gains on the more distant (economic) text.

## Results

Perplexity — lower is better. Fine-tuning was performed on the UNDRIP corpus; the
*Economic* column measures transfer to an unseen related document.

| Model               | Trainable params | UNDRIP PPL | Economic PPL |
| ------------------- | ---------------- | ---------- | ------------ |
| Baseline DistilGPT2 | —                | 33.03      | 45.13        |
| Full fine-tune      | 82M (100%)       | 30.97      | 43.92        |
| LoRA adapter        | 147K (0.18%)     | 32.21      | 44.64        |

Qualitatively, on a shared prompt: the base model tended to ramble and invent details;
the fully fine-tuned model produced the most authentic policy-style continuations; and
the LoRA model was concise and formal but captured less fine-grained detail.

## Method

1. **Extract & chunk** — convert the policy PDFs to text and split into overlapping
   512-token windows (256-token stride).
2. **Baseline** — compute DistilGPT2's perplexity on both corpora as an out-of-the-box
   benchmark.
3. **Full fine-tune** — train all parameters on the UNDRIP chunks (3 epochs), then
   re-measure perplexity on both corpora.
4. **LoRA** — freeze the base model, inject low-rank adapters into the attention
   projection, train only those, then re-measure.
5. **Compare** — perplexity, trainable-parameter counts, training time, and generated
   text quality across all three.

## LoRA configuration

| Hyperparameter   | Value          |
| ---------------- | -------------- |
| Rank (`r`)       | 8              |
| `lora_alpha`     | 32             |
| Target modules   | `c_attn`       |
| `lora_dropout`   | 0.05           |
| Bias             | `none`         |
| Task type        | `CAUSAL_LM`    |

## Scope & honest notes

This began as a postgraduate coursework experiment and is **intentionally small-scale**:
the corpus is ~26 training chunks and training runs for only 12 optimization steps. The
goal is a clean, controlled comparison of the two adaptation methods — not a
large-benchmark result. The perplexity deltas are correspondingly modest (1–2 points);
a larger corpus and longer training would be expected to widen the gap between full
fine-tuning and LoRA. The numbers above are reported as-run, with no rounding in the
model's favour.

## Tech stack

- **Python**
- **PyTorch**
- **Hugging Face Transformers** — model, tokenizer, `Trainer`
- **PEFT** — LoRA adapters
- **Datasets** — data handling
- **PyPDF2** — PDF text extraction
- **Matplotlib** — perplexity / loss visualisation

## Repository structure

```
.
├── notebook.ipynb        # End-to-end experiment (baseline → full FT → LoRA → comparison)
├── data/                 # Source policy PDFs (not committed — see note below)
├── requirements.txt
└── README.md
```

> Note: the source PDFs are public policy documents. Add them under `data/` (or update
> the paths in the notebook) before running.

## Getting started

```bash
# 1. Install dependencies
pip install torch transformers peft datasets PyPDF2 matplotlib

# 2. Place the source PDFs under data/ and update the paths in the notebook if needed

# 3. Run the notebook top to bottom
jupyter notebook notebook.ipynb
```

A CUDA-capable GPU is recommended but not required — the model and dataset are small
enough to run on CPU.

## References

- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
- Houlsby et al. (2019). *Parameter-Efficient Transfer Learning for NLP.* arXiv:1902.00751
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Hugging Face — [Transformers](https://huggingface.co/docs/transformers/) ·
  [PEFT](https://github.com/huggingface/peft) ·
  [distilgpt2](https://huggingface.co/distilgpt2)
