# Project Journal

## Day 1 — [today's date]

**Goal:** Validate Kaggle environment, load Qwen-2.5-1.5B in 4-bit, smoke-test inference.

**What worked:**
- Qwen-2.5-1.5B loaded at 0.71 GB VRAM in 4-bit NF4 with double quantization
- Inference produced correct answer (28 apples) with step-by-step reasoning
- Stack validated end-to-end: PyTorch + transformers + bitsandbytes + CUDA 12.8

**What broke and how I fixed it:**
1. **protobuf v7 conflict** — `transformers` was lazily importing TensorFlow, which 
   requires protobuf 5.x, but pip had upgraded to v7. Diagnosed via import traceback. 
   Fix: pinned `protobuf>=5.29,<6.0`, set `TRANSFORMERS_NO_TF=1` and `USE_TF=0` 
   environment variables to prevent TF imports.
2. **bitsandbytes CUDA mismatch** — bitsandbytes 0.44.1 had no binary for Kaggle's 
   CUDA 12.8, and depended on Triton 2.x's removed `triton.ops` module. Fix: upgraded 
   to bitsandbytes 0.49.2 which ships CUDA 12.8 binaries and works with Triton 3.x.

**Observations:**
- Qwen produces reasoning in plain numbered steps with `**bold**` answers — NOT in 
  `<think>` tags or `\boxed{}` format. This means OpenR1-Math-220k will teach the 
  model a new format on top of teaching reasoning. Design decision: keep `<think>` 
  tags (cleaner for GRPO reward).
- 4-bit NF4 + double quantization gives ~770 MB raw model size, matching theoretical 
  expectations (1.54B params × 0.5 bytes - dequant savings).

**Open questions:**
- Why does QLoRA's double quantization save bits — what exactly gets quantized twice?
- What's the difference between fp16 compute dtype and the 4-bit storage dtype during forward pass?
- For GRPO's KL penalty β=0.04 — what's the math behind this specific value?

## Day 2 — [today's date]

**Goal:** Prepare OpenR1-Math-220k for SFT training.

**Key dataset findings:**
- Dataset has 2 R1-generated reasoning traces per problem; majority pass 
  `correctness_math_verify`
- Reasoning traces are LONG — median 4843 tokens, p90 11.7K, max 19K
- All traces use `<think>...</think>` then final `\boxed{answer}` structure 
  (DeepSeek-R1 distillation format)
- `messages` field already chat-formatted, applies cleanly to Qwen tokenizer

**Engineering decisions:**
- **Quality filter:** kept examples where `correctness_math_verify` was True, 
  fell back to `correctness_llama` when math verify was unavailable
- **Generation expansion:** treated each verified-correct trace as a separate 
  training example — expanded from ~94K filtered problems to 124K training rows
- **MAX_TOKENS = 4096:** deliberate tradeoff. p50 of traces is 4843 tokens, 
  so cap retains 41.6% of expanded examples (~52K). Going to 8192 would 
  retain ~85% but make a single epoch unfit in a 9hr Kaggle session. 
  4096 keeps 1 epoch at ~5-6 hours while preserving 51.8K high-quality examples 
  — far more than the 20K we plan to train on.
- **Single-process tokenization:** dropped `num_proc=4` after OOM. Kaggle's 
  ~13GB system RAM can't hold 4 worker copies of long-trace tokenization. 
  Single-process is ~5x slower but stable.

**Final dataset:** 19,600 train / 400 val, saved to `/kaggle/working/`

**Open questions:**
- How will SFT loss curve look? Expect rapid initial drop (format learning), 
  then slower decline (reasoning learning).
- Will the model overfit on 19.6K examples for 1 epoch?
- What's the right learning rate? 2e-4 is QLoRA default; may need adjustment 
  for long-context training.