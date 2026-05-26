# ReasonForge

A small reasoning language model trained via SFT → GRPO (RLVR) → distillation, 
running entirely on free Kaggle GPUs.

## Goals

- Take Qwen-2.5-1.5B base model and improve math/code reasoning capability
- Apply the full LLM lifecycle: SFT with QLoRA → GRPO with verifiable rewards 
  → distillation to 0.5B → GGUF quantization → vLLM serving
- Document every design decision (rank, target modules, reward shaping, etc.) 
  with reasoning grounded in the literature

## Stack

- **Base model:** Qwen-2.5-1.5B-Instruct
- **Training:** QLoRA (4-bit NF4) via HuggingFace TRL
- **RL:** GRPO with verifiable rewards on GSM8K
- **Serving:** vLLM + llama.cpp (GGUF)
- **Compute:** Kaggle free T4 GPU (16GB VRAM)
- **Tracking:** Weights & Biases

## Progress

- [x] Day 1: Environment setup, model loading, inference smoke test
- [ ] Day 2: Dataset preparation (OpenR1-Math-220k)
- [ ] Day 3-4: SFT training
- [ ] Day 5-7: Evaluation harness (GSM8K, MATH-500, MBPP, MMLU)
- [ ] Week 2: GRPO with verifiable rewards
- [ ] Week 3: Distillation + quantization + serving benchmarks
- [ ] Week 4: HuggingFace demo + technical writeup

## Results

(Updated as project progresses)

| Stage | GSM8K | MATH-500 | Notes |
|-------|-------|----------|-------|
| Base Qwen-2.5-1.5B | TBD | TBD | Zero-shot baseline |
| After SFT (QLoRA) | TBD | TBD | OpenR1-Math-220k subset |
| After GRPO | TBD | TBD | RLVR on GSM8K train |

## License

MIT