# ReasonForge

Can a 1.5B model get better at math on free Kaggle GPUs? This repo tries three ways to improve Qwen2.5-1.5B-Instruct on GSM8K and MATH-500: supervised fine-tuning on DeepSeek-R1 traces, GRPO (reinforcement learning with a correctness reward), and next, DPO on the model's own answers. Everything runs on a single Tesla T4.

## Results

GSM8K test (1,319 problems) and MATH-500, greedy decoding, 4,096-token limit. "Boxed" means the prompt asks for the answer in `\boxed{}`; "plain" is the bare question.

| Model | GSM8K boxed | GSM8K plain | MATH-500 boxed | MATH-500 plain | Median answer length |
|---|---|---|---|---|---|
| Base Qwen2.5-1.5B-Instruct | 75.7 | 71.8 | 56.4 | 53.0 | 289 tokens |
| SFT on R1 traces (QLoRA) | 61.3 | 56.8 | 31.6 | 32.4 | 963 tokens, 25% cut off |
| GRPO from base | **76.9** | **77.2** | **58.0** | **56.0** | 223 tokens |

Paired significance against base (McNemar): SFT is worse on every setting (p < 0.0001). GRPO is better on GSM8K plain (+5.4, p < 0.0001); the other GRPO gains (+1.2, +1.6, +3.0) are within noise at these sample sizes. Running the same GRPO adapter twice gives numbers 0.2 to 0.4 points apart, so treat anything smaller than that as noise.

What the numbers say:

- SFT on 10K R1-style traces made the model worse. It copied R1's habit of re-checking itself but could not resolve the doubt, so 25% of GSM8K answers and 52% of MATH-500 answers loop until the token limit. 92 to 98% of those cut-off answers are repetition loops. Even where it finished, it scored below base.
- GRPO from base never hurts and makes the model robust to the prompt: without the boxed instruction, base drops 4 points, GRPO does not. Answers also get 23% shorter.
- The base model already solves 96% of GSM8K training problems in at least one of 8 samples, so most of the remaining gap is consistency rather than missing skill. GRPO's gains on a held-out dev set came from problems the base solved only some of the time.

## How it was done

Model: Qwen2.5-1.5B-Instruct. Compute: one Kaggle T4 (16 GB), sessions capped at 12 hours.

1. **SFT** (`notebooks/03-sft-training.ipynb`): QLoRA, rank 32 on all projection layers, 10K verified-correct OpenR1-Math-220k traces up to 4,096 tokens, one epoch, 8.4 hours. Two mistakes found afterwards: the end-of-turn token was set as the pad token and so masked out of the loss, and the loss covered the prompt as well as the answer.
2. **Evaluation** (`notebooks/05-eval-vllm.ipynb`): vLLM in fp16, batched, full test sets in one session. The first evaluation harness used HF generate one problem at a time with a 512-token limit, 4-bit weights and a hidden repetition penalty; it understated base by 17 points and made SFT look like a tie. Answer extraction takes the last `\boxed{}` with balanced braces, then `math-verify` for symbolic equality.
3. **Sampling** (`notebooks/07-sample-base.ipynb`): base sampled 8 times at temperature 1.0 on every GSM8K training problem. Greedy 85.1%, majority of 8 89.1%, at least one of 8 correct 96.2%. The per-problem pass rate feeds GRPO's prompt selection, and the samples are the data for DPO.
4. **GRPO** (`notebooks/06-grpo-base.ipynb`): Unsloth + TRL, LoRA rank 32, fp16 base, 8 samples per prompt, 4 prompts per step, 600 steps, learning rate 1e-5, no KL term, DAPO loss. Reward is 1 if the last boxed answer matches, else 0. Prompts with mixed pass rates get full weight, all-right or all-wrong ones a quarter. A 500-problem dev split of GSM8K train was held out for checkpoint selection; dev accuracy rose from 85.4 to 88.8 and plateaued around step 350. Test numbers come from the same harness as base and SFT.

The scoring code lives in `analysis/`. `rf_extract.py` is the one answer checker used by the GRPO reward, the Kaggle evaluation and the local scorer, so they cannot disagree. `score.py` gives accuracy, bootstrap intervals, paired tests, a failure breakdown and MATH-500 results by subject and level. `notebooks/build_notebooks.py` generates notebooks 05 to 07 from those files.

## Reproduce

Run on Kaggle with a T4 and internet on, as a committed version so the outputs are kept.

1. `07-sample-base.ipynb`: about 4 hours. Produces `samples/gsm8k_train_k8.pkl`.
2. `06-grpo-base.ipynb` with the samples file as input. `MODE = "smoke"` (15 min) to check the setup, then `"full"` (about 7.5 hours). Produces the adapter and the dev curve.
3. `05-eval-vllm.ipynb` with the adapter and any earlier results as inputs. Earlier results are reused, new adapters are evaluated. Produces `results/*.pkl`.
4. Locally: `python analysis/score.py analysis/results/*.pkl`. Needs Python 3.12 and `math-verify`.

## Status

Done: SFT, evaluation harness, base sampling, GRPO from base.
Next: DPO on the sampled right/wrong pairs (off-policy comparison to GRPO), SFT redo with the training bugs fixed, then distillation to 0.5B.

## License

MIT
