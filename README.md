# ReasonForge

Can a 1.5B model get better at math on free Kaggle GPUs? This repo compares three ways of fine-tuning Qwen2.5-1.5B-Instruct on GSM8K: supervised fine-tuning on DeepSeek-R1 traces, GRPO (reinforcement learning with a correctness reward), and DPO on the model's own answers. Everything runs on a single Tesla T4.

## Results

GSM8K test (1,319 problems) and MATH-500, greedy decoding, 4,096-token limit. "Boxed" means the prompt asks for the answer in `\boxed{}`; "plain" is the bare question.

| Model | GSM8K boxed | GSM8K plain | MATH-500 boxed | MATH-500 plain | Median answer |
|---|---|---|---|---|---|
| Base Qwen2.5-1.5B-Instruct | 75.7 | 71.8 | 56.4 | 53.0 | 289 tokens |
| SFT on R1 traces | 61.3 | 56.8 | 31.6 | 32.4 | 963 tokens, 25% cut off |
| GRPO from base | **76.9** | **77.2** | **58.0** | **56.0** | 223 tokens |
| DPO from base | 76.4 | 74.6 | 54.4 | 54.4 | 295 tokens |

Paired tests against base (McNemar):

- SFT is worse everywhere (p < 0.0001).
- GRPO is better on GSM8K plain (+5.4, p < 0.0001). Its other gains are within noise.
- DPO is better on GSM8K plain (+2.8, p = 0.005), flat elsewhere. GRPO beats DPO on GSM8K plain (p = 0.03).

Evaluating the same adapter twice gives numbers 0.2 to 0.4 points apart, so differences smaller than that mean nothing.

What this says:

- SFT on 10K R1 traces made the model worse. It copied R1's habit of re-checking itself but could not resolve the doubt, so a quarter of its GSM8K answers and half of its MATH-500 answers loop until the token limit.
- GRPO never hurts. Its main effect is robustness: base drops 4 points when the prompt does not ask for a boxed answer, GRPO does not. Answers also get 23% shorter.
- DPO on stored right/wrong pairs matched GRPO on the dev set (88.6 vs 88.8) but not on the test set. Learning from fresh samples generalised better than learning from stored ones.
- The base model already solves 96% of GSM8K training problems in at least one of 8 samples. What is left to gain is mostly consistency, and that is where both GRPO and DPO got their improvement.

## How it was done

Model: Qwen2.5-1.5B-Instruct. Compute: one Kaggle T4 (16 GB), sessions capped at 12 hours.

1. **SFT** (`03-sft-training`): QLoRA, rank 32, 10K verified-correct OpenR1-Math-220k traces up to 4,096 tokens, one epoch, 8.4 hours. Two bugs found afterwards: the end-of-turn token was used as the pad token and so masked out of the loss, and the loss covered the prompt as well as the answer.
2. **Evaluation** (`05-eval-vllm`): vLLM in fp16, full test sets in one session. The first harness (HF generate, 512 tokens, 4-bit, 200 problems) understated base by 17 points and made SFT look like a tie. Answer extraction takes the last `\boxed{}`, then `math-verify` for symbolic equality.
3. **Sampling** (`07-sample-base`): base sampled 8 times per GSM8K training problem. Greedy 85.1%, majority of 8 89.1%, at least one of 8 correct 96.2%. The pass rates drive GRPO's prompt selection; the samples are DPO's training pairs.
4. **GRPO** (`06-grpo-base`): Unsloth + TRL, LoRA rank 32 on fp16, 8 samples per prompt, 600 steps, 6 hours. Reward is 1 if the boxed answer matches, else 0. Prompts with mixed pass rates get full weight, the rest a quarter. Dev accuracy 85.4 to 88.8.
5. **DPO** (`08-dpo-base`): one correct and one wrong sample per problem, 4,930 pairs, one epoch, beta 0.1, 2.4 hours plus 1.4 hours of reference log-prob precomputation. Dev accuracy 85.4 to 88.6.

A 500-problem dev split of GSM8K train is used for every checkpoint decision; the test sets are only reported, never selected on. Dev scores run about 10 points above test because the model has seen the training problems, which is why dev gains shrink on test.

The scoring code is in `analysis/`. `rf_extract.py` is the one answer checker shared by the GRPO reward, the Kaggle evaluation and the local scorer. `score.py` gives accuracy, confidence intervals, paired tests, a failure breakdown and MATH-500 results by subject and level. `notebooks/build_notebooks.py` generates notebooks 05 to 08 from the files in `analysis/`.

## Reproduce

Run on Kaggle with a T4 and internet on, as a committed version so the outputs are kept.

1. `07-sample-base.ipynb`, about 4 hours. Produces `samples/gsm8k_train_k8.pkl`.
2. `06-grpo-base.ipynb` with the samples file as input. `MODE = "smoke"` to check the setup, then `"full"`, about 7.5 hours.
3. `08-dpo-base.ipynb` with the samples file as input. Same modes, about 3 hours for `"full"`.
4. `05-eval-vllm.ipynb` with the adapters and any earlier results as inputs. Earlier results are reused, new adapters are evaluated.
5. Locally: `python analysis/score.py analysis/results/*.pkl`. Needs Python 3.12 and `math-verify`.

## Status

Done: SFT, evaluation harness, base sampling, GRPO, DPO.
Possible next steps: SFT redo with the training bugs fixed, distillation to 0.5B.

## License

MIT
