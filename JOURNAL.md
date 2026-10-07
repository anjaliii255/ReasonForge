# Project Journal

## 26 May 2026: smoke test

Goal: load Qwen2.5-1.5B-Instruct in 4-bit on a Kaggle T4 and run inference.

- Loaded at 0.71 GB with NF4 and double quantization. A simple word problem came back correct with numbered steps.
- Two install problems. transformers pulled in TensorFlow, which needed an older protobuf; fixed by pinning protobuf below 6 and setting `TRANSFORMERS_NO_TF=1`. bitsandbytes 0.44 had no CUDA 12.8 build; 0.49 did.
- Qwen writes plain numbered steps with bold answers, not `<think>` tags or `\boxed{}`. The SFT data uses both, so SFT will teach a new format on top of reasoning.

## 26 May 2026: data prep

Goal: prepare OpenR1-Math-220k for SFT.

- 93,733 problems, two R1 traces each, most verified correct. Kept every verified-correct trace as its own example: 124,445 rows.
- Traces are long. Median 4,843 tokens, p90 11.7K. Capping at 4,096 keeps 41.6% (51,802 rows); a single epoch over more than that would not fit a Kaggle session.
- Multi-process tokenization ran out of RAM on 13 GB; single process was slow but fine.
- Final: 19,600 train, 400 validation.

## 28 May 2026: SFT

- QLoRA rank 32 on all seven projection layers, 36.9M trainable parameters. 10K examples, one epoch, 312 steps, batch 1 with 32 accumulation, learning rate 2e-4, 8.4 hours. Final loss 0.558.
- Spot checks showed the model now writes `<think>` blocks and boxes its answer.

## 29 May 2026: first evaluation

- 200 GSM8K problems with HF generate, one at a time. Base 48%, SFT 56%. Looked like an 8-point gain.
- Re-scoring the same outputs with a better extractor (last box instead of first, tolerate `$` and units) gave base 58.5%, SFT 59.0%. The gain was a scoring artifact. SFT also hit the 1,500-token limit on 18% of problems.

## 30 Sep 2026: new evaluation harness

- The old harness could not evaluate the full test set; 200 SFT problems took 8 hours. Rebuilt it on vLLM with Unsloth's T4 package pins (vLLM 0.11.2, transformers 4.56.2, TRL 0.22.2). Full GSM8K plus MATH-500 for two models now takes about 8 hours total, most of it SFT's long outputs.
- vLLM crashed on the T4 because FlashInfer tries to compile kernels and cannot find `libcuda.so` on Kaggle. Forcing the Triton attention backend fixed it.
- The answer checker now lives in one file used everywhere. Fixed a bug where any two `\text{...}` answers matched each other.
- Learned the hard way that a draft session wipes its files when it times out. Long runs go through Save & Run All.

## 1 Oct 2026: SFT is worse than base

Full test sets, greedy, 4,096 tokens, boxed prompt. Base 75.7 on GSM8K and 56.4 on MATH-500. SFT 61.3 and 31.6. Both gaps significant at p < 0.0001.

- Base is much stronger than the first harness showed. 4-bit weights, a 512-token limit, a hidden repetition penalty of 1.05 from the generation config and no boxed prompt cost it about 17 points.
- SFT's failures are loops. 25% of GSM8K and 52% of MATH-500 answers run to the limit; 92 to 98% of those repeat the same 8-word phrase five or more times near the end, and almost none close `</think>`. The model writes "Wait, let me approach this differently" and repeats the same steps. Base solves 60% of the GSM8K problems SFT loops on.
- Sampling at temperature 0.6 reduces cut-offs but does not recover accuracy (58.4 and 30.6).
- Where SFT did finish, it was still below base (77.2 vs 80.9 on those GSM8K problems), so the damage is not only the loops.
- Found two bugs in the SFT notebook: `pad_token = eos_token` masks the end-of-turn token out of the loss, and the loss covers the prompt. Neither explains the loops; the data does. A 671B model's self-correction style cannot be copied by a 1.5B model from 10K examples.
- Hand-checked 50 of base's wrong answers: all were real mismatches, none were scorer mistakes.
- Decided on a dev split of 500 GSM8K train problems for all model selection, with the test set used only for reporting.

## 2 Oct 2026: sampling the base model

8 samples at temperature 1.0 for every GSM8K training problem, 1,024-token limit, about 4 hours.

- Greedy 85.1%, mean of samples 71.7%, majority of 8 89.1%, at least one of 8 correct 96.2%. The gap between greedy and pass@8 is 11 points, so there is room for methods that use the model's own samples.
- 70.7% of problems have mixed outcomes, 3.8% are never solved, 25.5% always solved.
- Greedy on train is 85% against 76% on test. The training problems are easier for this model, probably seen in pretraining. Dev numbers are for comparing models, not for reporting.
- Looping is 0.3%, so the base model has no length problem.

## 3 Oct 2026: GRPO from base

- Setup: Unsloth + TRL GRPO, LoRA rank 32, fp16, 8 samples per prompt, 4 prompts per step, 1,024-token limit, learning rate 1e-5, no KL term, DAPO loss, reward 1 for a correct boxed answer and 0 otherwise. Dropped the format reward because base already boxes 99% of answers. Mixed-pass-rate prompts get full weight, the rest a quarter. About 36 seconds per step.
- Smoke test of 5 steps passed. Pilot of 100 steps: dev 85.4 to 87.4, not significant but trending up, with gains on problems base solved 3 to 5 times out of 8.
- Full run of 600 steps, 7.5 hours: dev 85.4 to 88.8 (p = 0.058), best checkpoints 89.2 at steps 350 and 450, plateau after 350. Gains concentrated on problems base solved 1 to 5 times out of 8 (+10 and +12 points); no change on the easiest and hardest buckets. Answers shortened from 274 to 217 tokens. Training reward rose from 0.77 to 0.86. Evaluating the same weights twice differs by 2 problems in 500, which is the noise floor.

## 4 Oct 2026: GRPO on the test sets

Same harness as base and SFT.

| | GSM8K boxed | GSM8K plain | MATH-500 boxed | MATH-500 plain |
|---|---|---|---|---|
| Base | 75.7 | 71.8 | 56.4 | 53.0 |
| GRPO | 76.9 | 77.2 | 58.0 | 56.0 |

- GRPO is ahead everywhere and never worse. The only significant gain is GSM8K plain (+5.4, p < 0.0001). The boxed gains are within noise.
- The dev gain of +3.4 became +1.2 on the boxed test. Dev is drawn from the training distribution, so it overstated the gain. This is what the held-out test set is for.
- The real effect is robustness. Base relies on being told to box the answer; GRPO boxes 88% of the time without the instruction and scores the same either way. Answers are 23% shorter with no loss.
- MATH-500 moved a little from GSM8K-only training. Gains on Intermediate Algebra, Prealgebra and Precalculus, a drop on Geometry; by level, gains on 1 to 3 and 5, a drop on 4. The slices are small, so no conclusions from any single one.
- The same adapter got evaluated twice under two labels by accident. The two runs differ by 0.2 to 0.4 points. Useful as a noise gauge.

## 5 Oct 2026: DPO on the base model's own samples

- Data: for each training problem with both right and wrong samples, one random correct answer as chosen and one random wrong answer as rejected. 4,930 pairs, dev problems excluded.
- Setup: LoRA rank 32 on fp16, beta 0.1, learning rate 5e-6, 16 pairs per step, one epoch. The first attempt ran out of memory because vLLM was loaded alongside the training model; DPO keeps full logits for four sequences per step, which GRPO avoids. Fixed by training without vLLM and loading a fresh engine afterwards for the dev evaluation.
- 309 steps at about 11 seconds each, 2.4 hours, plus 1.4 hours to precompute reference log-probs. Reward margin grew from 0.35 to about 2.2.
- Dev accuracy 85.4 to 88.6 (p = 0.024). On dev, DPO and GRPO are indistinguishable: 414 problems both right, 29 only DPO, 30 only GRPO. Both gain on problems base solved 1 to 5 times out of 8. DPO did not shorten answers; GRPO did.

## 6 Oct 2026: DPO on the test sets

| | GSM8K boxed | GSM8K plain | MATH-500 boxed | MATH-500 plain |
|---|---|---|---|---|
| Base | 75.7 | 71.8 | 56.4 | 53.0 |
| GRPO | 76.9 | 77.2 | 58.0 | 56.0 |
| DPO | 76.4 | 74.6 | 54.4 | 54.4 |

- DPO's only significant gain is GSM8K plain (+2.8, p = 0.005). On MATH-500 it is flat to slightly worse, within noise.
- GRPO beats DPO on GSM8K plain (p = 0.03) and is ahead, though not significantly, everywhere else.
- The same dev gain (+3.2 vs +3.4) turned into different test gains. DPO trained on samples drawn once from the base model; GRPO drew fresh samples from the current model at every step. The fresh samples generalised better.

## 6 Oct 2026: SFT redo on the model's own answers

First run of the fixed SFT code (own pad token, loss on the answer only, fp16 LoRA instead of 4-bit). Data: one random correct sample per training problem, 6,711 examples, one epoch, 420 steps, learning rate 2e-4 cosine, 37 minutes. The mask check confirmed 267 of 370 tokens in the loss and `<|im_end|>` as the last trained token.

- Dev accuracy stayed flat: base 85.4, checkpoints 84.4 to 85.0, every difference p > 0.5. Boxing, answer length and looping did not change either.
- Gains on problems base solved 1 to 2 times out of 8 (+8) cancelled by losses on 3 to 5 (-5) and 6 to 7 (-3).
- Training on the model's own correct answers gives it nothing new. GRPO and DPO got +3 on the same dev set from the same samples because they also see the wrong answers. Not taken to the test set since no checkpoint beat base on dev.

## 7 Oct 2026: SFT redo on the R1 traces

Same fixed code, 6,000 of the original R1 traces (median 2,606 tokens), one epoch, 375 steps at 30 seconds each, 3.2 hours. Mask check passed. Loss 0.80 to 0.51.

- Dev accuracy 85.4 to 54.2 (p < 1e-30). Checkpoints range 52.6 to 56.2, so it is bad from step 100 onwards.
- Same failure as the first SFT: a third of answers hit the 2,048-token limit, 90% of those are repetition loops, and only two thirds ever close `</think>`. Median answer 1,129 tokens against 277 for base.
- Where it finishes it is still behind base on the same problems (82.6 vs 89.3). Base solves 78% of the problems it loops on.

So the two code bugs were real but not the cause. A 1.5B model trained on long R1 traces copies the self-checking style without the ability to resolve it, and that produces loops no matter how the loss is set up. The SFT redo is closed: on its own answers it is flat, on R1 traces it is harmful, both on dev, neither taken to test.
