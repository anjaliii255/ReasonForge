import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
NB_DIR = ROOT / "notebooks"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def writefile(name):
    return code(f"%%writefile {name}\n" + (ANALYSIS / name).read_text())


def has_outputs(path):
    if not path.exists():
        return False
    return any(c.get("outputs") for c in json.loads(path.read_text())["cells"])


def save(name, cells):
    path = NB_DIR / name
    if has_outputs(path) and "--force" not in sys.argv:
        print("skip", path, "(has run outputs; use --force to overwrite)")
        return
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (NB_DIR / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", NB_DIR / name)


INSTALL = code(r'''
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
!pip install --upgrade -qqq uv
try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
except: _numpy = "numpy"; _pil = "pillow"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
_vllm, _triton = ('vllm==0.11.2', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')
!uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
!uv pip install -qqq {_triton} "huggingface_hub>=0.34.0" "datasets==4.3.0"
try:
    import importlib.metadata as _md; _torch_v = tuple(int(_p) for _p in _md.version("torch").split("+")[0].split(".")[:2])
except Exception:
    _torch_v = ()
_torchao = "torchao>=0.16.0" if _torch_v >= (2, 10) else "torchao>=0.16.0,<0.18.0"
!uv pip install -qqq --no-deps --upgrade "{_torchao}"
!uv pip install -qqq transformers==4.56.2
!uv pip install -qqq --no-deps trl==0.22.2
!uv pip install -qqq math-verify
print("is_t4:", is_t4, "| vllm:", _vllm)
''')

ENV = code(r'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TF"] = "1"; os.environ["USE_TF"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import importlib.metadata as md
for pkg in ["torch", "vllm", "transformers", "trl", "peft", "unsloth", "datasets", "math-verify"]:
    try: print(f"{pkg:13s} {md.version(pkg)}")
    except md.PackageNotFoundError: print(f"{pkg:13s} NOT INSTALLED")

import torch
print("GPU:", torch.cuda.get_device_name(0), f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
''')


def build_05():
    cells = [
        md("# 05 - Fast evaluation with vLLM (base + every adapter found)"),
        md("## 1. Installations"),
        INSTALL,
        md("## 2. Environment check"),
        ENV,
        md("## 3. Shared extractor + eval helpers"),
        writefile("rf_extract.py"),
        writefile("rf_eval.py"),
        md("## 4. Config + adapter discovery"),
        code(r'''
import glob, os, shutil

MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"
DATASETS  = ["gsm8k", "math500"]
STYLES    = ["boxed", "plain"]
MAX_NEW   = 4096
N_LIMIT   = None
OUT_DIR   = "/kaggle/working/results"
os.makedirs(OUT_DIR, exist_ok=True)

for p in glob.glob("/kaggle/input/**/results/*.pkl", recursive=True):
    dst = os.path.join(OUT_DIR, os.path.basename(p))
    if not os.path.exists(dst):
        shutil.copy(p, dst); print("resuming from", p)

def short_name(path):
    s = path.lower()
    if "grpo" in s:
        return "grpo_b" if ("grpo_b" in s or "grpo-b" in s) else "grpo_a"
    if "sft" in s:
        return "sft"
    return os.path.basename(path.rstrip("/"))

ADAPTERS = {}
for cfg in sorted(glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)):
    d = os.path.dirname(cfg)
    if "checkpoint-" in d:
        continue
    ADAPTERS.setdefault(short_name(d), d)
print("adapters:", ADAPTERS if ADAPTERS else "NONE (base only) — did you Add Data?")
'''),
        md("## 5. Load vLLM"),
        code(r'''
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model=MODEL_ID,
    dtype="half",
    max_model_len=MAX_NEW + 1024,
    gpu_memory_utilization=0.85,
    enable_lora=bool(ADAPTERS),
    max_lora_rank=64,
    max_loras=1,
    seed=0,
)
tokenizer = llm.get_tokenizer()

GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, repetition_penalty=1.0)
SAMPLED = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=MAX_NEW, seed=0)
SAMPLED_FOR = ["base", "sft"]

RUNS = {"base": (None, GREEDY, STYLES)}
for i, (name, path) in enumerate(ADAPTERS.items(), start=1):
    RUNS[name] = (LoRARequest(name, i, path), GREEDY, STYLES)
for name in SAMPLED_FOR:
    if name in RUNS:
        RUNS[f"{name}_t06"] = (RUNS[name][0], SAMPLED, ["boxed"])
print("runs:", {k: v[2] for k, v in RUNS.items()})
'''),
        md("## 6. Generate"),
        code(r'''
import time
from rf_eval import load_eval_set, chat_prompts, rows_from_outputs, quick_acc, load_result, save_result

summary = []
for ds_name in DATASETS:
    items = load_eval_set(ds_name, N_LIMIT)
    for style in STYLES:
        path = f"{OUT_DIR}/{ds_name}_{style}.pkl"
        result = load_result(path, {"dataset": ds_name, "style": style, "n": len(items),
                                    "max_new": MAX_NEW, "model_id": MODEL_ID})
        prompts = chat_prompts(tokenizer, items, style)
        for name, (lora, sp, styles) in RUNS.items():
            if style not in styles:
                continue
            if len(result.get(name, [])) == len(items):
                print(f"[skip] {ds_name}/{style}/{name} already done")
            else:
                t0 = time.time()
                outs = llm.generate(prompts, sp, lora_request=lora)
                result[name] = rows_from_outputs(items, outs)
                result["config"].setdefault("decoding", {})[name] = f"temperature={sp.temperature}, top_p={sp.top_p}"
                save_result(path, result)
                print(f"[done] {ds_name}/{style}/{name}: {(time.time()-t0)/60:.1f} min")
            rows = result[name]
            summary.append((ds_name, style, name, quick_acc(rows),
                            sum(r["truncated"] for r in rows) / len(rows),
                            sum(r["has_box"] for r in rows) / len(rows)))

print(f"\n{'dataset':8s} {'style':6s} {'model':8s} {'acc':>6s} {'trunc':>6s} {'boxed':>6s}")
for d, s, m, a, t, b in summary:
    print(f"{d:8s} {s:6s} {m:8s} {a*100:5.1f}% {t*100:5.1f}% {b*100:5.1f}%")
'''),
        md("## 7. Package for download"),
        code(r'''
shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working", "results")
print("download /kaggle/working/results.zip from the Output panel")
'''),
    ]
    save("05-eval-vllm.ipynb", cells)


def build_06():
    cells = [
        md("# 06 - GRPO on GSM8K from base Qwen2.5-1.5B-Instruct"),
        md("## 1. Installations"),
        INSTALL,
        md("## 2. Environment check"),
        ENV,
        md("## 3. Shared extractor + eval helpers"),
        writefile("rf_extract.py"),
        writefile("rf_eval.py"),
        md("## 4. Config"),
        code(r"""
MODE = "smoke"

MODEL_ID          = "unsloth/Qwen2.5-1.5B-Instruct"
LOAD_4BIT         = False
LORA_R, LORA_ALPHA = 32, 64

NUM_GENERATIONS   = 8
PROMPTS_PER_STEP  = 4
MAX_PROMPT_LEN    = 320
MAX_COMPLETION    = 1024
LR                = 1e-5
BETA              = 0.0
NUM_ITERATIONS    = 1
EPSILON, EPSILON_HIGH = 0.2, 0.28
EDGE_WEIGHT       = 0.25

MAX_STEPS, SAVE_STEPS, EVAL_N, TIME_BUDGET_HOURS = {
    "smoke": (5, 5, 50, 1.0),
    "pilot": (100, 25, None, 3.0),
    "full":  (600, 50, None, 9.0),
}[MODE]

RUN_NAME  = f"grpo_a_{MODE}"
OUT_DIR   = f"/kaggle/working/{RUN_NAME}"
FINAL_DIR = f"/kaggle/working/{RUN_NAME}-final"
RES_DIR   = "/kaggle/working/results"
os.makedirs(RES_DIR, exist_ok=True)
print(f"{MODE}: {MAX_STEPS} steps x {PROMPTS_PER_STEP} prompts x {NUM_GENERATIONS} samples")
"""),
        md("## 5. Weights & Biases"),
        code(r"""
REPORT_TO = "none"
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    os.environ["WANDB_PROJECT"] = "reasonforge"
    REPORT_TO = "wandb"
except Exception as e:
    print("W&B disabled:", type(e).__name__)
print("report_to:", REPORT_TO)
"""),
        md("## 6. Load model"),
        code(r"""
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_PROMPT_LEN + MAX_COMPLETION + 64,
    load_in_4bit=LOAD_4BIT,
    fast_inference=True,
    max_lora_rank=LORA_R,
    gpu_memory_utilization=0.85,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print(f"allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
"""),
        md("## 7. Training set from the base samples"),
        code(r"""
import glob, pickle
import numpy as np
from datasets import Dataset
from rf_extract import user_prompt, extract_boxed, answers_match

SAMPLES = glob.glob("/kaggle/input/**/gsm8k_train_k8.pkl", recursive=True)[0]
rows = pickle.load(open(SAMPLES, "rb"))["rows"]
train_rows = [r for r in rows if r["split"] == "train"]
dev_rows = [r for r in rows if r["split"] == "dev"]
K = len(train_rows[0]["samples"])

passes = np.array([sum(s["reward"] for s in r["samples"]) for r in train_rows])
weights = np.where((passes == 0) | (passes == K), EDGE_WEIGHT, 1.0)
n_prompts = MAX_STEPS * PROMPTS_PER_STEP
rng = np.random.default_rng(42)
idx = rng.choice(len(train_rows), size=n_prompts, replace=n_prompts > len(train_rows), p=weights / weights.sum())

train = Dataset.from_list([{
    "prompt": [{"role": "user", "content": user_prompt(train_rows[i]["question"], "boxed")}],
    "gold": train_rows[i]["gold"],
    "pass_rate": int(passes[i]),
} for i in idx])

print(f"samples file: {SAMPLES}")
print(f"train problems {len(train_rows)}  dev {len(dev_rows)}  prompts for this run {len(train)}")
print("pass-rate mix in this run:", np.bincount([int(passes[i]) for i in idx], minlength=K + 1).tolist())
print(train[0]["prompt"][0]["content"], "\n-> gold:", train[0]["gold"])
"""),
        md("## 8. Reward"),
        code(r"""
_calls = {"n": 0}

def correctness_reward(completions, gold, **kwargs):
    texts = [c[0]["content"] for c in completions]
    rewards = [1.0 if answers_match(extract_boxed(t), g) else 0.0 for t, g in zip(texts, gold)]
    _calls["n"] += 1
    if _calls["n"] % 25 == 1:
        print(f"\n--- sample (call {_calls['n']}) gold={gold[0]} pred={extract_boxed(texts[0])} r={rewards[0]}\n"
              f"{texts[0][-500:]}\n---")
    return rewards
"""),
        md("## 9. Trainer"),
        code(r"""
import json, re, time
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

def step_of(path):
    return int(re.search(r"checkpoint-(\d+)", path).group(1))

def latest_checkpoint():
    cands = glob.glob("/kaggle/input/**/checkpoint-*/trainer_state.json", recursive=True)
    cands += glob.glob(f"{OUT_DIR}/checkpoint-*/trainer_state.json")
    cands = [os.path.dirname(c) for c in cands if RUN_NAME in c]
    return max(cands, key=step_of, default=None)

RESUME = None if MODE == "smoke" else latest_checkpoint()
print("resume from:", RESUME)

class TimeBudgetAndLogs(TrainerCallback):
    def __init__(self, hours):
        self.deadline = time.time() + hours * 3600
        self.t_last = None
    def on_step_end(self, args, state, control, **kw):
        now = time.time()
        if self.t_last is not None and state.global_step % 5 == 0:
            print(f"step {state.global_step}: {now - self.t_last:.0f}s/step")
        self.t_last = now
        if now > self.deadline:
            print(f"time budget reached at step {state.global_step} -> saving and stopping")
            control.should_save = True
            control.should_training_stop = True
    def on_save(self, args, state, control, **kw):
        with open(f"{args.output_dir}/log_history.json", "w") as f:
            json.dump(state.log_history, f)

args = GRPOConfig(
    output_dir=OUT_DIR,
    run_name=RUN_NAME,
    report_to=REPORT_TO,
    seed=42,
    learning_rate=LR,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    optim="adamw_8bit",
    adam_beta1=0.9, adam_beta2=0.99,
    weight_decay=0.0,
    max_grad_norm=0.2,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    per_device_train_batch_size=NUM_GENERATIONS,
    gradient_accumulation_steps=PROMPTS_PER_STEP,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION,
    temperature=1.0,
    use_vllm=True,
    mask_truncated_completions=True,
    loss_type="dapo",
    beta=BETA,
    num_iterations=NUM_ITERATIONS,
    epsilon=EPSILON,
    epsilon_high=EPSILON_HIGH,
    max_steps=MAX_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=None,
    logging_steps=1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward],
    args=args,
    train_dataset=train,
    callbacks=[TimeBudgetAndLogs(TIME_BUDGET_HOURS)],
)
"""),
        md("## 10. Train"),
        code(r"""
t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME)
print(f"training wall time: {(time.time()-t0)/3600:.2f} h")

model.save_lora(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
with open(f"{FINAL_DIR}/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f)
"""),
        md("## 11. Training curve"),
        code(r"""
keys = ["reward", "reward_std", "frac_reward_zero_std", "completions/mean_length",
        "completions/clipped_ratio", "loss", "grad_norm", "kl"]
hist = [h for h in trainer.state.log_history if "reward" in h]
every = max(1, len(hist) // 20)
print(f"{'step':>5s} " + " ".join(f"{k.split('/')[-1][:12]:>12s}" for k in keys))
for h in hist[::every] + ([hist[-1]] if hist[-1] not in hist[::every] else []):
    print(f"{h['step']:5d} " + " ".join(f"{h[k]:12.3f}" if k in h else f"{'-':>12s}" for k in keys))
"""),
        md("## 12. Dev accuracy per checkpoint"),
        code(r"""
import shutil
from vllm import SamplingParams
from rf_eval import chat_prompts, graded, save_result

sp = SamplingParams(temperature=0.0, max_tokens=MAX_COMPLETION, repetition_penalty=1.0)
dev_items = dev_rows[:EVAL_N] if EVAL_N else dev_rows
prompts = chat_prompts(tokenizer, dev_items, "boxed")

ckpts = sorted(glob.glob(f"{OUT_DIR}/checkpoint-*"), key=step_of)
runs = [("base", None)] + [(f"step{step_of(c)}", c) for c in ckpts] + [("final", FINAL_DIR)]

dev = {"config": {"n": len(dev_items), "max_new": MAX_COMPLETION, "mode": MODE}}
print(f"{'model':>10s} {'dev acc':>8s} {'trunc':>6s} {'tokens':>7s}")
for name, path in runs:
    lora = model.load_lora(path) if path else None
    outs = model.fast_generate(prompts, sampling_params=sp, lora_request=lora)
    dev[name] = [{**it, **graded(o.outputs[0], it["gold"])} for it, o in zip(dev_items, outs)]
    acc = sum(r["reward"] for r in dev[name]) / len(dev[name])
    trunc = sum(r["truncated"] for r in dev[name]) / len(dev[name])
    toks = sorted(r["n_new_tokens"] for r in dev[name])[len(dev[name]) // 2]
    print(f"{name:>10s} {acc*100:7.1f}% {trunc*100:5.1f}% {toks:7d}")
    save_result(f"{RES_DIR}/dev_{RUN_NAME}.pkl", dev)
"""),
        md("## 13. Package for download"),
        code(r"""
shutil.make_archive(f"/kaggle/working/{RUN_NAME}_adapter", "zip", FINAL_DIR)
shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working", "results")
print(f"download: {RUN_NAME}_adapter.zip, results.zip")
"""),
    ]
    save("06-grpo-base.ipynb", cells)


def build_07():
    cells = [
        md("# 07 - Sample base 8x on GSM8K train"),
        md("## 1. Installations"),
        INSTALL,
        md("## 2. Environment check"),
        ENV,
        md("## 3. Shared extractor + eval helpers"),
        writefile("rf_extract.py"),
        writefile("rf_eval.py"),
        md("## 4. Config"),
        code(r"""
import glob, os, shutil

MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
K           = 8
TEMPERATURE = 1.0
MAX_NEW     = 1024
CHUNK       = 500
N_LIMIT     = None
OUT_DIR     = "/kaggle/working/samples"
OUT         = f"{OUT_DIR}/gsm8k_train_k{K}.pkl"
os.makedirs(OUT_DIR, exist_ok=True)

for p in glob.glob("/kaggle/input/**/samples/*.pkl", recursive=True):
    dst = os.path.join(OUT_DIR, os.path.basename(p))
    if not os.path.exists(dst):
        shutil.copy(p, dst); print("resuming from", p)
"""),
        md("## 5. Load vLLM"),
        code(r"""
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_ID,
    dtype="half",
    max_model_len=MAX_NEW + 512,
    gpu_memory_utilization=0.85,
    seed=0,
)
tokenizer = llm.get_tokenizer()

GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, repetition_penalty=1.0)
SAMPLED = SamplingParams(n=K, temperature=TEMPERATURE, top_p=1.0, max_tokens=MAX_NEW,
                         repetition_penalty=1.0, seed=0)
"""),
        md("## 6. Generate"),
        code(r"""
import time
from rf_eval import gsm8k_train, chat_prompts, sample_rows, sample_summary, load_result, save_result

items = gsm8k_train()
if N_LIMIT:
    items = items[:N_LIMIT]

result = load_result(OUT, {"dataset": "gsm8k_train", "style": "boxed", "n": len(items), "k": K,
                           "temperature": TEMPERATURE, "max_new": MAX_NEW, "model_id": MODEL_ID})
rows = result.setdefault("rows", [])
print(f"{len(rows)}/{len(items)} problems already done")

t0 = time.time()
for start in range(len(rows), len(items), CHUNK):
    chunk = items[start:start + CHUNK]
    prompts = chat_prompts(tokenizer, chunk, "boxed")
    greedy = llm.generate(prompts, GREEDY)
    sampled = llm.generate(prompts, SAMPLED)
    rows.extend(sample_rows(chunk, greedy, sampled))
    save_result(OUT, result)
    s = sample_summary(rows)
    print(f"[{len(rows)}/{len(items)}] greedy {s['greedy']*100:.1f}%  pass@{K} {s['pass_any']*100:.1f}%  "
          f"mixed {s['mixed']*100:.1f}%  ({(time.time()-t0)/60:.0f} min)", flush=True)
"""),
        md("## 7. Summary"),
        code(r"""
import json

summary = {}
for split in ["train", "dev"]:
    part = [r for r in rows if r["split"] == split]
    if not part:
        continue
    s = summary[split] = sample_summary(part)
    print(f"\n{split} (n={s['n']})")
    print(f"  greedy            {s['greedy']*100:5.1f}%")
    print(f"  mean of samples   {s['sample_mean']*100:5.1f}%")
    print(f"  majority@{K}        {s['majority']*100:5.1f}%")
    print(f"  pass@{K}            {s['pass_any']*100:5.1f}%")
    print(f"  mixed (1..{K-1} of {K}) {s['mixed']*100:5.1f}%")
    print(f"  truncated {s['truncated']*100:.1f}%  looping {s['looping']*100:.1f}%  median tokens {s['median_tokens']}")
    print("  correct out of", K, ":", "  ".join(f"{i}:{c}" for i, c in enumerate(s["hist"])))

with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
"""),
        md("## 8. Package for download"),
        code(r"""
shutil.make_archive("/kaggle/working/samples", "zip", "/kaggle/working", "samples")
print("download /kaggle/working/samples.zip from the Output panel")
"""),
    ]
    save("07-sample-base.ipynb", cells)


def build_08():
    cells = [
        md("# 08 - DPO on the base model's own samples"),
        md("## 1. Installations"),
        INSTALL,
        md("## 2. Environment check"),
        ENV,
        md("## 3. Shared extractor + eval helpers"),
        writefile("rf_extract.py"),
        writefile("rf_eval.py"),
        md("## 4. Config"),
        code(r"""
MODE = "smoke"

MODEL_ID          = "unsloth/Qwen2.5-1.5B-Instruct"
LORA_R, LORA_ALPHA = 32, 64

BETA              = 0.1
LR                = 5e-6
BATCH             = 1
GRAD_ACCUM        = 16
MAX_PROMPT_LEN    = 320
MAX_LEN           = 1536
EVAL_MAX_NEW      = 1024

N_PAIRS, SAVE_STEPS, EVAL_N, TIME_BUDGET_HOURS = {
    "smoke": (80, 5, 50, 1.0),
    "pilot": (1600, 25, None, 3.0),
    "full":  (None, 50, None, 9.0),
}[MODE]

RUN_NAME  = f"dpo_{MODE}"
OUT_DIR   = f"/kaggle/working/{RUN_NAME}"
FINAL_DIR = f"/kaggle/working/{RUN_NAME}-final"
RES_DIR   = "/kaggle/working/results"
os.makedirs(RES_DIR, exist_ok=True)
print(f"{MODE}: pairs={N_PAIRS or 'all'}  pairs/step={BATCH*GRAD_ACCUM}")
"""),
        md("## 5. Load model"),
        code(r"""
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_LEN,
    load_in_4bit=False,
    fast_inference=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print(f"allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
"""),
        md("## 6. Preference pairs from the base samples"),
        code(r"""
import glob, pickle, random
from datasets import Dataset
from rf_extract import user_prompt

SAMPLES = glob.glob("/kaggle/input/**/gsm8k_train_k8.pkl", recursive=True)[0]
rows = pickle.load(open(SAMPLES, "rb"))["rows"]
train_rows = [r for r in rows if r["split"] == "train"]
dev_rows = [r for r in rows if r["split"] == "dev"]

rng = random.Random(42)
pairs = []
for r in train_rows:
    right = [s for s in r["samples"] if s["reward"] and not s["truncated"]]
    wrong = [s for s in r["samples"] if not s["reward"]]
    if right and wrong:
        pairs.append({
            "prompt": [{"role": "user", "content": user_prompt(r["question"], "boxed")}],
            "chosen": [{"role": "assistant", "content": rng.choice(right)["resp"]}],
            "rejected": [{"role": "assistant", "content": rng.choice(wrong)["resp"]}],
        })
rng.shuffle(pairs)
if N_PAIRS:
    pairs = pairs[:N_PAIRS]
train = Dataset.from_list(pairs)

lens = sorted(len(tokenizer(p["chosen"][0]["content"]).input_ids) for p in pairs[:500])
print(f"samples file: {SAMPLES}")
print(f"pairs {len(train)} from {len(train_rows)} train problems; chosen median {lens[len(lens)//2]} tokens, p95 {lens[int(len(lens)*0.95)]}")
print(train[0]["prompt"][0]["content"][:200], "...")
"""),
        md("## 7. Trainer"),
        code(r"""
import json, re, time
from transformers import TrainerCallback
from trl import DPOConfig, DPOTrainer
from unsloth import is_bfloat16_supported

def step_of(path):
    return int(re.search(r"checkpoint-(\d+)", path).group(1))

def latest_checkpoint():
    cands = glob.glob("/kaggle/input/**/checkpoint-*/trainer_state.json", recursive=True)
    cands += glob.glob(f"{OUT_DIR}/checkpoint-*/trainer_state.json")
    cands = [os.path.dirname(c) for c in cands if RUN_NAME in c]
    return max(cands, key=step_of, default=None)

RESUME = None if MODE == "smoke" else latest_checkpoint()
print("resume from:", RESUME)

class TimeBudgetAndLogs(TrainerCallback):
    def __init__(self, hours):
        self.deadline = time.time() + hours * 3600
        self.t_last = None
    def on_step_end(self, args, state, control, **kw):
        now = time.time()
        if self.t_last is not None and state.global_step % 5 == 0:
            print(f"step {state.global_step}: {now - self.t_last:.0f}s/step")
        self.t_last = now
        if now > self.deadline:
            print(f"time budget reached at step {state.global_step} -> saving and stopping")
            control.should_save = True
            control.should_training_stop = True
    def on_save(self, args, state, control, **kw):
        with open(f"{args.output_dir}/log_history.json", "w") as f:
            json.dump(state.log_history, f)

args = DPOConfig(
    output_dir=OUT_DIR,
    run_name=RUN_NAME,
    report_to="none",
    seed=42,
    beta=BETA,
    loss_type="sigmoid",
    learning_rate=LR,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    optim="adamw_8bit",
    weight_decay=0.0,
    max_grad_norm=1.0,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=1,
    max_length=MAX_LEN,
    max_prompt_length=MAX_PROMPT_LEN,
    precompute_ref_log_probs=True,
    use_logits_to_keep=True,
    save_steps=SAVE_STEPS,
    save_total_limit=None,
    logging_steps=1,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    train_dataset=train,
    processing_class=tokenizer,
    callbacks=[TimeBudgetAndLogs(TIME_BUDGET_HOURS)],
)
print("steps:", trainer.state.max_steps or len(train) // (BATCH * GRAD_ACCUM))
"""),
        md("## 8. Train"),
        code(r"""
t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME)
print(f"training wall time: {(time.time()-t0)/3600:.2f} h")

model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
with open(f"{FINAL_DIR}/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f)
log_history = trainer.state.log_history
"""),
        md("## 9. Training curve"),
        code(r"""
keys = ["rewards/accuracies", "rewards/margins", "rewards/chosen", "rewards/rejected", "loss", "grad_norm"]
hist = [h for h in log_history if "rewards/margins" in h]
every = max(1, len(hist) // 20)
print(f"{'step':>5s} " + " ".join(f"{k.split('/')[-1][:10]:>10s}" for k in keys))
for h in hist[::every] + ([hist[-1]] if hist[-1] not in hist[::every] else []):
    print(f"{h['step']:5d} " + " ".join(f"{h[k]:10.3f}" if k in h else f"{'-':>10s}" for k in keys))
"""),
        md("## 10. Dev accuracy per checkpoint"),
        code(r"""
import gc, shutil
from rf_eval import chat_prompts, graded, save_result

del trainer, model
gc.collect()
torch.cuda.empty_cache()
print(f"allocated after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model=MODEL_ID,
    dtype="half",
    max_model_len=EVAL_MAX_NEW + 512,
    gpu_memory_utilization=0.6,
    enable_lora=True,
    max_lora_rank=LORA_R,
    max_loras=1,
    seed=0,
)
sp = SamplingParams(temperature=0.0, max_tokens=EVAL_MAX_NEW, repetition_penalty=1.0)
dev_items = dev_rows[:EVAL_N] if EVAL_N else dev_rows
prompts = chat_prompts(tokenizer, dev_items, "boxed")

ckpts = sorted(glob.glob(f"{OUT_DIR}/checkpoint-*"), key=step_of)
runs = [("base", None)] + [(f"step{step_of(c)}", c) for c in ckpts] + [("final", FINAL_DIR)]

dev = {"config": {"n": len(dev_items), "max_new": EVAL_MAX_NEW, "mode": MODE}}
print(f"{'model':>10s} {'dev acc':>8s} {'trunc':>6s} {'tokens':>7s}")
for i, (name, path) in enumerate(runs):
    lora = LoRARequest(name, i + 1, path) if path else None
    outs = llm.generate(prompts, sp, lora_request=lora)
    dev[name] = [{**it, **graded(o.outputs[0], it["gold"])} for it, o in zip(dev_items, outs)]
    acc = sum(r["reward"] for r in dev[name]) / len(dev[name])
    trunc = sum(r["truncated"] for r in dev[name]) / len(dev[name])
    toks = sorted(r["n_new_tokens"] for r in dev[name])[len(dev[name]) // 2]
    print(f"{name:>10s} {acc*100:7.1f}% {trunc*100:5.1f}% {toks:7d}")
    save_result(f"{RES_DIR}/dev_{RUN_NAME}.pkl", dev)
"""),
        md("## 11. Package for download"),
        code(r"""
shutil.make_archive(f"/kaggle/working/{RUN_NAME}_adapter", "zip", FINAL_DIR)
shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working", "results")
print(f"download: {RUN_NAME}_adapter.zip, results.zip")
"""),
    ]
    save("08-dpo-base.ipynb", cells)


if __name__ == "__main__":
    build_05()
    build_06()
    build_07()
    build_08()
