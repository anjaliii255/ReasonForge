import os
import pickle
import random
from collections import Counter

from rf_extract import answers_match, extract_answer, extract_boxed, gsm8k_gold, has_boxed, user_prompt

DEV_SIZE = 500


def gsm8k_train(split=None, seed=0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    dev = set(order[:DEV_SIZE])
    items = [{"id": i, "split": "dev" if i in dev else "train",
              "question": r["question"], "gold": gsm8k_gold(r["answer"])}
             for i, r in enumerate(ds)]
    return [it for it in items if split in (None, it["split"])]


def load_eval_set(name, n=None):
    from datasets import load_dataset
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"question": r["question"], "gold": gsm8k_gold(r["answer"])} for r in ds]
    elif name == "gsm8k_dev":
        items = gsm8k_train("dev")
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"question": r["problem"], "gold": r["answer"],
                  "subject": r["subject"], "level": r["level"]} for r in ds]
    else:
        raise ValueError(name)
    return items[:n] if n else items


def chat_prompts(tokenizer, items, style):
    return [tokenizer.apply_chat_template([{"role": "user", "content": user_prompt(it["question"], style)}],
                                          tokenize=False, add_generation_prompt=True)
            for it in items]


def is_looping(text, tail=3000, n=8, min_repeats=5):
    words = text[-tail:].split()
    if len(words) < 50:
        return False
    grams = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n))
    return grams.most_common(1)[0][1] >= min_repeats


def completion_row(c):
    return {"resp": c.text, "n_new_tokens": len(c.token_ids),
            "truncated": int(c.finish_reason == "length"),
            "has_box": has_boxed(c.text), "looping": is_looping(c.text)}


def rows_from_outputs(items, outputs):
    return [{**it, **completion_row(out.outputs[0])} for it, out in zip(items, outputs)]


def graded(c, gold):
    row = completion_row(c)
    row["pred"] = extract_boxed(c.text)
    row["reward"] = int(answers_match(row["pred"], gold))
    return row


def sample_rows(items, greedy_outputs, sampled_outputs):
    return [{**it, "greedy": graded(g.outputs[0], it["gold"]),
             "samples": [graded(c, it["gold"]) for c in s.outputs]}
            for it, g, s in zip(items, greedy_outputs, sampled_outputs)]


def majority_correct(row):
    preds = [s["pred"] for s in row["samples"] if s["pred"] is not None]
    return bool(preds) and answers_match(Counter(preds).most_common(1)[0][0], row["gold"])


def sample_summary(rows):
    n, k = len(rows), len(rows[0]["samples"])
    passes = [sum(s["reward"] for s in r["samples"]) for r in rows]
    samples = [s for r in rows for s in r["samples"]]
    tokens = sorted(s["n_new_tokens"] for s in samples)
    return {
        "n": n, "k": k,
        "greedy": sum(r["greedy"]["reward"] for r in rows) / n,
        "sample_mean": sum(passes) / (n * k),
        "majority": sum(majority_correct(r) for r in rows) / n,
        "pass_any": sum(p > 0 for p in passes) / n,
        "mixed": sum(0 < p < k for p in passes) / n,
        "hist": [passes.count(i) for i in range(k + 1)],
        "truncated": sum(s["truncated"] for s in samples) / len(samples),
        "looping": sum(s["looping"] for s in samples) / len(samples),
        "median_tokens": tokens[len(tokens) // 2],
    }


def quick_acc(rows):
    return sum(answers_match(extract_answer(r["resp"]), r["gold"]) for r in rows) / max(len(rows), 1)


def load_result(path, config):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {"config": config}


def save_result(path, result):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    os.replace(tmp, path)
