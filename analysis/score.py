import sys, pickle, math, random
from itertools import combinations
from rf_extract import extract_answer, answers_match

SEED, N_BOOT = 0, 2000


def load(path):
    d = pickle.load(open(path, "rb"))
    return d.get("config"), {k: v for k, v in d.items() if isinstance(v, list)}


def score_records(records):
    flags = []
    for r in records:
        pred = extract_answer(r["resp"])
        flags.append(1 if answers_match(pred, r["gold"]) else 0)
    return flags


def bootstrap_ci(flags, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(flags) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def mcnemar(a, b):
    n01 = sum(1 for i in range(len(a)) if a[i] and not b[i])
    n10 = sum(1 for i in range(len(a)) if b[i] and not a[i])
    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, chi2, math.erfc(math.sqrt(chi2 / 2))


def taxonomy(records, flags):
    wrong = [(r, f) for r, f in zip(records, flags) if not f]
    trunc = sum(1 for r, _ in wrong if r.get("truncated"))
    nobox = sum(1 for r, _ in wrong if not r.get("has_box"))
    genuine = sum(1 for r, _ in wrong if not r.get("truncated") and r.get("has_box"))
    return len(wrong), trunc, nobox, genuine


def report(path):
    config, models = load(path)
    print(f"\n{'#' * 70}\nfile: {path}\nconfig: {config}")

    scored = {name: score_records(recs) for name, recs in models.items()}

    print("\n=== accuracy ===")
    for name, flags in scored.items():
        n = len(flags)
        m, lo, hi = bootstrap_ci(flags)
        print(f"{name:8s} {m*100:5.1f}%  95% CI [{lo*100:.1f}, {hi*100:.1f}]  ({sum(flags)}/{n})")

    print("\n=== McNemar (paired, same problems) ===")
    for a, b in combinations(scored, 2):
        if len(scored[a]) != len(scored[b]):
            print(f"{a} vs {b}: different n, skipped")
            continue
        n = len(scored[a])
        only_a, only_b, chi2, p = mcnemar(scored[a], scored[b])
        lift = (sum(scored[b]) - sum(scored[a])) / n * 100
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"{b} vs {a}: {lift:+.1f} pts  ({a}-only={only_a}, {b}-only={only_b}, "
              f"chi2={chi2:.2f}, p={p:.4f}) -> {sig}")

    print("\n=== failure taxonomy ===")
    for name, recs in models.items():
        n = len(recs)
        tot, trunc, nobox, genuine = taxonomy(recs, scored[name])
        print(f"{name:8s} wrong={tot}  truncated={trunc}  no-box={nobox}  genuine-error={genuine}  "
              f"| ceiling if truncations recovered: {(sum(scored[name]) + trunc) / n * 100:.1f}%")

    for key in ("subject", "level"):
        if not all(key in r for recs in models.values() for r in recs):
            continue
        print(f"\n=== accuracy by {key} ===")
        first = next(iter(models.values()))
        groups = sorted({r[key] for r in first})
        print(f"{key:24s} {'n':>4s}  " + "  ".join(f"{name:>8s}" for name in models))
        for g in groups:
            cols = []
            for name, recs in models.items():
                idx = [i for i, r in enumerate(recs) if r[key] == g]
                cols.append(f"{sum(scored[name][i] for i in idx) / len(idx) * 100:7.1f}%")
            print(f"{str(g):24s} {len(idx):4d}  " + "  ".join(cols))

    print("\n=== format / verbosity ===")
    for name, recs in models.items():
        n = len(recs)
        box = sum(1 for r in recs if r.get("has_box"))
        tr = sum(1 for r in recs if r.get("truncated"))
        lens = sorted(len(r["resp"]) for r in recs)
        toks = sorted(r["n_new_tokens"] for r in recs if "n_new_tokens" in r)
        tok_s = f"  median_tokens {toks[len(toks)//2]}" if toks else ""
        print(f"{name:8s} has_box {box/n*100:.0f}%  truncated {tr/n*100:.0f}%  "
              f"median_chars {lens[n//2]}{tok_s}")


def main():
    for path in (sys.argv[1:] or ["eval_raw_v2.pkl"]):
        report(path)


if __name__ == "__main__":
    main()
