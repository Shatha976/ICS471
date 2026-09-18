#!/usr/bin/env python3
"""
Dataset exploration and fixed palm-level split for the KSU Date Fruit dataset
(DATASET-1: 8,079 images of date bunches, 29 palms, 5 varieties).

Dataset:  Altaheri et al., "Date Fruit Dataset for Automated Harvesting and Visual
          Yield Estimation", IEEE DataPort, doi:10.21227/x46j-sk98
Article:  doi:10.1016/j.dib.2019.104514

Inputs (download from IEEE DataPort, free IEEE account needed):
  DATASET-1 (224 X 224).zip        -> --img-dir   (folder with the 8,079 .JPG files)
  DATASET-1 (annotation files).zip -> --annot-dir (folder with the .txt label files)

Outputs (relative to --out-dir):
  figures/sample_grid.png            random samples for each of the 7 maturity classes
  figures/sample_varieties.png       random samples for each of the 5 varieties
  figures/class_distribution.png     maturity-class and variety counts per split
  figures/palm_split.png             images per palm, coloured by split
  results/dataset_stats.json         numbers quoted in the proposal
  results/split_manifest.csv         one row per image: palm, variety, labels, split

Usage:
  python src/explore_dataset.py --img-dir "data/DATASET-1 (224 X 224)" \
      --annot-dir data/annotations --out-dir .
"""
import argparse
import collections as C
import csv
import json
import os
import random
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SEED = 42
N_SPLIT_TRIALS = 60000
VARIETY = {"B": "Barhi", "K": "Khalas", "M": "Meneifi", "N": "NabootSaif", "S": "Sullaj"}
MATURITY_LABEL = {
    "1_Immature-1": "1 Immature-1",
    "2_Immature-2": "2 Immature-2",
    "3_Pre-Khalal": "3 Pre-Khalal",
    "4_Khalal": "4 Khalal",
    "5_KhalalwithRutab": "5 Khalal\nwith Rutab",
    "6_Pre-Tamar": "6 Pre-Tamar",
    "7_Tamar": "7 Tamar",
}
NAME_RE = re.compile(r"^([A-Za-z])(\d+)\.(S\d+(?:_\d+)?)\.B\.(\d+)\.JPG$", re.I)
SPLIT_COLOURS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}


def read_labels(path):
    """Annotation lines look like: \\DATASET-1\\B1.S1.B.1.JPG 1_Immature-1"""
    out = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                out[parts[0].split("\\")[-1]] = parts[1]
    return out


def choose_palm_split(palms_by_variety, palm_class_counts, palm_index, total, n_total):
    """Random search over palm assignments.

    Every variety must appear in val and test (one palm each; Sullaj, with 9 palms,
    gets 1 or 2). Every maturity class must appear in val and test. The score is the
    largest deviation of any class share (and of the overall share) from 15%.
    The score only looks at label counts, never at model results.
    """
    rng = np.random.default_rng(SEED)
    best = None
    for _ in range(N_SPLIT_TRIALS):
        test, val = [], []
        for letter, palms in palms_by_variety.items():
            palms = list(rng.permutation(palms))
            n_t = int(rng.integers(1, 3)) if letter == "S" else 1
            n_v = int(rng.integers(1, 3)) if letter == "S" else 1
            test += palms[:n_t]
            val += palms[n_t:n_t + n_v]
        ct = palm_class_counts[[palm_index[p] for p in test]].sum(0)
        cv = palm_class_counts[[palm_index[p] for p in val]].sum(0)
        if (ct == 0).any() or (cv == 0).any():
            continue
        score = 0.0
        for c in (cv, ct):
            score += np.max(np.abs(c / total - 0.15)) + abs(c.sum() / n_total - 0.15)
        if best is None or score < best[0]:
            best = (score, [str(p) for p in test], [str(p) for p in val])
    return best


def grid(paths_by_col, col_titles, rows, out, title):
    fig, axes = plt.subplots(rows, len(col_titles), figsize=(1.9 * len(col_titles), 1.95 * rows + 0.6),
                             squeeze=False)
    for c, (paths, t) in enumerate(zip(paths_by_col, col_titles)):
        for r in range(rows):
            ax = axes[r, c]
            ax.imshow(Image.open(paths[r]).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(t, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--annot-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    random.seed(SEED)

    # ---------- images
    files = sorted(f for f in os.listdir(args.img_dir) if f.lower().endswith(".jpg"))
    meta, bad, sizes, modes = {}, [], C.Counter(), C.Counter()
    for f in files:
        m = NAME_RE.match(f)
        if not m:
            bad.append(f)
            continue
        with Image.open(os.path.join(args.img_dir, f)) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
        meta[f] = dict(palm=m.group(1).upper() + m.group(2), letter=m.group(1).upper(), session=m.group(3))
    assert not bad, f"unparsed file names: {bad[:5]}"

    # ---------- labels
    rl = lambda n: read_labels(os.path.join(args.annot_dir, n))  # noqa: E731
    authors = {
        "types": (rl("DateTypes_train.txt"), rl("DateTypes_test.txt")),
        "maturity7": (rl("DateMaturity_7-classes_train.txt"), rl("DateMaturity_7-classes_test.txt")),
    }
    variety_lab = {**authors["types"][0], **authors["types"][1]}
    maturity_lab = {**authors["maturity7"][0], **authors["maturity7"][1]}
    classes = sorted(set(maturity_lab.values()))
    cidx = {c: i for i, c in enumerate(classes)}

    # ---------- leakage in the authors' own train/test files
    leak = {}
    for name, (tr, te) in authors.items():
        seen = {(meta[f]["palm"], meta[f]["session"]) for f in tr}
        shared = sum(1 for f in te if (meta[f]["palm"], meta[f]["session"]) in seen)
        leak[name] = dict(
            palms_in_both=len({meta[f]["palm"] for f in tr} & {meta[f]["palm"] for f in te}),
            test_images=len(te),
            test_images_sharing_palm_and_session_with_train=shared,
        )

    # ---------- fixed palm-level split (maturity-labelled images drive the search)
    palms = sorted({v["palm"] for v in meta.values()})
    pidx = {p: i for i, p in enumerate(palms)}
    pcc = np.zeros((len(palms), len(classes)), dtype=int)
    for f, lab in maturity_lab.items():
        pcc[pidx[meta[f]["palm"]], cidx[lab]] += 1
    by_var = C.defaultdict(list)
    for p in palms:
        by_var[p[0]].append(p)
    score, test_palms, val_palms = choose_palm_split(by_var, pcc, pidx, pcc.sum(0), pcc.sum())
    split_of_palm = {p: "train" for p in palms}
    split_of_palm.update({p: "val" for p in val_palms})
    split_of_palm.update({p: "test" for p in test_palms})

    # ---------- manifest
    rows = []
    for f in files:
        m = meta[f]
        rows.append(dict(file=f, palm=m["palm"], variety_code=m["letter"], session=m["session"],
                         variety=variety_lab.get(f, ""), maturity7=maturity_lab.get(f, ""),
                         split=split_of_palm[m["palm"]]))
    with open(os.path.join(res_dir, "split_manifest.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---------- statistics
    def counts(key, only_labelled):
        out = {}
        for s in ("train", "val", "test"):
            sel = [r for r in rows if r["split"] == s and (r[key] or not only_labelled)]
            out[s] = dict(C.Counter(r[key] for r in sel if r[key]))
            out[s]["_total"] = len(sel)
        return out

    stats = {
        "n_images": len(files),
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "image_modes": dict(modes),
        "n_palms": len(palms),
        "palms_per_variety": {VARIETY[k]: len(v) for k, v in sorted(by_var.items())},
        "images_per_palm_min_median_max": [min(C.Counter(m["palm"] for m in meta.values()).values()),
                                           int(np.median(list(C.Counter(m["palm"] for m in meta.values()).values()))),
                                           max(C.Counter(m["palm"] for m in meta.values()).values())],
        "n_variety_labelled": len(variety_lab),
        "n_maturity7_labelled": len(maturity_lab),
        "n_maturity7_unlabelled": len(files) - len(maturity_lab),
        "variety_counts_all": dict(C.Counter(variety_lab.values())),
        "maturity7_counts_all": {k: dict(C.Counter(maturity_lab.values()))[k] for k in classes},
        "authors_split_leakage": leak,
        "split_search": {"trials": N_SPLIT_TRIALS, "seed": SEED, "best_score": round(float(score), 4)},
        "palms": {s: sorted(p for p in palms if split_of_palm[p] == s) for s in ("train", "val", "test")},
        "maturity7_by_split": counts("maturity7", True),
        "variety_by_split": counts("variety", True),
    }
    with open(os.path.join(res_dir, "dataset_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    # ---------- figures
    path = lambda f: os.path.join(args.img_dir, f)  # noqa: E731
    by_class = C.defaultdict(list)
    for f, lab in maturity_lab.items():
        by_class[lab].append(f)
    cols = [[path(f) for f in random.sample(sorted(by_class[c]), 3)] for c in classes]
    grid(cols, [MATURITY_LABEL[c] for c in classes], 3, os.path.join(fig_dir, "sample_grid.png"),
         "Random samples per maturity stage (224x224, seed=42)")
    by_var_files = C.defaultdict(list)
    for f, lab in variety_lab.items():
        by_var_files[lab].append(f)
    vnames = sorted(by_var_files)
    cols = [[path(f) for f in random.sample(sorted(by_var_files[v]), 3)] for v in vnames]
    grid(cols, vnames, 3, os.path.join(fig_dir, "sample_varieties.png"), "Random samples per variety (seed=42)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [7, 5]})
    for ax, keys, label_fn, key, title in (
        (a1, classes, lambda k: MATURITY_LABEL[k].replace("\n", " "), "maturity7", "Maturity stage (7 classes, 6,647 images)"),
        (a2, sorted(stats["variety_counts_all"]), lambda k: k, "variety", "Variety (5 classes, 8,072 images)"),
    ):
        bottom = np.zeros(len(keys))
        for s in ("train", "val", "test"):
            vals = np.array([stats[f"{'maturity7' if key == 'maturity7' else 'variety'}_by_split"][s].get(k, 0) for k in keys])
            ax.bar(range(len(keys)), vals, bottom=bottom, label=s, color=SPLIT_COLOURS[s])
            bottom += vals
        for i, t in enumerate(bottom):
            ax.text(i, t + 20, int(t), ha="center", fontsize=7)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([label_fn(k) for k in keys], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("images")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(bottom) * 1.15)
    a1.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=150)
    plt.close(fig)

    per_palm = C.Counter(m["palm"] for m in meta.values())
    order = sorted(palms, key=lambda p: (p[0], int(p[1:])))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(range(len(order)), [per_palm[p] for p in order], color=[SPLIT_COLOURS[split_of_palm[p]] for p in order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel("images")
    ax.set_title("Images per palm (colour = split); B/K/M/N/S = Barhi/Khalas/Meneifi/NabootSaif/Sullaj", fontsize=9)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in SPLIT_COLOURS.values()],
              labels=list(SPLIT_COLOURS), fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "palm_split.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
