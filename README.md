# Date bunch maturity classification (ICS 471 project proposal)

This repository is at the proposal stage. No model has been trained yet. It holds the code
that inspects the dataset, fixes the train/validation/test split, and produces the figures
and numbers used in the proposal PDF.

## Problem

Given one photo of a date bunch on the palm, predict its maturity stage.

- Input: one RGB photo, 224x224 pixels.
- Output: one of seven stages (Immature-1, Immature-2, Pre-Khalal, Khalal, Khalal with Rutab, Pre-Tamar, Tamar).
- Task: multi-class image classification.
- Validation metric (single): macro-averaged F1 over the seven stages.

Growers and harvest crews currently judge ripeness by eye, palm by palm. A stage predictor is
the first piece a phone app or a harvesting robot would need.

## Dataset

Date Fruit Dataset for Automated Harvesting and Visual Yield Estimation (DATASET-1),
Center of Smart Robotics Research, King Saud University.

- Page: https://doi.org/10.21227/x46j-sk98 (IEEE DataPort, Open Access, free IEEE account needed)
- Data article: https://doi.org/10.1016/j.dib.2019.104514
- Photos taken in an orchard near Riyadh between 29 June and 21 September 2016 (session dates come from `ImagingSessions.txt`).
- 8,079 images, 29 palms, 5 varieties (Naboot Saif, Khalas, Barhi, Meneifi, Sullaj). Palms per variety: 5, 5, 5, 5 and 9.
- Labels used: maturity stage for 6,647 images (7 classes), variety for 8,072 images (5 classes).
- License: the dataset page shows no license field. IEEE DataPort says its datasets are released under CC BY and must be cited, so we treat the dataset as CC BY and cite it.

The images are not stored here. See [data/README.md](data/README.md) for what to download.

## Reproduce

```
pip install -r requirements.txt
python src/explore_dataset.py --img-dir "data/DATASET-1 (224 X 224)" --annot-dir data/annotations --out-dir .
```

The script is deterministic (seed 42). It writes:

| File | Content |
|---|---|
| `figures/sample_grid.png` | 3 random images for each maturity stage |
| `figures/sample_varieties.png` | 3 random images for each variety |
| `figures/class_distribution.png` | class counts per split, for maturity stage and for variety |
| `figures/palm_split.png` | images per palm, coloured by split |
| `results/dataset_stats.json` | every number quoted in the proposal |
| `results/split_manifest.csv` | one row per image: palm, variety, labels, split |

## The split (fixed before training)

We split by palm, not by image. Photos of the same palm taken in the same session are, in
practice, burst shots of the same bunch. An image-level split would let a model memorise the
palms and inflate the test score.

Target 70/15/15. The rules: every variety appears in validation and test, and every maturity
stage appears in both. The palm assignment was picked from 60,000 random tries (seed 42) by
scoring only the class shares. No model results were used.

| Split | Palms | Images (maturity task) | Share |
|---|---|---|---|
| train | 19 | 4,585 | 69.0% |
| val | B1, K1, M5, N2, S2 | 1,002 | 15.1% |
| test | B2, K5, M3, N3, S6 | 1,060 | 15.9% |

Palm codes: the letter is the variety (B Barhi, K Khalas, M Meneifi, N Naboot Saif, S Sullaj)
and the number is the palm. The same split works for the variety task (5,499 / 1,322 / 1,251 images).

## What we found while inspecting the data

- The authors' own train/test label files put all 29 palms in both sets. For the maturity task,
  3,418 of their 3,420 test images come from a palm and session that also appear in their training set.
  We do not use their split for this reason.
- 1,432 images have no 7-class maturity label and are left out of the maturity task. 7 images have no variety label.
- Harvest-decision labels are released for test images only (2,682), so that task cannot be trained.
- Class sizes for maturity range from 659 to 1,660 images (ratio 2.5).

## Limitations

- One orchard and one season. Results say how the model does on new palms in this orchard, not on other orchards or years.
- Validation and test have 5 palms each, so scores will move depending on which palms landed there.
- For the variety task, Sullaj has a single test palm (122 images).
- Neighbouring stages look alike (for example Pre-Khalal and Khalal), and some bunches are covered with nets or bags.
- We use the authors' 224x224 resized images, not the full-resolution originals.

## Repository layout

```text
data/
└── README.md                  

figures/
├── class_distribution.png
├── palm_split.png
├── sample_grid.png
└── sample_varieties.png

results/
├── dataset_stats.json         
└── split_manifest.csv
     
src/
└── explore_dataset.py         

```

## Citation

H. Altaheri, M. Alsulaiman, M. Faisal, G. Muhammad, "Date Fruit Dataset for Automated
Harvesting and Visual Yield Estimation," IEEE DataPort, doi:10.21227/x46j-sk98.
