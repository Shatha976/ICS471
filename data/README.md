# Data folder

The dataset is not stored in this repository. Download two files from IEEE DataPort
(Open Access, free IEEE account needed):

https://doi.org/10.21227/x46j-sk98

| File on the dataset page | Size | What it holds |
|---|---|---|
| `DATASET-1_(224 X 224).zip` | 108 MB | all 8,079 images resized to 224x224 |
| `DATASET-1_(annotation files).zip` | 104 KB | the label files (.txt) |

Do not download the 42 GB file with the full-resolution images. The code does not use it.

Unzip so the folder looks like this:

```
data/
  DATASET-1 (224 X 224)/     <- 8,079 .JPG files
  annotations/               <- 10 .txt files
```

Commands (run from the repository root):

```
unzip "DATASET-1_(224 X 224).zip" -d data/
unzip "DATASET-1_(annotation files).zip" -d data/annotations
```

If your zip creates a differently named image folder, pass its path with `--img-dir`.
