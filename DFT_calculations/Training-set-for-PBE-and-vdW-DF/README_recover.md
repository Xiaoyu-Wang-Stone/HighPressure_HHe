# Training Set Download and Recovery Guide

This repository stores `train-all.xyz` as split archive parts so each file stays below common GitHub upload limits.

## Included files

Download these four files together:

- `train-all.xyz.tar.xz.part-000` (~19.0 MB)
- `train-all.xyz.tar.xz.part-001` (~19.0 MB)
- `train-all.xyz.tar.xz.part-002` (~11.3 MB)
- `train-all.xyz.tar.xz.sha256` (checksum of the merged archive)

All parts are required. Missing any `part-*` file will make recovery fail.

## Quick start (macOS/Linux)

Run in the directory containing all files above:

```bash
# 1) Merge parts
cat train-all.xyz.tar.xz.part-* > train-all.xyz.tar.xz

# 2) Verify integrity (must report: OK)
shasum -a 256 -c train-all.xyz.tar.xz.sha256

# 3) Extract dataset
tar -xJf train-all.xyz.tar.xz
```

After extraction, `train-all.xyz` will be restored.

## Windows notes

Use one of the following environments:

- Git Bash
- WSL (recommended)

Then run the same commands as in the macOS/Linux section.

## Verify your download before merging

```bash
ls -lh train-all.xyz.tar.xz.part-*
ls train-all.xyz.tar.xz.part-* | wc -l
```

The part count should be `3`.

## Troubleshooting

- `No such file or directory`: one or more `part-*` files are missing or filenames were changed.
- Checksum mismatch: re-download all parts in binary mode and verify file sizes.
- Extraction error: remove `train-all.xyz.tar.xz`, merge again, then re-run checksum and extraction.

## For maintainers: regenerate split package

```bash
tar -cJf train-all.xyz.tar.xz train-all.xyz
split -b 19m -d -a 3 train-all.xyz.tar.xz train-all.xyz.tar.xz.part-
shasum -a 256 train-all.xyz.tar.xz > train-all.xyz.tar.xz.sha256
```
