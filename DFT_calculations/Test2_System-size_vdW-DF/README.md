# vdw Training Set Download and Recovery Guide

This directory stores `vdw_training_set.tar` as split archive parts so each file stays below common GitHub upload limits.

## Included files

Download all files together:

- `vdw_training_set.tar.xz.part-000` (~19.0 MB)
- `vdw_training_set.tar.xz.part-001` (~19.0 MB)
- `vdw_training_set.tar.xz.part-002` (~19.0 MB)
- `vdw_training_set.tar.xz.part-003` (~19.0 MB)
- `vdw_training_set.tar.xz.part-004` (~5.3 MB)
- `vdw_training_set.tar.xz.sha256`

All `part-*` files are required.

## Recover the original archive (macOS/Linux)

Run in the directory containing all part files:

```bash
# 1) Merge split parts
cat vdw_training_set.tar.xz.part-* > vdw_training_set.tar.xz

# 2) Verify archive integrity (must report: OK)
shasum -a 256 -c vdw_training_set.tar.xz.sha256

# 3) Decompress to restore vdw_training_set.tar
xz -dk vdw_training_set.tar.xz
```

## Windows notes

Use Git Bash or WSL, then run the same commands above.

## Notes on ASE key names

There was a recent update to the VASP reader and XYZ writer in ASE. Some keys in the XYZ comment line were renamed:

- `Extrapolated_E0_energy=297.85836895`
- `Force_consistent_energy=297.23812176`
- `Energy_without_entropy=298.47861614`

Interpretation:

- Forces are consistent with `Force_consistent_energy` (electronic free energy).
- The actual potential energy is `Energy_without_entropy`.
- `atoms.get_potential_energy()` returns `Extrapolated_E0_energy`.

`Extrapolated_E0_energy` is an extrapolation to zero smearing. It can be useful for low-temperature systems where DFT SCF convergence requires artificially high smearing (for example, magnetic systems, metals, and radicals). It is included for completeness but is not the target energy for this dataset.

## For maintainers: regenerate package

```bash
xz -zk -T0 vdw_training_set.tar
split -b 19m -d -a 3 vdw_training_set.tar.xz vdw_training_set.tar.xz.part-
shasum -a 256 vdw_training_set.tar.xz > vdw_training_set.tar.xz.sha256
```
