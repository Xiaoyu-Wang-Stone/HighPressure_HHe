# High-pressure H/He mixtures

Supporting information for “Hydrogen–helium immiscibility boundary in planets”.

### Repository layout

```
HighPressure_HHe/
├─ DFT_calculations/        # Reference DFT datasets, EOS data, convergence tests, and train/test sets
├─ MLPs/                    # Fitted machine-learning potentials and training assets (N2P2, CACE, MACE)
├─ MLPs_benchmark/          # Benchmark results for the fitted MLPs
├─ MLP_MD/                  # MD and PIMD setup, simulation scripts, and analysis utilities
├─ MLP-MD_results/          # Immiscibility boundaries, chemical-potential data, S(k) results, and RK fits
└─ Manuscript_plots_Rawdata/ # Raw data, scripts, and notebooks used for manuscript figures
```

## Training dataset 
The training dataset is provided in the ./DFT_calculations directory.
Note that some large datasets are stored as split archives to stay within GitHub file-size limits.

## MLP packages 
- N2P2: [https://github.com/CompPhysVienna/n2p2](https://github.com/CompPhysVienna/n2p2)

- CACE: [https://github.com/BingqingCheng/cace](https://github.com/BingqingCheng/cace)

- MACE: [https://github.com/ACEsuit/mace](https://github.com/ACEsuit/mace)
  
Training scripts and fitted MLPs are provided in ./MLPs .

## MLP MD and related analysis scripts
Molecular dynamics inputs, runs, and analysis utilities are provided in the ./MLP_MD directory.
- If you only need the final scientific outputs, `MLP-MD_results/` is the fastest place to begin.
    - `./Immiscibility_boundaries/`: chemical-potential results in HDF5 format and contour data in CSV format for PBE, vdW-DF, and HSE.
    - `./RK-model-fits/`: Redlich-Kister model fits and phase-boundary summaries.
    - `./Sk-values-all/`: aggregated structure-factor datasets.
    - `./System_size_tests/`: size-convergence and RDF comparison results.
    - `./PIMD_CMD_compare/`: classical MD vs PIMD comparison data and scripts.
      
As for the S0 method, please also refer to: https://github.com/BingqingCheng/S0

## Results 
The ./MLP-MD_results directory contains determined immiscibility boundaries and Redlich–Kister model fits.

