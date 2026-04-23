# fMRI Analysis — Task-Based Activation Mapping (t-scores)

Julia pipeline for task-based fMRI GLM analysis, designed to compare activation maps across multiple MRI reconstruction methods (SMS-EPI/slice-GRAPPA, CG-SENSE, L+S, LLR, and multi-scale low-rank / MSLR).

---

## Repository structure

```
src/
  fmri_analysis.jl    # Core library: HRF, design matrix, GLM, correction, plotting
  export.jl           # NIfTI export helpers
scripts/
  main_<session>.jl   # Per-session analysis scripts (one file per scan date)
```

---

## Core library (`src/fmri_analysis.jl`)

### 1. Haemodynamic Response Function
- `canonical_hrf(tr)` — SPM-style double-gamma HRF sampled at `tr` seconds. Uses `SpecialFunctions.gamma`.

### 2. Design Matrix
- `build_design_matrix(onsets, durations, n_scans, tr)` — Constructs an `(n_scans × n_conditions + 1)` GLM design matrix via FFT-based convolution with 16× temporal oversampling. Last column is the intercept.

### 3. GLM Fitting & Contrasts
- `fit_glm(X, Y)` — OLS fit returning `β`, residuals, and `(X'X)⁻¹`.
- `compute_tscores(β, residuals, XtXinv, contrast)` — Voxel-wise t-scores for a contrast vector.
- `run_glm(Y, onsets, durations, contrast, n_scans, tr)` — Full pipeline wrapper: design matrix → fit → t-map.

### 4. Multiple Comparisons Correction
- `fdr_correct(t_map, df; q=0.05)` — Benjamini-Hochberg FDR correction; returns thresholded t-map, binary mask, raw p-values, and t-threshold.
- `bonferroni_correct(t_map, df; alpha=0.05)` — Bonferroni FWER correction; same return signature.
- Both convert t-scores to p-values via `Distributions.TDist`.

### 5. Brain Mask Extraction
- `extract_brain_mask(X; intensity_threshold=0.1, closing_radius=3)` — Derives a binary brain mask from a 4-D BOLD volume using intensity thresholding on the temporal mean, morphological closing, and largest-connected-component selection. Used internally by the analysis pipelines to restrict GLM fitting and FDR correction to brain voxels.

### 6. Visualisation
- `plot_tmap_flat(t_map)` — Two-panel Plots.jl figure: per-voxel bar chart + t-score histogram.
- `plot_design_matrix(X)` — Heatmap of the design matrix (sanity check).
- `plot_tmap_slices(t_vol)` — Orthogonal axial/coronal/sagittal slice view (CairoMakie) with optional anatomical underlay.
- `plot_tmap_slices_shared(t_vol)` — Same as above but accepts an explicit `underlay_range` for consistent normalisation across multiple plots.
- `tmap_summary(t_map)` — Prints a table of voxel counts surviving common t-thresholds (p < .10 down to p < 10⁻⁶).

### 7. Experiment Parameters
- `ExperimentParams` — Struct holding `tr`, `onsets`, `durations`, `contrast`, and `n_discard` (leading frames to drop before fitting).

### 8. Analysis Pipelines
- `analyze_and_plot(X, params, title)` — Runs the full GLM pipeline on a single 4-D volume: extracts a brain mask, fits the GLM on brain voxels only, applies FDR correction to set the display threshold, and displays an orthogonal slice plot with a temporal mean underlay. Returns the slice index used (pass as `ref_slice_idx` to align subsequent plots) and the 3-D t-score volume.
- `analyze_and_plot_mslr(X, params, Nscales, patch_sizes, title; q=0.05)` — Runs the same pipeline on each scale of a 5-D MSLR reconstruction. The brain mask is derived from the temporal mean of the summed reconstruction and reused across all scales. Each scale is thresholded independently by FDR at level `q`. The t-score colour scale and anatomical underlay intensity are normalised globally across scales, so per-scale comparisons are directly interpretable.

### 9. NIfTI Export (`src/export.jl`)
- `export_niftis(X, t_vol, params, prefix, out_dir)` — Exports a post-discard magnitude timeseries and t-score volume as NIfTI files for a single 4-D reconstruction.
- `export_niftis(X, t_vols, patch_sizes, Nscales, params, prefix, out_dir)` — Same for a 5-D MSLR reconstruction; writes one magnitude + t-map pair per scale.

---

## Experiment design (tapping paradigm)

- **Task:** finger-tapping, alternating tap/rest blocks
- **Block duration:** 20 s tap / 20 s rest (40 s period)
- **TR:** 0.8 s
- **Contrast:** tap > rest (`[1, -1, 0]`)
- **Instructional frames:** discarded before fitting (session-dependent; set via `n_discard` in `ExperimentParams`)

---

## Session scripts (`scripts/main_<session>.jl`)

Each session script loads reconstructed volumes from disk, defines an `ExperimentParams`, and calls the analysis pipelines. The first reconstruction analysed (typically the product SMS-EPI/slice-GRAPPA scan) establishes a reference slice index that is reused across all subsequent comparisons so that all plots show the same anatomical location.

Reconstructions compared per session may include:

| Label | Description |
|---|---|
| SMS-EPI + slice-GRAPPA | Product reconstruction (NIfTI) |
| Gaussian / CAIPI / PD + CG-SENSE | Compressed-sensing or parallel-imaging recon (NIfTI) |
| L+S | Low-rank + sparse recon (MAT, scale 2 extracted) |
| LLR | Locally low-rank recon (MAT, scales summed) |
| MSLR (*N* scales) | Multi-scale low-rank recon (MAT, per-scale + summed) |

---

## Dependencies

Add via the Julia package manager (`]add <pkg>`):

```julia
CairoMakie       # 3-D orthogonal slice visualisation
Plots            # flat t-map and design matrix plots
FFTW             # FFT-based HRF convolution
Distributions    # t-distribution CDF for p-value conversion
ImageMorphology  # morphological operations for brain mask extraction
SpecialFunctions # gamma function for HRF construction
MAT              # loading .mat reconstruction files
NIfTI            # loading/writing .nii / .nii.gz volumes
Revise           # hot-reloading during interactive development
```

Standard library modules used (no installation needed): `Statistics`, `LinearAlgebra`, `Printf`, `Random`.

---

## Usage

```julia
# In a Julia session or notebook
using Revise
includet("src/fmri_analysis.jl")   # hot-reload on edits

# Define experiment parameters
params = ExperimentParams(
    tr        = 0.8f0,
    onsets    = [collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations = [fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast  = [1.0f0, -1.0f0, 0.0f0],
    n_discard = 12)

# Run on a 4-D NIfTI volume
using NIfTI
Y = niread("path/to/bold.nii.gz")
ref_idx, t_vol = analyze_and_plot(Y, params, "My recon label")

# Run on a multi-scale MAT file, pinning to the same slice
using MAT
vars = matread("path/to/mslr.mat")
X = vars["X"]
analyze_and_plot_mslr(X, params, Int(vars["Nscales"]), vars["patch_sizes"],
    "MSLR recon"; ref_slice_idx=ref_idx, q=0.05)
```

Run the self-contained demo (synthetic data, no files needed):

```julia
includet("src/fmri_analysis.jl")
demo()
```

---

## Notes

- Complex-valued input arrays are automatically converted to magnitude (`abs.()`) before fitting; a warning is printed when this occurs.
- The analysis pipelines fit the GLM on brain voxels only. The brain mask is derived automatically via `extract_brain_mask` and is not written to disk; for registration or surface analysis, use a dedicated tool such as FSL BET.
- FDR thresholding (Benjamini-Hochberg, q < 0.05 by default) is applied automatically within `analyze_and_plot` and `analyze_and_plot_mslr`. The `fdr_correct` and `bonferroni_correct` functions are also available for standalone use on any returned t-map.
- `analyze_and_plot_mslr` applies FDR thresholding independently per scale, so each scale's activation map reflects its own correction. The t-score colour scale and anatomical underlay are nonetheless shared across scales to keep comparisons interpretable.