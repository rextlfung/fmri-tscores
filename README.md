# fMRI Analysis — Task-Based Activation Mapping (t-scores)

Julia pipeline for task-based fMRI GLM analysis, designed to compare activation maps across multiple MRI reconstruction methods (SMS-EPI/slice-GRAPPA, CG-SENSE, L+S, LLR, and multi-scale low-rank / MSLR).

---

## Repository structure

```
fmri_analysis.jl        # Core library: HRF, design matrix, GLM, correction, plotting
main_<session>.jl       # Per-session analysis scripts (one file per scan date)
```

---

## Core library (`fmri_analysis.jl`)

### 1. Haemodynamic Response Function
- `canonical_hrf(tr)` — SPM-style double-gamma HRF sampled at `tr` seconds. No external dependencies; uses a recursive gamma approximation.

### 2. Design Matrix
- `build_design_matrix(onsets, durations, n_scans, tr)` — Constructs an `(n_scans × n_conditions + 1)` GLM design matrix via FFT-based convolution with 16× temporal oversampling. Last column is the intercept.

### 3. GLM Fitting & Contrasts
- `fit_glm(X, Y)` — OLS fit returning `β`, residuals, and `(X'X)⁻¹`.
- `compute_tscores(β, residuals, XtXinv, contrast)` — Voxel-wise t-scores for a contrast vector.
- `run_glm(Y, onsets, durations, contrast, n_scans, tr)` — Full pipeline wrapper: design matrix → fit → t-map.

### 4. Multiple Comparisons Correction
- `fdr_correct(t_map, df)` — Benjamini-Hochberg FDR correction; returns thresholded t-map, binary mask, raw p-values, and t-threshold.
- `bonferroni_correct(t_map, df)` — Bonferroni FWER correction; same return signature.
- Both use a built-in regularised incomplete beta function (no `Distributions.jl` required).

### 5. Visualisation
- `plot_tmap_flat(t_map)` — Two-panel Plots.jl figure: per-voxel bar chart + t-score histogram.
- `plot_design_matrix(X)` — Heatmap of the design matrix (sanity check).
- `plot_tmap_slices(t_vol)` — Orthogonal axial/coronal/sagittal slice view (CairoMakie) with optional anatomical underlay.
- `plot_tmap_slices_shared(t_vol)` — Same as above but accepts an explicit `underlay_range` for consistent normalisation across multiple plots.
- `tmap_summary(t_map)` — Prints a table of voxel counts surviving common t-thresholds (p < .10 down to p < 10⁻⁶).

### 6. Experiment Parameters
- `ExperimentParams` — Struct holding `tr`, `onsets`, `durations`, `contrast`, and `n_discard` (leading frames to drop before fitting).

### 7. Analysis Pipelines
- `analyze_and_plot(X, params, title)` — Runs the full GLM on a single 4-D volume and displays an orthogonal slice plot. Returns the slice index used (pass as `ref_slice_idx` to align subsequent plots).
- `analyze_and_plot_mslr(X, params, Nscales, patch_sizes, title)` — Runs GLM on each scale of a 5-D MSLR reconstruction and plots all scales on a shared colour scale.

---

## Experiment design (tapping paradigm)

- **Task:** finger-tapping, alternating tap/rest blocks
- **Block duration:** 20 s tap / 20 s rest (40 s period)
- **TR:** 0.8 s
- **Contrast:** tap > rest (`[1, -1, 0]`)
- **Instructional frames:** discarded before fitting (session-dependent; set via `n_discard` in `ExperimentParams`)

---

## Session scripts (`main_<session>.jl`)

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
CairoMakie   # 3-D orthogonal slice visualisation
Plots        # flat t-map and design matrix plots
FFTW         # FFT-based HRF convolution
MAT          # loading .mat reconstruction files
NIfTI        # loading .nii / .nii.gz volumes
Revise       # hot-reloading during interactive development
```

Standard library modules used (no installation needed): `Statistics`, `LinearAlgebra`, `Printf`, `Random`.

---

## Usage

```julia
# In a Julia session or notebook
using Revise
includet("fmri_analysis.jl")   # hot-reload on edits

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
ref_idx = analyze_and_plot(Y, params, "My recon label")

# Run on a multi-scale MAT file, pinning to the same slice
using MAT
vars = matread("path/to/mslr.mat")
X = vars["X"]
analyze_and_plot_mslr(X, params, Int(vars["Nscales"]), vars["patch_sizes"],
    "MSLR recon"; ref_slice_idx=ref_idx)
```

Run the self-contained demo (synthetic data, no files needed):

```julia
includet("fmri_analysis.jl")
demo()
```

---

## Notes

- Complex-valued input arrays are automatically converted to magnitude (`abs.()`) before fitting; a warning is printed when this occurs.
- `analyze_and_plot_mslr` normalises both the t-score colour scale and the anatomical underlay globally across all scales, making per-scale comparisons directly interpretable.
- FDR correction (`fdr_correct`) is available but not called automatically by the pipeline functions; apply it to a returned t-map as a post-processing step if needed.