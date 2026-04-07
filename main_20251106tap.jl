using MAT, NIfTI, Revise
includet("fmri_analysis.jl")

# ==============================================================================
# Experiment & GLM parameters
# ==============================================================================

# Tapping paradigm: alternating tap/rest blocks, 20 s each, 40 s period
# Note: instructional frames already discarded upstream for this dataset
params = ExperimentParams(
    tr=0.8f0,
    onsets=[collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations=[fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast=[1.0f0, -1.0f0, 0.0f0],
    n_discard=0)

# ==============================================================================
# Script Execution
# ==============================================================================

# %% Load file: SMS-EPI + slice-GRAPPA recon
fn = "/mnt/storage/rexfung/20251106balltap/tap/product/prod.nii"
Y = niread(fn)
global_slice_idx = analyze_and_plot(Y, params, "SMS-EPI + slice-GRAPPA recon")

# %% Load file: Gaussian random sampling and L+S recon
# Extract scale 2 specifically for L+S
fn = "/mnt/storage/rexfung/20251106balltap/tap/recon/recon_2scales_L+S.mat"
vars = matread(fn)
X = vars["X"]
Y = reverse(ndims(X) > 4 ? X[:, :, :, :, 2] : X, dims=1)
analyze_and_plot(Y, params, "Gaussian random sampling and L+S recon";
    ref_slice_idx=global_slice_idx)

# %% Load file: Gaussian random sampling and LLR recon
fn = "/mnt/storage/rexfung/20251106balltap/tap/recon/recon_1scalesLLR_overlapping.mat"
vars = matread(fn)
X = vars["X"]
Y = reverse(ndims(X) > 4 ? dropdims(sum(X, dims=5), dims=5) : X, dims=1)
analyze_and_plot(Y, params, "Gaussian random sampling and LLR recon";
    ref_slice_idx=global_slice_idx)

# %% Load file: Gaussian random sampling and MSLR (5 scales) recon
fn = "/mnt/storage/rexfung/20251106balltap/tap/recon/recon_5scales_overlapping.mat"
vars = matread(fn)
X = vars["X"]
Nscales = ndims(X) > 4 ? size(X, 5) : 1
patch_sizes = string.(1:Nscales)   # no patch_sizes in file; use scale index as label

Y = reverse(ndims(X) > 4 ? dropdims(sum(X, dims=5), dims=5) : X, dims=1)
idx_5scales = analyze_and_plot(Y, params,
    "Gaussian random sampling and MSLR ($Nscales scales) recon (sum)";
    ref_slice_idx=global_slice_idx)

if ndims(X) > 4
    X_flip = reverse(X, dims=1)
    analyze_and_plot_mslr(X_flip, params, Nscales, patch_sizes,
        "Gaussian random sampling and MSLR ($Nscales scales) recon";
        ref_slice_idx=idx_5scales)
end