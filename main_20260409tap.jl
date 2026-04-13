using MAT, NIfTI, Revise
includet("fmri_analysis.jl")

# ==============================================================================
# Experiment & GLM parameters
# ==============================================================================

# Tapping paradigm: alternating tap/rest blocks, 20 s each, 40 s period
params = ExperimentParams(
    tr=0.8f0,
    onsets=[collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
    durations=[fill(20.0f0, 9), fill(20.0f0, 9)],
    contrast=[1.0f0, -1.0f0, 0.0f0],
    n_discard=12)

# ==============================================================================
# Script Execution
# ==============================================================================

# %% Load file: CAIPI + CG-SENSE
fn = "/mnt/storage/rexfung/20260409tap/recon/caipi_recon_cgs.nii"
Y = niread(fn)
cg_idx = analyze_and_plot(Y, params, "CAIPI sampling + CG-SENSE recon")

# %% Load file: time-shifted CAIPI + CG-SENSE
fn = "/mnt/storage/rexfung/20260409tap/recon/caipi_ts_recon_cgs.nii"
Y = niread(fn)
analyze_and_plot(Y, params, "time-shifted CAIPI sampling + CG-SENSE recon"; ref_slice_idx=cg_idx)

# %% Load file: PD + CG-SENSE
fn = "/mnt/storage/rexfung/20260409tap/recon/pd_recon_cgs.nii"
Y = niread(fn)
analyze_and_plot(Y, params, "PD sampling + CG-SENSE recon"; ref_slice_idx=cg_idx)

# %% Load file: CAIPI 5 scales
fn = "/mnt/storage/rexfung/20260409tap/recon/caipi_recon_3scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

caipi_3_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "CAIPI sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "CAIPI sampling + MSLR recon, $Nscales scales"; ref_slice_idx=caipi_3_idx)

# %% Load file: time-shifted CAIPI 5 scales
fn = "/mnt/storage/rexfung/20260409tap/recon/caipi_ts_recon_3scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

caipi_ts_3_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "time-shifted CAIPI sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "time-shifted CAIPI sampling + MSLR recon, $Nscales scales"; ref_slice_idx=caipi_ts_3_idx)

# %% Load file: PD 5 scales
fn = "/mnt/storage/rexfung/20260409tap/recon/pd_recon_3scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

pd_3_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "PD sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "PD sampling + MSLR recon, $Nscales scales"; ref_slice_idx=pd_3_idx)