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

# %% Load file: SMS-EPI + slice-GRAPPA
fn = "/mnt/storage/rexfung/20260317tap/prod/smsepi.nii.gz"
Y = reverse(niread(fn), dims=1)
analyze_and_plot(Y, params, "SMS-EPI + slice-GRAPPA recon")

# %% Load file: CAIPI + CG-SENSE
fn = "/mnt/storage/rexfung/20260317tap/recon/caipi_recon_cgs.nii"
Y = niread(fn)
analyze_and_plot(Y, params, "CAIPI sampling + CG-SENSE recon")

# %% Load file: PD + CG-SENSE
fn = "/mnt/storage/rexfung/20260317tap/recon/pd_recon_cgs.nii"
Y = niread(fn)
analyze_and_plot(Y, params, "PD sampling + CG-SENSE recon")

# %% Load file: CAIPI 5 scales
fn = "/mnt/storage/rexfung/20260317tap/recon/caipi_recon_5scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

caipi_5_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "CAIPI sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "CAIPI sampling + MSLR recon, $Nscales scales"; ref_slice_idx=caipi_5_idx)

# %% Load file: PD 5 scales
fn = "/mnt/storage/rexfung/20260317tap/recon/pd_recon_5scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

pd_5_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "Poisson-disc random sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "Poisson-disc random sampling + MSLR recon, $Nscales scales"; ref_slice_idx=pd_5_idx)

# %% Load file: CAIPI 4 scales
fn = "/mnt/storage/rexfung/20260317tap/recon/caipi_recon_4scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

caipi_4_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "CAIPI sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "CAIPI sampling + MSLR recon, $Nscales scales"; ref_slice_idx=caipi_4_idx)

# %% Load file: PD 4 scales
fn = "/mnt/storage/rexfung/20260317tap/recon/pd_recon_4scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

pd_4_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "Poisson-disc random sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "Poisson-disc random sampling + MSLR recon, $Nscales scales"; ref_slice_idx=pd_4_idx)

# %% Load file: CAIPI 3 scales
fn = "/mnt/storage/rexfung/20260317tap/recon/caipi_recon_3scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

caipi_3_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "CAIPI sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "CAIPI sampling + MSLR recon, $Nscales scales"; ref_slice_idx=caipi_3_idx)

# %% Load file: PD 3 scales
fn = "/mnt/storage/rexfung/20260317tap/recon/pd_recon_3scales.mat"
vars = matread(fn)
X = vars["X"]
Nscales = Int(vars["Nscales"])
patch_sizes = vars["patch_sizes"]

pd_3_idx = analyze_and_plot(dropdims(sum(X, dims=5), dims=5), params,
    "Poisson-disc random sampling + MSLR recon, $Nscales scales (sum)")
analyze_and_plot_mslr(X, params, Nscales, patch_sizes,
    "Poisson-disc random sampling + MSLR recon, $Nscales scales"; ref_slice_idx=pd_3_idx)