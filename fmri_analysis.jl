"""
fmri_analysis.jl

Task-based fMRI GLM analysis and visualisation.

Sections
────────
  1.  Haemodynamic Response Function
  2.  Design Matrix
  3.  GLM Fitting & Contrast t-scores
  4.  FDR / Bonferroni Correction
  5.  Visualisation
  6.  Experiment Parameters
  7.  Analysis Pipelines

Dependencies — add via Pkg:
  ]add CairoMakie Plots FFTW
"""

import Random
using Statistics
using LinearAlgebra
using FFTW
using Printf
using Plots
using CairoMakie


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Haemodynamic Response Function
# ─────────────────────────────────────────────────────────────────────────────

"""
    canonical_hrf(tr; peak=6.0, undershoot=16.0, peak_disp=1.0,
                  undershoot_disp=1.0, ratio=0.167)

Double-gamma canonical HRF (SPM-style), sampled at `tr` seconds.
Returns a normalised vector covering 32 s.
"""
function canonical_hrf(tr::Real;
    peak::Real=6.0,
    undershoot::Real=16.0,
    peak_disp::Real=1.0,
    undershoot_disp::Real=1.0,
    ratio::Real=0.167)

    t = 0:tr:32.0
    gamma_pdf(t, a, b) = t^(a - 1) * exp(-t / b) / (b^a * _gamma(a))

    h = [gamma_pdf(ti, peak / peak_disp, peak_disp) -
         ratio * gamma_pdf(ti, undershoot / undershoot_disp, undershoot_disp)
         for ti in t]

    return h ./ maximum(abs.(h))
end

# Simple recursive gamma function for the HRF (avoids SpecialFunctions.jl)
function _gamma(n::Real)
    n < 1 && return _gamma(n + 1) / n
    n < 2 && return 1.0
    return (n - 1) * _gamma(n - 1)
end


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Design Matrix
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_design_matrix(onsets, durations, n_scans, tr; hrf=nothing)

Construct an (n_scans × n_conditions + 1) design matrix X.

- `onsets`    : vector of vectors, onset times in **seconds** per condition
- `durations` : vector of vectors, duration in seconds per condition
- `n_scans`   : total number of TR volumes
- `tr`        : repetition time in seconds

The last column is a column of ones (intercept / baseline).
"""
function build_design_matrix(
    onsets::Vector{<:AbstractVector{<:Real}},
    durations::Vector{<:AbstractVector{<:Real}},
    n_scans::Int,
    tr::Real;
    hrf::Union{AbstractVector{<:Real},Nothing}=nothing)

    isnothing(hrf) && (hrf = canonical_hrf(tr))

    n_cond = length(onsets)
    X = zeros(n_scans, n_cond + 1)

    for c in 1:n_cond
        oversampling = 16
        dt = tr / oversampling
        n_fine = n_scans * oversampling
        stimulus = zeros(n_fine)

        for (onset, dur) in zip(onsets[c], durations[c])
            i_start = max(1, round(Int, onset / dt) + 1)
            i_end = min(n_fine, round(Int, (onset + dur) / dt))
            stimulus[i_start:i_end] .= 1.0
        end

        hrf_fine = repeat(hrf, inner=oversampling)[1:min(end, n_fine)]
        convolved = _conv(stimulus, hrf_fine)[1:n_fine]
        X[:, c] = convolved[oversampling:oversampling:end][1:n_scans]
    end

    X[:, end] .= 1.0   # intercept
    return X
end

"""FFT-based 1-D convolution — O(n log n) replacement for the naive O(n²) version."""
function _conv(u::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    n = length(u) + length(v) - 1
    N = nextpow(2, n)          # pad to next power of 2 for efficient FFT
    U = fft([Float64.(u); zeros(N - length(u))])
    V = fft([Float64.(v); zeros(N - length(v))])
    return real(ifft(U .* V))[1:n]
end


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GLM Fitting & Contrast t-scores
# ─────────────────────────────────────────────────────────────────────────────

"""
    fit_glm(X, Y)

OLS fit of Y = Xβ + ε for data matrix Y (n_scans × n_voxels).

Returns `beta`, `residuals`, and `(X'X)⁻¹`.
"""
function fit_glm(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    XtXinv = inv(X' * X)
    beta = XtXinv * (X' * Y)
    residuals = Y - X * beta
    return beta, residuals, XtXinv
end


"""
    compute_tscores(beta, residuals, XtXinv, contrast)

Voxel-wise t-scores for a contrast vector c:

    t = c'β / sqrt( σ² · c'(X'X)⁻¹c )

where σ² = RSS / (n − p). NaN values in t are replaced with 0.
"""
function compute_tscores(
    beta::AbstractMatrix{<:Real},
    residuals::AbstractMatrix{<:Real},
    XtXinv::AbstractMatrix{<:Real},
    contrast::AbstractVector{<:Real})

    n, _ = size(residuals)
    p = size(beta, 1)
    df = n - p

    sigma2 = max.(vec(sum(residuals .^ 2; dims=1)) ./ df, eps())
    c_var = dot(contrast, XtXinv * contrast)
    c_beta = contrast' * beta

    t = vec(c_beta) ./ sqrt.(sigma2 .* c_var)
    t[isnan.(t)] .= 0.0
    return t
end


"""
    run_glm(Y, onsets, durations, contrast, n_scans, tr)

Full pipeline: design matrix → GLM fit → t-scores.

# Arguments
- `Y`         : (n_scans × n_voxels) BOLD data matrix
- `onsets`    : onset times per condition in seconds
- `durations` : durations per condition in seconds
- `contrast`  : contrast vector (length = n_conditions + 1)
- `n_scans`   : number of volumes
- `tr`        : repetition time in seconds

# Returns `t_map`, `beta`, `X`
"""
function run_glm(
    Y::AbstractMatrix{<:Real},
    onsets::Vector{<:AbstractVector{<:Real}},
    durations::Vector{<:AbstractVector{<:Real}},
    contrast::AbstractVector{<:Real},
    n_scans::Int,
    tr::Real)

    X = build_design_matrix(onsets, durations, n_scans, tr)

    @assert size(Y, 1) == n_scans "Y rows must equal n_scans"
    @assert length(contrast) == size(X, 2) "Contrast length must equal n_regressors"

    beta, residuals, XtXinv = fit_glm(X, Y)
    t_map = compute_tscores(beta, residuals, XtXinv, contrast)

    return t_map, beta, X
end



# ─────────────────────────────────────────────────────────────────────────────
# 4.  FDR Correction
# ─────────────────────────────────────────────────────────────────────────────

"""
    t_to_p(t, df; two_tailed=true)

Convert a vector of t-scores to p-values using the t-distribution CDF.
Uses a rational approximation of the regularised incomplete beta function
to avoid needing Distributions.jl.

- `df`         : residual degrees of freedom (n_scans - n_regressors)
- `two_tailed` : if true (default), returns two-tailed p-values
"""
function t_to_p(t::AbstractVector{<:Real}, df::Int; two_tailed::Bool=true)
    # P(T > |t|) via regularised incomplete beta function
    # I_x(a, b) with x = df/(df+t²), a = df/2, b = 0.5
    function p_one_tail(tv)
        x = df / (df + tv^2)
        return 0.5 * _regularised_ibeta(x, df / 2.0, 0.5)
    end
    p = p_one_tail.(abs.(t))
    two_tailed && (p .*= 2)
    return clamp.(p, 0.0, 1.0)
end

# Lanczos approximation for log-gamma (accurate to ~1e-15)
function _lgamma(z::Real)
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    z < 0.5 && return log(π / sin(π * z)) - _lgamma(1 - z)
    z -= 1
    x = c[1]
    for i in 1:g+1
        x += c[i+1] / (z + i)
    end
    t = z + g + 0.5
    return 0.5 * log(2π) + (z + 0.5) * log(t) - t + log(x)
end


# Accurate to ~1e-10 for the parameter ranges typical in fMRI (df > 10).
function _regularised_ibeta(x::Real, a::Real, b::Real)
    (x <= 0) && return 0.0
    (x >= 1) && return 1.0
    # Use symmetry relation when x > (a+1)/(a+b+2) for faster convergence
    if x > (a + 1) / (a + b + 2)
        return 1.0 - _regularised_ibeta(1 - x, b, a)
    end
    # Lentz continued fraction
    lbeta_ab = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = exp(log(x) * a + log(1 - x) * b - lbeta_ab) / a
    return front * _ibeta_cf(x, a, b)
end

function _ibeta_cf(x::Real, a::Real, b::Real)
    max_iter, tol = 200, 1e-10
    f = 1.0
    C = f
    D = 0.0
    for m in 0:max_iter
        for n in (2m, 2m + 1)
            d = if n == 0
                1.0
            elseif n % 2 == 0
                m′ = n ÷ 2
                m′ * (b - m′) * x / ((a + n - 1) * (a + n))
            else
                m′ = (n - 1) ÷ 2
                -((a + m′) * (a + b + m′) * x) / ((a + n - 1) * (a + n))
            end
            D = 1.0 + d * D
            abs(D) < 1e-30 && (D = 1e-30)
            D = 1.0 / D
            C = 1.0 + d / C
            abs(C) < 1e-30 && (C = 1e-30)
            f *= C * D
            abs(C * D - 1.0) < tol && return f
        end
    end
    return f
end

"""
    fdr_correct(t_map, df; q=0.05, two_tailed=true)

Benjamini-Hochberg FDR correction for a voxel-wise t-map.

# Arguments
- `t_map`      : vector of t-scores (one per voxel)
- `df`         : residual degrees of freedom (n_scans - n_regressors)
- `q`          : desired FDR level (default 0.05)
- `two_tailed` : whether to use two-tailed p-values (default true)

# Returns
- `t_map_fdr`  : t-map with sub-threshold voxels zeroed out
- `mask`       : BitVector — true for voxels surviving FDR correction
- `p_vals`     : raw p-value for each voxel
- `t_threshold`: the t-score cutoff corresponding to the FDR threshold
                 (NaN if no voxels survive)

# Example
    t_map_fdr, mask, p_vals, t_thr = fdr_correct(t_map, n_scans - size(X, 2))
    println("Voxels surviving FDR q<0.05: ", sum(mask))
"""
function fdr_correct(t_map::AbstractVector{<:Real}, df::Int;
    q::Real=0.05, two_tailed::Bool=true)

    n = length(t_map)
    p = t_to_p(t_map, df; two_tailed)

    # Benjamini-Hochberg: sort p-values, find largest k where p_(k) ≤ k/n · q
    order = sortperm(p)
    sorted_p = p[order]
    bh_line = (1:n) .* (q / n)
    surviving = sorted_p .<= bh_line

    mask = falses(n)
    if any(surviving)
        k_max = findlast(surviving)
        p_threshold = sorted_p[k_max]
        mask = p .<= p_threshold
        t_threshold = Float64(minimum(abs.(vec(t_map)[mask])))
    else
        t_threshold = NaN
    end

    t_map_fdr = t_map .* mask

    return t_map_fdr, mask, p, t_threshold
end


"""
    bonferroni_correct(t_map, df; alpha=0.05, two_tailed=true)

Bonferroni correction for a voxel-wise t-map. Divides the desired alpha level
by the number of voxels to control the family-wise error rate (FWER) — i.e.
the probability of *any* false positive across the whole brain.

# Arguments
- `t_map`      : vector of t-scores (one per voxel)
- `df`         : residual degrees of freedom (n_scans - n_regressors)
- `alpha`      : desired FWER level (default 0.05)
- `two_tailed` : whether to use two-tailed p-values (default true)

# Returns
- `t_map_bonf` : t-map with sub-threshold voxels zeroed out
- `mask`       : BitVector — true for voxels surviving correction
- `p_vals`     : raw p-value for each voxel
- `t_threshold`: equivalent t-score cutoff

# Example
    t_map_bonf, mask, p_vals, t_thr = bonferroni_correct(t_map, df; alpha=0.05)
    println("Voxels surviving Bonferroni: ", sum(mask))
"""
function bonferroni_correct(t_map::AbstractVector{<:Real}, df::Int;
    alpha::Real=0.05, two_tailed::Bool=true)

    n = length(t_map)
    p = t_to_p(t_map, df; two_tailed)
    p_threshold = alpha / n        # Bonferroni-adjusted threshold

    mask = p .<= p_threshold
    t_threshold = any(mask) ? Float64(minimum(abs.(vec(t_map)[mask]))) : NaN
    t_map_bonf = t_map .* mask

    return t_map_bonf, mask, p, t_threshold
end


"""
    plot_tmap_flat(t_map; threshold=2.0, title="t-score map")

Two-panel Plots.jl figure for a 1-D t-map vector:
  - Left  : bar chart coloured by sign / threshold
  - Right : histogram with threshold lines
"""
function plot_tmap_flat(t_map::AbstractVector{<:Real};
    threshold=nothing,
    title::String="t-score map")

    threshold = isnothing(threshold) ? 1.96 : Float64(threshold)

    n = length(t_map)
    colors = [t >= threshold ? :crimson :
              t <= -threshold ? :dodgerblue : :lightgray
              for t in t_map]

    p1 = Plots.bar(1:n, t_map;
        color=colors,
        legend=false,
        xlabel="Voxel index",
        ylabel="t-score",
        title="per-voxel t-scores",
        linecolor=:match,
        ylims=(minimum(t_map) * 1.1, maximum(t_map) * 1.1))

    Plots.hline!(p1, [threshold, -threshold];
        linestyle=:dash, color=:black, linewidth=1.5, label="")

    p2 = Plots.histogram(t_map;
        bins=40,
        color=:steelblue,
        alpha=0.7,
        legend=false,
        xlabel="t-score",
        ylabel="Voxel count",
        title="t-score distribution")

    Plots.vline!(p2, [threshold, -threshold];
        linestyle=:dash, color=:black, linewidth=1.5, label="")

    return Plots.plot(p1, p2;
        layout=(1, 2),
        plot_title=title,
        size=(1000, 400),
        margin=5Plots.mm)
end


"""
    plot_tmap_slices(t_vol; threshold=2.0, clim=(-6,6),
                     underlay=nothing, title="t-map slices",
                     slice_indices=nothing)

Orthogonal (axial / coronal / sagittal) slice view of a 3-D t-map using
CairoMakie. Sub-threshold voxels are transparent.

- `underlay`      : optional same-size anatomical volume shown in greyscale
- `slice_indices` : NamedTuple `(x=i, y=j, z=k)`; defaults to volume centre

Returns a `CairoMakie.Figure` — call `display(fig)` or `save("out.png", fig)`.
"""
function plot_tmap_slices(t_vol::AbstractArray{<:Real,3};
    threshold=nothing,
    clim=nothing,
    underlay=nothing,
    title::String="t-map slices",
    slice_indices=nothing)

    t_vals = filter(x -> !isnan(x) && !iszero(x), vec(t_vol))

    threshold = isnothing(threshold) ? [-1.96f0, 1.96f0] : Float32.(threshold)
    clim = if isnothing(clim)
        isempty(t_vals) ? (-1.0f0, 1.0f0) :
        (minimum(t_vals), maximum(t_vals))
    else
        clim
    end

    sx, sy, sz = size(t_vol)
    if isnothing(slice_indices)
        abs_vol = abs.(t_vol)
        peak_idx = all(isnan, abs_vol) ? CartesianIndex(sx ÷ 2, sy ÷ 2, sz ÷ 2) :
                   argmax(replace(abs_vol, NaN => -Inf))
        si = (x=peak_idx[1], y=peak_idx[2], z=peak_idx[3])
    else
        si = slice_indices
    end

    masked = Float32.(t_vol)
    masked[masked.>threshold[1].&&masked.<threshold[2]] .= NaN32

    function get_slices(dim, idx)
        sl_t = Matrix(selectdim(masked, dim, idx))
        sl_u = isnothing(underlay) ? nothing :
               Matrix(selectdim(underlay, dim, idx))
        return sl_t, sl_u
    end

    slices = [
        ("Axial (z=$(si.z))", get_slices(3, si.z)...),
        ("Coronal (y=$(si.y))", get_slices(2, si.y)...),
        ("Sagittal (x=$(si.x))", get_slices(1, si.x)...),
    ]

    fig = CairoMakie.Figure(size=(2200, 840), backgroundcolor=:black)
    CairoMakie.Label(fig[0, 1:3], "$title, t ∉ [$(round(threshold[1], digits=2)), $(round(threshold[2], digits=2))]";
        fontsize=18, color=:white, font=:bold)

    for (col, (slab, sl_t, sl_u)) in enumerate(slices)
        ax = CairoMakie.Axis(fig[1, col];
            title=slab,
            titlecolor=:white,
            backgroundcolor=:black,
            aspect=CairoMakie.DataAspect(),
            yreversed=false,
            xticksvisible=false,
            yticksvisible=false,
            xticklabelsvisible=false,
            yticklabelsvisible=false)

        if !isnothing(sl_u)
            u_norm = (sl_u .- minimum(sl_u)) ./
                     (maximum(sl_u) - minimum(sl_u) .+ eps())
            CairoMakie.heatmap!(ax, u_norm; colormap=:grays, colorrange=(0, 1))
        end

        sym_range = maximum(abs.(collect(clim)))
        hm = CairoMakie.heatmap!(ax, sl_t;
            colormap=CairoMakie.Reverse(:RdYlBu),
            colorrange=(-sym_range, sym_range),
            nan_color=(:black, 0.0))

        col == 3 && CairoMakie.Colorbar(fig[1, 4], hm;
            label="t-score",
            labelcolor=:white,
            tickcolor=:white,
            ticklabelcolor=:white,
            width=16)
    end

    return fig
end


"""
    plot_design_matrix(X; condition_names=nothing)

Heatmap of the GLM design matrix — useful for sanity-checking your model.
"""
function plot_design_matrix(X::AbstractMatrix{<:Real};
    condition_names::Union{Vector{String},Nothing}=nothing)

    n_regressors = size(X, 2)
    labels = isnothing(condition_names) ?
             ["Cond $i" for i in 1:(n_regressors-1)] :
             condition_names
    push!(labels, "Intercept")

    return Plots.heatmap(X;
        color=:grays,
        xlabel="Regressor",
        ylabel="Scan (TR)",
        title="Design matrix",
        xticks=(1:n_regressors, labels),
        size=(600, 400))
end


"""
    tmap_summary(t_map; thresholds=[1.65, 1.96, 2.58, 3.29, 4.42, 5.0], title=nothing)

Print a table of how many voxels survive common t-thresholds, with
approximate two-tailed p-values and percentage of total voxels.

Default thresholds correspond roughly to:
  p<.10, p<.05, p<.01, p<.001, p<.00001, p<.000001 (uncorrected)
"""
function tmap_summary(t_map::AbstractArray{<:Real};
    thresholds::Vector{Float64}=[1.65, 1.96, 2.33, 2.58, 3.09, 3.29, 4.42, 5.0],
    title::Union{String,Nothing}=nothing)

    total = length(t_map)
    header = isnothing(title) ? "── t-map summary" : "── t-map summary: $title"
    println("\n$header")
    @printf("   Total voxels : %d\n", total)
    @printf("   Mean t       : %+.3f\n", mean(t_map))
    @printf("   Std  t       : %.3f\n", std(t_map))
    @printf("   Min / Max    : %.3f  /  %.3f\n", minimum(t_map), maximum(t_map))
    @printf("   Median |t|   : %.3f\n", median(abs.(t_map)))
    @printf("   99th pct |t| : %.3f\n", quantile(abs.(t_map), 0.99))
    println("   ┌───────────┬────────────┬────────┬────────┬─────────┬────────┐")
    println("   │ threshold │  approx p  │  pos   │  neg   │  total  │   %    │")
    println("   ├───────────┼────────────┼────────┼────────┼─────────┼────────┤")
    approx_p = [0.10, 0.05, 0.02, 0.01, 0.002, 0.001, 0.00001, 0.000001]
    for (thr, p) in zip(thresholds, approx_p)
        pos = count(t_map .> thr)
        neg = count(t_map .< -thr)
        both = pos + neg
        pct = 100.0 * both / total
        @printf("   │   |t|>%-4.2f│  p<%-6.0e  │ %6d │ %6d │ %7d │ %5.1f%% │\n",
            thr, p, pos, neg, both, pct)
    end
    println("   └───────────┴────────────┴────────┴────────┴─────────┴────────┘")
end


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Demo
# ─────────────────────────────────────────────────────────────────────────────

function demo()
    Random.seed!(42)

    tr = 2.0
    n_scans = 200
    n_voxels = 500

    onsets_A = collect(10.0:30.0:190.0)
    onsets_B = collect(20.0:30.0:190.0)
    durations_A = fill(10.0, length(onsets_A))
    durations_B = fill(10.0, length(onsets_B))

    onsets = [onsets_A, onsets_B]
    durations = [durations_A, durations_B]

    X_true = build_design_matrix(onsets, durations, n_scans, tr)
    beta_true = zeros(size(X_true, 2), n_voxels)
    beta_true[1, 1:50] .= 2.0   # voxels   1–50  respond to condition A
    beta_true[2, 51:100] .= 2.0   # voxels  51–100 respond to condition B

    Y = X_true * beta_true + 0.5 * randn(n_scans, n_voxels)
    contrast = [1.0, -1.0, 0.0]   # A > B

    t_map, beta_hat, X = run_glm(Y, onsets, durations, contrast, n_scans, tr)

    println("=== fMRI GLM demo ===")
    println("Design matrix size           : $(size(X))")
    println("Mean |t| all voxels          : $(round(mean(abs.(t_map)),          digits=3))")
    println("Mean |t| voxels   1–50  (A)  : $(round(mean(abs.(t_map[1:50])),    digits=3))")
    println("Mean |t| voxels  51–100 (B)  : $(round(mean(abs.(t_map[51:100])),  digits=3))")
    println("Mean |t| voxels 101–500 (∅)  : $(round(mean(abs.(t_map[101:end])), digits=3))")

    tmap_summary(t_map)

    # Flat t-map: bar chart + histogram (Plots.jl)
    display(plot_tmap_flat(t_map; title="Contrast: A > B"))

    # Design matrix heatmap
    display(plot_design_matrix(X; condition_names=["Condition A", "Condition B"]))

    # Orthogonal slice view (CairoMakie) — reshape flat map into toy volume
    t_vol = reshape([t_map; zeros(500)], 10, 10, 10)
    display(plot_tmap_slices(t_vol; title="demo 3-D t-map"))

    return t_map, beta_hat, X
end

# ─────────────────────────────────────────────────────────────────────────────
# 5b. Shared-scale slice viewer
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_tmap_slices_shared(t_vol; threshold, clim, underlay, underlay_range,
                             slice_indices, title)

Identical to `plot_tmap_slices` but accepts an explicit `underlay_range`
`(u_min, u_max)` so that the anatomical background is normalised on a
**shared scale** across multiple calls.

All keyword arguments have sensible defaults and the function is usable
standalone as well as from `analyze_mslr_scales`.
"""
function plot_tmap_slices_shared(
    t_vol::AbstractArray{<:Real,3};
    threshold=nothing,
    clim=nothing,
    underlay=nothing,
    underlay_range=nothing,
    title::String="t-map slices",
    slice_indices=nothing)

    t_vals = filter(x -> !isnan(x) && !iszero(x), vec(t_vol))

    threshold = isnothing(threshold) ? [-1.96f0, 1.96f0] : Float32.(threshold)
    clim = if isnothing(clim)
        isempty(t_vals) ? (-1.0f0, 1.0f0) : (minimum(t_vals), maximum(t_vals))
    else
        Float32.(clim)
    end

    sx, sy, sz = size(t_vol)
    if isnothing(slice_indices)
        abs_vol = abs.(t_vol)
        peak_idx = all(isnan, abs_vol) ? CartesianIndex(sx ÷ 2, sy ÷ 2, sz ÷ 2) :
                   argmax(replace(abs_vol, NaN => -Inf))
        si = (x=peak_idx[1], y=peak_idx[2], z=peak_idx[3])
    else
        si = slice_indices
    end

    masked = Float32.(t_vol)
    masked[masked.>threshold[1].&&masked.<threshold[2]] .= NaN32

    function get_slices(dim, idx)
        sl_t = Matrix(selectdim(masked, dim, idx))
        sl_u = isnothing(underlay) ? nothing :
               Matrix(selectdim(underlay, dim, idx))
        return sl_t, sl_u
    end

    slices = [
        ("Axial (z=$(si.z))", get_slices(3, si.z)...),
        ("Coronal (y=$(si.y))", get_slices(2, si.y)...),
        ("Sagittal (x=$(si.x))", get_slices(1, si.x)...),
    ]

    fig = CairoMakie.Figure(size=(2200, 840), backgroundcolor=:black)
    CairoMakie.Label(fig[0, 1:3],
        "$title, t ∉ [$(round(threshold[1], digits=2)), $(round(threshold[2], digits=2))]";
        fontsize=18, color=:white, font=:bold)

    sym_range = maximum(abs.(collect(clim)))

    for (col, (slab, sl_t, sl_u)) in enumerate(slices)
        ax = CairoMakie.Axis(fig[1, col];
            title=slab,
            titlecolor=:white,
            backgroundcolor=:black,
            aspect=CairoMakie.DataAspect(),
            yreversed=false,
            xticksvisible=false,
            yticksvisible=false,
            xticklabelsvisible=false,
            yticklabelsvisible=false)

        if !isnothing(sl_u)
            u_min, u_max = if !isnothing(underlay_range)
                Float32(underlay_range[1]), Float32(underlay_range[2])
            else
                minimum(sl_u), maximum(sl_u)
            end
            u_norm = (sl_u .- u_min) ./ (u_max - u_min + eps())
            CairoMakie.heatmap!(ax, u_norm; colormap=:grays, colorrange=(0, 1))
        end

        hm = CairoMakie.heatmap!(ax, sl_t;
            colormap=CairoMakie.Reverse(:RdYlBu),
            colorrange=(-sym_range, sym_range),
            nan_color=(:black, 0.0))

        col == 3 && CairoMakie.Colorbar(fig[1, 4], hm;
            label="t-score",
            labelcolor=:white,
            tickcolor=:white,
            ticklabelcolor=:white,
            width=16)
    end

    return fig
end


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Experiment Parameters
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExperimentParams(; tr, onsets, durations, contrast, n_discard=12)

Experiment and GLM parameters, passed to `analyze_and_plot` and
`analyze_and_plot_mslr` to avoid hard-coding them in the analysis functions.

# Fields
- `tr`         : repetition time in seconds (`Float32`)
- `onsets`     : vector of onset-time vectors, one per condition (seconds)
- `durations`  : vector of duration vectors, one per condition (seconds)
- `contrast`   : contrast vector (length = n_conditions + 1, for the intercept)
- `n_discard`  : number of leading frames to discard (default: `12`)

# Example
    params = ExperimentParams(
        tr        = 0.8f0,
        onsets    = [collect(0.0f0:40.0f0:320.0f0), collect(20.0f0:40.0f0:320.0f0)],
        durations = [fill(20.0f0, 9), fill(20.0f0, 9)],
        contrast  = [1.0f0, -1.0f0, 0.0f0])
"""
Base.@kwdef struct ExperimentParams
    tr::Float32
    onsets::Vector{Vector{Float32}}
    durations::Vector{Vector{Float32}}
    contrast::Vector{Float32}
    n_discard::Int = 12
end


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Analysis Pipelines
# ─────────────────────────────────────────────────────────────────────────────

"""
    analyze_and_plot(X, params, title_base; ref_slice_idx=nothing)

Run the full GLM pipeline on a single 4-D volume and display an orthogonal
slice plot.

Complex-valued input is automatically converted to magnitude (`abs.()`) before
fitting; a warning is printed when this happens.

# Arguments
- `X`             : 4-D array (nx, ny, nz, nt_raw); the first `params.n_discard`
                    frames are discarded as instructional frames.
- `params`        : `ExperimentParams` holding TR, onsets, durations, contrast,
                    and number of frames to discard.
- `title_base`    : string used in figure and summary titles.
- `ref_slice_idx` : NamedTuple `(x=i, y=j, z=k)` to pin the display slice.
                    When `nothing` (default) the peak-|t| voxel is used.

# Returns
- `slice_idx` : the NamedTuple slice index actually used (pass to subsequent
                calls via `ref_slice_idx` to keep slices aligned).
"""
function analyze_and_plot(X::AbstractArray{<:Number,4}, params::ExperimentParams,
    title_base::String; ref_slice_idx=nothing)
    # Discard instructional frames
    Y = X[:, :, :, (params.n_discard+1):end]
    (nx, ny, nz, nt) = size(Y)

    # Auto-convert complex input to magnitude
    if eltype(Y) <: Complex
        @warn "analyze_and_plot: complex input detected for \"$title_base\" — " *
              "applying abs.() before GLM fitting."
        Y = abs.(Y)
    end

    Y_mat = Matrix{Float32}(transpose(reshape(Float32.(Y), :, nt)))

    t_map, beta_hat, design_matrix = run_glm(Y_mat, params.onsets, params.durations,
        params.contrast, nt, params.tr)

    # Visualise
    tmap_summary(t_map; title="t-map summary for $title_base")

    fig_flat = plot_tmap_flat(t_map; title="t-scores for $title_base")
    # display(fig_flat)

    t_vol = reshape(t_map, nx, ny, nz)
    underlay = reshape(Y_mat, nt, nx, ny, nz)[1, :, :, :]

    # Use provided slice index or calculate peak
    if isnothing(ref_slice_idx)
        peak_idx = argmax(abs.(t_vol))
        slice_idx = (x=peak_idx[1], y=peak_idx[2], z=peak_idx[3])
    else
        slice_idx = ref_slice_idx
    end

    fig_vol = plot_tmap_slices(t_vol; underlay=underlay, slice_indices=slice_idx,
        threshold=[minimum(t_vol), quantile(abs.(t_map), 0.99)],
        title="t-scores for $title_base")
    display(fig_vol)

    return slice_idx
end


"""
    analyze_and_plot_mslr(X, params, Nscales, patch_sizes, title_base;
                          ref_slice_idx=nothing,
                          threshold_quantile=0.99,
                          plot_summary=false)

Run GLM on each signal component of a multi-scale low-rank (MSLR)
reconstruction and plot all scales on a **shared colour scale**.

Both the t-score overlay and the anatomical underlay are normalised
globally across all scales, so components are directly comparable.

Complex-valued input is automatically converted to magnitude (`abs.()`) before
fitting; a warning is printed when this happens.

# Arguments
- `X`                  : 5-D array (nx, ny, nz, nt, Nscales)
- `params`             : `ExperimentParams` holding TR, onsets, durations, contrast,
                         and number of frames to discard.
- `Nscales`            : number of signal components
- `patch_sizes`        : vector of patch sizes (used for subplot titles)
- `title_base`         : string prefix, e.g. `"CAIPI + MSLR recon, 5 scales"`
- `ref_slice_idx`      : NamedTuple `(x=i, y=j, z=k)` to fix the display
                         slice. When `nothing` (default) the peak |t| in the
                         **summed** reconstruction is used.
- `threshold_quantile` : quantile of pooled |t| used as the lower display
                         threshold. Default `0.99` (top 1 % shown).
- `plot_summary`       : if `true`, call `tmap_summary` for each scale.

# Returns
- `slice_idx` : the NamedTuple slice index used (for reuse across calls)
- `t_maps`    : vector of per-scale t-map vectors (length `Nscales`)

# Usage
    # Auto-detect peak slice from the summed volume:
    caipi_5_idx, _ = analyze_and_plot_mslr(
        X, params, Nscales, patch_sizes, "CAIPI + MSLR recon, \$Nscales scales")

    # Pin slice to one already computed:
    _, _ = analyze_and_plot_mslr(
        X, params, Nscales, patch_sizes, "PD + MSLR recon, \$Nscales scales";
        ref_slice_idx=caipi_5_idx)
"""
function analyze_and_plot_mslr(
    X::AbstractArray{<:Number,5},
    params::ExperimentParams,
    Nscales::Int,
    patch_sizes,
    title_base::String;
    ref_slice_idx=nothing,
    threshold_quantile::Real=0.99,
    plot_summary::Bool=false)

    # Auto-convert complex input to magnitude
    if eltype(X) <: Complex
        @warn "analyze_and_plot_mslr: complex input detected for \"$title_base\" — " *
              "applying abs.() before GLM fitting."
        X = abs.(X)
    end

    (nx, ny, nz, nt_raw, _) = size(X)
    nt = nt_raw - params.n_discard

    # ── Pass 1: compute all t-maps and collect global statistics ───────────
    t_maps = Vector{Vector{Float32}}(undef, Nscales)
    underlays = Vector{Array{Float32,3}}(undef, Nscales)

    for scale in 1:Nscales
        GC.gc()
        Y_scale = X[:, :, :, (params.n_discard+1):end, scale]
        Y_mat = Matrix{Float32}(transpose(reshape(Float32.(Y_scale), :, nt)))

        t_map, _, _ = run_glm(Y_mat, params.onsets, params.durations,
            params.contrast, nt, params.tr)
        t_maps[scale] = t_map
        underlays[scale] = reshape(Y_mat, nt, nx, ny, nz)[1, :, :, :]
    end

    # ── Shared t-score colour scale: symmetric around global max |t| ───────
    global_max_t = maximum(maximum(abs.(tm)) for tm in t_maps)
    shared_clim = (-global_max_t, global_max_t)

    # Shared threshold: top (1 - threshold_quantile) of |t| pooled globally
    all_t_abs = vcat([abs.(tm) for tm in t_maps]...)
    shared_thr_hi = quantile(all_t_abs, threshold_quantile)
    shared_threshold = [minimum(minimum(t_maps)), shared_thr_hi]

    # ── Shared underlay intensity range: global min/max across all scales ──
    u_global_min = minimum(minimum(u) for u in underlays)
    u_global_max = maximum(maximum(u) for u in underlays)
    shared_underlay_range = (u_global_min, u_global_max)

    # ── Determine slice index from summed reconstruction if not provided ───
    if isnothing(ref_slice_idx)
        Y_sum = dropdims(sum(X[:, :, :, (params.n_discard+1):end, :], dims=5), dims=5)
        Y_sum_mat = Matrix(transpose(reshape(Y_sum, :, nt)))
        t_sum, _, _ = run_glm(Y_sum_mat, params.onsets, params.durations,
            params.contrast, nt, params.tr)
        t_sum_vol = reshape(t_sum, nx, ny, nz)
        peak_idx = argmax(abs.(t_sum_vol))
        ref_slice_idx = (x=peak_idx[1], y=peak_idx[2], z=peak_idx[3])
    end

    # ── Pass 2: plot every scale with shared scales ────────────────────────
    for scale in 1:Nscales
        GC.gc()
        scale_title = "$title_base, scale = $(patch_sizes[scale])"
        t_map = t_maps[scale]
        underlay = underlays[scale]

        plot_summary && tmap_summary(t_map; title=scale_title)

        t_vol = reshape(t_map, nx, ny, nz)

        fig = plot_tmap_slices_shared(
            t_vol;
            underlay=underlay,
            slice_indices=ref_slice_idx,
            threshold=shared_threshold,
            clim=shared_clim,
            underlay_range=shared_underlay_range,
            title=scale_title)

        display(fig)
    end

    return ref_slice_idx, t_maps
end