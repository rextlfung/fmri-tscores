# ==============================================================================
# NIfTI export helper functions
# ==============================================================================

"""
    export_niftis(X, t_vol, params, prefix, out_dir)

4-D method — for a single NIfTI reconstruction.

Writes:
  - `<prefix>_mag.nii`  : post-discard magnitude timeseries (4-D)
  - `<prefix>_tmap.nii` : 3-D voxel-wise t-scores
"""
function export_niftis(X::AbstractArray{<:Number,4}, t_vol::AbstractArray{<:Real,3},
    params::ExperimentParams, prefix::String, out_dir::String)

    Y = X[:, :, :, (params.n_discard+1):end]
    mag = eltype(Y) <: Complex ? Float32.(abs.(Y)) : Float32.(Y)
    niwrite(joinpath(out_dir, "$(prefix)_mag.nii"), NIVolume(mag))

    niwrite(joinpath(out_dir, "$(prefix)_tmap.nii"), NIVolume(Float32.(t_vol)))

    @printf("Exported %s\n", prefix)
end

"""
    export_niftis(X, t_vols, patch_sizes, Nscales, params, prefix, out_dir)

5-D method — for a multi-scale low-rank reconstruction.

Writes per scale:
  - `<prefix>_<N>scales_patchsize<P>_mag.nii`  : post-discard magnitude timeseries (4-D)
  - `<prefix>_<N>scales_patchsize<P>_tmap.nii` : 3-D voxel-wise t-scores
"""
function export_niftis(X::AbstractArray{<:Number,5}, t_vols::Vector{<:AbstractArray{<:Real,3}},
    patch_sizes, Nscales::Int,
    params::ExperimentParams, prefix::String, out_dir::String)
    for scale in 1:Nscales
        ps = Int.(patch_sizes[scale])
        tag = "$(prefix)_$(Nscales)scales_patchsize$(ps)"

        Y_scale = X[:, :, :, (params.n_discard+1):end, scale]
        mag = eltype(Y_scale) <: Complex ? Float32.(abs.(Y_scale)) : Float32.(Y_scale)
        niwrite(joinpath(out_dir, "$(tag)_mag.nii"), NIVolume(mag))

        niwrite(joinpath(out_dir, "$(tag)_tmap.nii"), NIVolume(Float32.(t_vols[scale])))

        @printf("Exported %s\n", tag)
    end
end