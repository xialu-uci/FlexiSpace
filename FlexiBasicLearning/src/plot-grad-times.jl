
# Loads the results table produced by timing_grads.jl and makes:
#   1. grad time vs num_dofs, overlaid by shape, fixed num_points   -> by-func/<func>/
#   2. grad time vs num_points, overlaid by shape, fixed num_dofs   -> by-func/<func>/
#   3. one combined 3D plot: dof (x), num_points (y), grad time (z),
#      colored by shape, marker by func form                        -> combined/
#   4. grad time vs num_dofs, overlaid by func form, fixed num_points -> by-shape/<shape>/

using JLD2
using DataFrames
using CairoMakie
using Statistics

expdir = "../FlexiSpaceLocal/exp/08032026/"
df = JLD2.load(joinpath(expdir, "grad_time_results.jld2"), "df")

# --- config -----------------------------------------------------------
# NOTE: the old sweep fixed num_points = 20 for the dof-sweep plots.
# 20 is not in the new num_points_list ([3, 5, 10, 50, 100]), so pick
# a stand-in fixed value here -- change as needed.
const FIXED_NUM_POINTS = 20
const FIXED_DOF = 5
const METRIC = :mean_grad_time   # or :median_grad_time / :total_grad_time
const METRIC_LABEL = "Mean gradient compute time (s)"

# --- helpers for consistent color/marker assignment --------------------
function color_for(labels)
    uniq = sort(unique(labels))
    palette = Makie.wong_colors()
    return Dict(u => palette[mod1(i, length(palette))] for (i, u) in enumerate(uniq))
end

function marker_for(labels)
    uniq = sort(unique(labels))
    markers = [:circle, :utriangle, :rect, :diamond, :star5, :cross, :xcross]
    return Dict(u => markers[mod1(i, length(markers))] for (i, u) in enumerate(uniq))
end

# ------------------------------------------------------------------
# 1. grad time vs num_dofs, overlaid by shape, fixed num_points
# ------------------------------------------------------------------
function plot_time_vs_dof_by_shape(df, func_form; num_points_fixed, outdir)
    sub = filter(r -> r.func_form == func_form && r.num_points == num_points_fixed, df)
    if isempty(sub)
        @warn "No data" func_form num_points_fixed
        return nothing
    end
    shapes_here = sort(unique(sub.shape))
    colors = color_for(shapes_here)

    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1], xlabel = "Number of DOFs", ylabel = METRIC_LABEL,
              title = "$func_form: grad time vs dof (num_points=$num_points_fixed)")
    for s in shapes_here
        ssub = sort(filter(r -> r.shape == s, sub), :dof)
        scatterlines!(ax, ssub.dof, getproperty(ssub, METRIC); label = s, color = colors[s])
    end
    axislegend(ax, position = :lt)

    mkpath(outdir)
    outfile = joinpath(outdir, "grad_time_vs_dof_by_shape_np$(num_points_fixed).png")
    save(outfile, fig)
    return outfile
end

# ------------------------------------------------------------------
# 2. grad time vs num_points, overlaid by shape, fixed num_dofs
# ------------------------------------------------------------------
function plot_time_vs_obs_by_shape(df, func_form; dof_fixed, outdir)
    sub = filter(r -> r.func_form == func_form && r.dof == dof_fixed, df)
    if isempty(sub)
        @warn "No data" func_form dof_fixed
        return nothing
    end
    shapes_here = sort(unique(sub.shape))
    colors = color_for(shapes_here)

    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1], xlabel = "Number of observed points", ylabel = METRIC_LABEL,
              title = "$func_form: grad time vs num_points (dof=$dof_fixed)")
    for s in shapes_here
        ssub = sort(filter(r -> r.shape == s, sub), :num_points)
        scatterlines!(ax, ssub.num_points, getproperty(ssub, METRIC); label = s, color = colors[s])
    end
    axislegend(ax, position = :lt)

    mkpath(outdir)
    outfile = joinpath(outdir, "grad_time_vs_obs_by_shape_dof$(dof_fixed).png")
    save(outfile, fig)
    return outfile
end

# ------------------------------------------------------------------
# 3. combined 3D plot: dof (x), num_points (y), grad time (z)
#    color = shape, marker = func form
# ------------------------------------------------------------------
function plot_3d_dof_obs_time(df; outdir)
    shapes_here = sort(unique(df.shape))
    funcs_here = sort(unique(df.func_form))
    colors = color_for(shapes_here)
    markers = marker_for(funcs_here)

    fig = Figure(size = (850, 700))
    ax = Axis3(fig[1, 1], xlabel = "Number of DOFs", ylabel = "Number of obs points",
               zlabel = METRIC_LABEL, title = "Grad compute time vs dof & num_points")

    # dummy legend entries: one per shape (color) and one per func (marker),
    # since Axis3 doesn't auto-merge legends from `label=` on scatter! well.
    legend_elems = Any[]
    legend_labels = String[]
    for s in shapes_here
        push!(legend_elems, MarkerElement(color = colors[s], marker = :circle))
        push!(legend_labels, "shape: $s")
    end
    for f in funcs_here
        push!(legend_elems, MarkerElement(color = :gray, marker = markers[f]))
        push!(legend_labels, "func: $f")
    end

    for s in shapes_here, f in funcs_here
        sub = filter(r -> r.shape == s && r.func_form == f, df)
        isempty(sub) && continue
        scatter!(ax, sub.dof, sub.num_points, getproperty(sub, METRIC);
                 color = colors[s], marker = markers[f], markersize = 14)
    end

    Legend(fig[1, 2], legend_elems, legend_labels)

    mkpath(outdir)
    outfile = joinpath(outdir, "grad_time_3d_dof_obs.png")
    save(outfile, fig)
    return outfile
end

# ------------------------------------------------------------------
# 4. grad time vs num_dofs, overlaid by func form, fixed num_points
# ------------------------------------------------------------------
function plot_time_vs_dof_by_func(df, shape; num_points_fixed, outdir)
    sub = filter(r -> r.shape == shape && r.num_points == num_points_fixed, df)
    if isempty(sub)
        @warn "No data" shape num_points_fixed
        return nothing
    end
    funcs_here = sort(unique(sub.func_form))
    colors = color_for(funcs_here)

    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1], xlabel = "Number of DOFs", ylabel = METRIC_LABEL,
              title = "$shape: grad time vs dof by func form (num_points=$num_points_fixed)")
    for f in funcs_here
        fsub = sort(filter(r -> r.func_form == f, sub), :dof)
        scatterlines!(ax, fsub.dof, getproperty(fsub, METRIC); label = f, color = colors[f])
    end
    axislegend(ax, position = :lt)

    mkpath(outdir)
    outfile = joinpath(outdir, "grad_time_vs_dof_by_func_np$(num_points_fixed).png")
    save(outfile, fig)
    return outfile
end

# ------------------------------------------------------------------
# run everything
# ------------------------------------------------------------------
for func_form in unique(df.func_form)
    outdir = joinpath(expdir, "by-func", func_form)
    f1 = plot_time_vs_dof_by_shape(df, func_form; num_points_fixed = FIXED_NUM_POINTS, outdir = outdir)
    f2 = plot_time_vs_obs_by_shape(df, func_form; dof_fixed = FIXED_DOF, outdir = outdir)
    f1 !== nothing && println("Saved $f1")
    f2 !== nothing && println("Saved $f2")
end

f3 = plot_3d_dof_obs_time(df; outdir = joinpath(expdir, "combined"))
f3 !== nothing && println("Saved $f3")

for shape in unique(df.shape)
    outdir = joinpath(expdir, "by-shape", shape)
    f4 = plot_time_vs_dof_by_func(df, shape; num_points_fixed = FIXED_NUM_POINTS, outdir = outdir)
    f4 !== nothing && println("Saved $f4")
end

println("Done.")