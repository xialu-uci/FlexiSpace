using JLD2
using DataFrames
using CairoMakie
using Statistics

# expdir = "../FlexiSpaceLocal/exp/08032026/"
expdir_base = "../FlexiSpaceLocal/exp/08132026/fixed-gts"
# df = JLD2.load(joinpath(expdir, "grad_time_results.jld2"), "df")


# const FIXED_NUM_POINTS = 32 # fix num_points for grad_time vs. num_dofs
# const FIXED_DOF = 32 # fix num_dofs for grad_time vs num_points
const NUM_FIXED = 32
const METRIC = :mean_grad_time   # TODO: potentially plot :median_grad_time or maximum(:grad_time)
const METRIC_LABEL = "Mean gradient compute time (s)"

function color_for(labels) # coloring
    uniq = sort(unique(labels))
    palette = Makie.wong_colors()
    return Dict(u => palette[mod1(i, length(palette))] for (i, u) in enumerate(uniq))
end

function marker_for(labels) # markers
    uniq = sort(unique(labels))
    markers = [:circle, :utriangle, :rect, :diamond, :star5, :cross, :xcross]
    return Dict(u => markers[mod1(i, length(markers))] for (i, u) in enumerate(uniq))
end


# plot metric vs. num_(), overlaid by shape
function plot_metric_vs_num_by_shape(df, func_form, fixed_num, x_num, y_metric, metric_label; num_fixed, outdir)
    sub = filter(r -> r.func_form == func_form && getproperty(r, fixed_num) == num_fixed, df) # only consider rows with specific func_form and fixed (either num_points or num_dofs)
    if isempty(sub)
        @warn "No data" func_form fixed_num
        return nothing
    end
    shapes_here = sort(unique(sub.shape)) # should get 3 shapes
    colors = color_for(shapes_here)

    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1], xlabel = "$x_num", ylabel = metric_label,
              title = "$func_form: $y_metric vs $x_num ($fixed_num=$num_fixed)", xscale = log10, yscale = log10)
    for s in shapes_here
        ssub = sort(filter(r -> r.shape == s, sub), :dof) # only consider shape == s
        scatterlines!(ax, getproperty(ssub, x_num), getproperty(ssub, y_metric); label = s, color = colors[s]) # x-axis = n_dof or num_points, y-axis = metric
    end
    axislegend(ax, position = :lt)

    mkpath(outdir)
    outfile = joinpath(outdir, "$(y_metric)_vs_$(x_num)_by_shape_$(fixed_num)$(num_fixed).png")
    save(outfile, fig)
    return outfile
end

function plot_metric_vs_num_by_func(df, shape, fixed_num, x_num, y_metric, metric_label; num_fixed, outdir)
    sub = filter(r -> r.shape == shape && getproperty(r, fixed_num) == num_fixed, df)
    if isempty(sub)
        @warn "No data" shape fixed_num
        return nothing
    end
    funcs_here = sort(unique(sub.func_form))
    colors = color_for(funcs_here)

    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1], xlabel = "$x_num", ylabel = metric_label,
              title = "$shape: $y_metric vs $x_num ($fixed_num=$num_fixed)", xscale = log10, yscale = log10)
    for f in funcs_here
        fsub = sort(filter(r -> r.func_form == f, sub), :dof)
        scatterlines!(ax, getproperty(fsub, x_num), getproperty(fsub, y_metric); label = f, color = colors[f])
    end
    axislegend(ax, position = :lt)

    mkpath(outdir)
    outfile = joinpath(outdir, "$(y_metric)_vs_$(x_num)_by_func_$(fixed_num)$(num_fixed).png")
    save(outfile, fig)
    return outfile
end


function plot_3d_dof_obs_metric(df, metric, metric_label; outdir)
    shapes_here = sort(unique(df.shape))
    funcs_here = sort(unique(df.func_form))
    colors = color_for(shapes_here)
    markers = marker_for(funcs_here)

    fig = Figure(size = (850, 700))
    ax = Axis3(fig[1, 1], xlabel = "Number of DOFs", ylabel = "Number of obs points",
               zlabel = metric_label, title = "$metric_label vs dof & num_points")

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
        scatter!(ax, sub.dof, sub.num_points, getproperty(sub, metric);
                 color = colors[s], marker = markers[f], markersize = 14)
    end

    Legend(fig[1, 2], legend_elems, legend_labels)

    mkpath(outdir)
    outfile = joinpath(outdir, "$(metric)_3d_dof_obs.png")
    save(outfile, fig)
    return outfile
end

# ------------------------------------------------------------------
# run everything
# ------------------------------------------------------------------


# dfs =

# expdir = ["../FlexiSpaceLocal/exp/08132026/fixed-gts/fd" , "../FlexiSpaceLocal/exp/08132026/fixed-gts/fw", "../FlexiSpaceLocal/exp/08132026/fixed-gts/rv"]
modes = ["fw", "rv", "fd"]

y_metrics = [:mean_grad_time, :median_grad_time, :n_grad_calls]
metric_labels = ["mean grad compute (s)", "median grad compute (s)", "number of grad calls in 10 iters"]
x_nums = [:dof, :num_points]
fixed_nums = [:num_points, :dof]

# for df in dfs
for mode in modes
    expdir = joinpath(expdir_base, mode)
    df = JLD2.load(joinpath(expdir, "$(mode)_grad_time_results.jld2"), "df")
    for (y_metric, metric_label) in zip(y_metrics, metric_labels)

        for func_form in unique(df.func_form)
            outdir = joinpath(expdir, "by-func", func_form)
            for (fixed_num, x_num) in zip(fixed_nums, x_nums)
                f1 = plot_metric_vs_num_by_shape(df, func_form, fixed_num, x_num, y_metric, metric_label; num_fixed = NUM_FIXED, outdir)
            # f2 = plot_time_vs_obs_by_shape(df, func_form; dof_fixed = FIXED_DOF, outdir = outdir)
                f1 !== nothing && println("Saved $f1")
            end
            # f2 !== nothing && println("Saved $f2")
        end
        f3 = plot_3d_dof_obs_metric(df, y_metric, metric_label; outdir = joinpath(expdir, "combined"))
        f3 !== nothing && println("Saved $f3")

        for shape in unique(df.shape)
            outdir = joinpath(expdir, "by-shape", shape)
            for (fixed_num, x_num) in zip(fixed_nums, x_nums)
                f4 = plot_metric_vs_num_by_func(df, shape, fixed_num, x_num, y_metric, metric_label; num_fixed = NUM_FIXED, outdir)
                f4 !== nothing && println("Saved $f4")
            end
        end

    end
end

# for func_form in unique(df.func_form)
#     outdir = joinpath(expdir, "by-func", func_form)
#     f1b = plot_ngradcalls_vs_dof_by_shape(df, func_form; num_points_fixed = FIXED_NUM_POINTS, outdir = outdir)
#     f2b = plot_ngradcalls_vs_obs_by_shape(df, func_form; dof_fixed = FIXED_DOF, outdir = outdir)
#     f1b !== nothing && println("Saved $f1b")
#     f2b !== nothing && println("Saved $f2b")
# end

# f3b = plot_3d_dof_obs_ngradcalls(df; outdir = joinpath(expdir, "combined"))
# f3b !== nothing && println("Saved $f3b")

# for shape in unique(df.shape)
#     outdir = joinpath(expdir, "by-shape", shape)
#     f4b = plot_ngradcalls_vs_dof_by_func(df, shape; num_points_fixed = FIXED_NUM_POINTS, outdir = outdir)
#     f4b !== nothing && println("Saved $f4b")
# end
# end
println("Done.")