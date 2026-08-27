using JLD2
using DataFrames
using CairoMakie
using Statistics

expdir_base = "../FlexiSpaceLocal/exp/08132026/fixed-gts"

const NUM_FIXED = 32
const METRIC = :mean_grad_time
const METRIC_LABEL = "Mean gradient compute time (s)"

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
# comparison plots: one panel per mode, linked axes, shared legend
# ------------------------------------------------------------------

# colored by shape, one panel per mode
function plot_metric_vs_num_by_shape_modes(dfs_by_mode, modes, func_form, fixed_num, x_num, y_metric, metric_label; num_fixed, outdir)
    subs = Dict(mode => filter(r -> r.func_form == func_form && getproperty(r, fixed_num) == num_fixed, dfs_by_mode[mode]) for mode in modes)

    all_shapes = sort(unique(vcat([unique(s.shape) for s in values(subs) if !isempty(s)]...)))
    if isempty(all_shapes)
        @warn "No data across any mode" func_form fixed_num
        return nothing
    end
    colors = color_for(all_shapes)

    fig = Figure(size = (380 * length(modes) + 150, 500))
    axes = Axis[]

    for (i, mode) in enumerate(modes)
        sub = subs[mode]
        ax = Axis(fig[2, i], xlabel = "$x_num", ylabel = i == 1 ? metric_label : "",
                  title = mode, xscale = log10, yscale = log10)
        push!(axes, ax)
        isempty(sub) && continue
        for s in sort(unique(sub.shape))
            ssub = sort(filter(r -> r.shape == s, sub), :dof)
            scatterlines!(ax, getproperty(ssub, x_num), getproperty(ssub, y_metric); color = colors[s])
        end
    end

    linkxaxes!(axes...)
    linkyaxes!(axes...)

    legend_elems = [LineElement(color = colors[s]) for s in all_shapes]
    Legend(fig[2, length(modes)+1], legend_elems, string.(all_shapes), "shape")

    Label(fig[1, 1:length(modes)], "$func_form: $y_metric vs $x_num ($fixed_num=$num_fixed)", fontsize = 16)

    mkpath(outdir)
    outfile = joinpath(outdir, "$(y_metric)_vs_$(x_num)_by_shape_$(fixed_num)$(num_fixed)_modes.png")
    save(outfile, fig)
    return outfile
end

# colored by func_form, one panel per mode
function plot_metric_vs_num_by_func_modes(dfs_by_mode, modes, shape, fixed_num, x_num, y_metric, metric_label; num_fixed, outdir)
    subs = Dict(mode => filter(r -> r.shape == shape && getproperty(r, fixed_num) == num_fixed, dfs_by_mode[mode]) for mode in modes)

    all_funcs = sort(unique(vcat([unique(s.func_form) for s in values(subs) if !isempty(s)]...)))
    if isempty(all_funcs)
        @warn "No data across any mode" shape fixed_num
        return nothing
    end
    colors = color_for(all_funcs)

    fig = Figure(size = (380 * length(modes) + 150, 500))
    axes = Axis[]

    for (i, mode) in enumerate(modes)
        sub = subs[mode]
        ax = Axis(fig[2, i], xlabel = "$x_num", ylabel = i == 1 ? metric_label : "",
                  title = mode, xscale = log10, yscale = log10)
        push!(axes, ax)
        isempty(sub) && continue
        for f in sort(unique(sub.func_form))
            fsub = sort(filter(r -> r.func_form == f, sub), :dof)
            scatterlines!(ax, getproperty(fsub, x_num), getproperty(fsub, y_metric); color = colors[f])
        end
    end

    linkxaxes!(axes...)
    linkyaxes!(axes...)

    legend_elems = [LineElement(color = colors[f]) for f in all_funcs]
    Legend(fig[2, length(modes)+1], legend_elems, string.(all_funcs), "func_form")

    Label(fig[1, 1:length(modes)], "$shape: $y_metric vs $x_num ($fixed_num=$num_fixed)", fontsize = 16)

    mkpath(outdir)
    outfile = joinpath(outdir, "$(y_metric)_vs_$(x_num)_by_func_$(fixed_num)$(num_fixed)_modes.png")
    save(outfile, fig)
    return outfile
end

# ------------------------------------------------------------------
# run everything
# ------------------------------------------------------------------

modes = ["fw", "rv", "fd"]

y_metrics = [:mean_grad_time, :median_grad_time, :n_grad_calls, :mean_grad_alloc, :median_grad_alloc]
metric_labels = ["mean grad compute (s)", "median grad compute (s)", "number of grad calls in 10 iters"]
x_nums = [:dof, :num_points]
fixed_nums = [:num_points, :dof]

# load all mode dataframes up front
dfs_by_mode = Dict(mode => JLD2.load(joinpath(expdir_base, mode, "$(mode)_grad_time_results.jld2"), "df") for mode in modes)

all_func_forms = sort(unique(vcat([unique(df.func_form) for df in values(dfs_by_mode)]...)))
all_shapes = sort(unique(vcat([unique(df.shape) for df in values(dfs_by_mode)]...)))

for (y_metric, metric_label) in zip(y_metrics, metric_labels)

    for func_form in all_func_forms
        outdir = joinpath(expdir_base, "compare-modes", "by-func", func_form)
        for (fixed_num, x_num) in zip(fixed_nums, x_nums)
            f1 = plot_metric_vs_num_by_shape_modes(dfs_by_mode, modes, func_form, fixed_num, x_num, y_metric, metric_label; num_fixed = NUM_FIXED, outdir)
            f1 !== nothing && println("Saved $f1")
        end
    end

    for shape in all_shapes
        outdir = joinpath(expdir_base, "compare-modes", "by-shape", shape)
        for (fixed_num, x_num) in zip(fixed_nums, x_nums)
            f2 = plot_metric_vs_num_by_func_modes(dfs_by_mode, modes, shape, fixed_num, x_num, y_metric, metric_label; num_fixed = NUM_FIXED, outdir)
            f2 !== nothing && println("Saved $f2")
        end
    end

end