# what plots are useful to make for monitoring gd slow?
# already checking the grad norms in not slow
#    for ndofs = 2 could just plot loss vs. params?
using FlexiBasicLearning
using CairoMakie

function end_to_end_gd_tracking(results_all_ig; func_form = FlexiBasicLearning.make_flexi1_func, func_string = "y = f(x)", n_points = 100, n_intermediate = 10)
    # load datafile from results_all_ig
    datafile = results_all_ig[1]["datafile"] # datafile is the same for all results in results_all_ig
    @load datafile true_params
    for ig in results_all_ig
        optimizers = ig["optimizers"]
        gd_results = [ig["gd_$(optimizer)_result"] for optimizer in ig["optimizers"]]
    
        # gd_result = ig["gd_result"]
        savedir = ig["save_dir"]
        # datafile = result["datafile"]
        for (alg, gd_result) in zip(optimizers, gd_results)
            result_gd_tracker = FlexiBasicLearning.gd_tracking(gd_result, true_params)
            # plot stuff
            plot_gd_tracker(result_gd_tracker, alg, savedir)
            plot_param_history(gd_result, alg, savedir, datafile; func_form = func_form, func_string = func_string, n_points = n_points, n_intermediate = n_intermediate)
        end 
    end

end

function gd_tracking(result, gt)
    norms = LinearAlgebra.norm.(result.gradient_history)
    unit_grads = normalize.(result.gradient_history) 
    dots = dot.(unit_grads, Ref(normalize(gt)))
    dists = dist.(result.parameter_history, Ref(gt))
    gd_tracker = (norms=norms, dots = dots, dists = dists)
    return gd_tracker
end

# sol.u, loss_history, grad_norm_history, grads, params

function dist(u,v)
    return LinearAlgebra.norm(u-v)
end

function plot_gd_tracker(gd_tracker, alg, savedir)
    iters = 1:length(gd_tracker.norms)

    fig = Figure(size = (900, 900))

    ax1 = CairoMakie.Axis(fig[1, 1], ylabel = "‖grad‖", yscale = log10,
               title = "Gradient norm vs. $alg iteration")
    lines!(ax1, iters, gd_tracker.norms)

    ax2 = CairoMakie.Axis(fig[2, 1], ylabel = "grad ⋅ grad_true",
               title = "Normalized Gradient ⋅ Truth vs. $alg iteration")
    lines!(ax2, iters, gd_tracker.dots)
    hlines!(ax2, [0.0], color = :gray, linestyle = :dash)

    ax3 = CairoMakie.Axis(fig[3, 1], xlabel = "iteration", ylabel = "‖u - u_true‖",
               yscale = log10, title = "Distance to true parameters vs. $alg iteration")
    lines!(ax3, iters, gd_tracker.dists)

    save(joinpath(savedir, "gd_tracker_$alg.png"), fig)
    return fig
end

function plot_param_history(result, alg, savedir, datafile; func_form = FlexiBasicLearning.make_flexi1_func, func_string = "y = f(x)", n_points = 100, n_intermediate = 10)
    @load datafile data
    @load datafile func_form
    @load datafile true_params
    @load datafile flexi_args

    labels = FlexiBasicLearning.func_form_labels(func_form) 

    x_data = data[:, 1]
    y_data = data[:, 2:end] # n×k, for second figure

    ig = result.parameter_history[1]
    best = result.parameter_history[end]
    n_best = length(result.parameter_history)
    log_idxs = exp.(range(log(2), log(n_best - 1), length = n_intermediate))
    inter_idxs = round.(Int, log_idxs)
    inter_idxs = unique(inter_idxs)

    intermediates = result.parameter_history[inter_idxs]

    params_list = vcat([ig], intermediates, [best], [true_params])
    xs = range(0.0, maximum(x_data), length = n_points)
    xs_flexi = range(0.0, 1.0, length = n_points)

    param_labels = vcat(["initial guess"],
                   ["iter $(i)" for i in inter_idxs],
                   ["best fit"], ["ground truth"])
    cmap = Makie.cgrad(:blues, n_intermediate, categorical = true)
    inter_colors = [cmap[i] for i in 1:n_intermediate]

    colors = vcat([:green], inter_colors, [:indigo], [:black])
    styles = vcat([:solid], fill(:dash, length(intermediates)), [:solid], [:dot])

    # ---- fig1: flexi-function-only, always scalar, unchanged ----
    fig1 = Figure(size = (800, 600))
    ax1 = CairoMakie.Axis(fig1[1, 1], xlabel = labels.flexi_x_label, ylabel = "f(x)",
              title = "Fiting with $alg - Flexifunction Only History for $func_string")

    for (params, label, color, style) in zip(params_list, param_labels, colors, styles)
        ys = [FlexiFunctions.evaluate_decompress(x, params) for x in xs_flexi]
        lines!(ax1, xs_flexi, ys; label = label, color = color, linestyle = style)
    end

    CairoMakie.vlines!(ax1, flexi_args, label = "flexi arg spacing",
                                linestyle = :solid, color = (:gray, 0.6)) # UNTESTED

    axislegend(ax1, position = :rt)
    save(joinpath(savedir, "flexifunction_history_$alg.png"), fig1)

    # ---- fig2: full model output, may be multi-component ----
    n_outputs = size(FlexiBasicLearning.as_matrix(y_data), 2)
    y_labels = n_outputs == 1 ? ["y"] : ["y$j" for j in 1:n_outputs]

    fig2 = Figure(size = (800, 400 * n_outputs))
    ax2 = [CairoMakie.Axis(fig2[j, 1], xlabel = labels.xlabel, ylabel = y_labels[j],
                            title = j == 1 ? "Fiting with $alg - $func_string with Flexifunction History" : "")
           for j in 1:n_outputs]

    for (params, label, color, style) in zip(params_list, param_labels, colors, styles)
        params_func = func_form(params)
        ys = FlexiBasicLearning.as_matrix([params_func(x) for x in xs])   # n_points × n_outputs
        for j in 1:n_outputs
            lines!(ax2[j], xs, ys[:, j]; label = label, color = color, linestyle = style)
        end
    end

    # plot data points on top, one column per output
    y_data_mat = FlexiBasicLearning.as_matrix(y_data)
    for j in 1:n_outputs
        CairoMakie.scatter!(ax2[j], x_data, y_data_mat[:, j], label = "data", markersize = 10, color = (:red, 0.4))
        axislegend(ax2[j], position = :rt)
    end

    save(joinpath(savedir, "fullfunction_history_$alg.png"), fig2)

    return fig1, fig2
end