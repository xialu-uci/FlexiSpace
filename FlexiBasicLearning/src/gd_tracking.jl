# what plots are useful to make for monitoring gd slow?
# already checking the grad norms in not slow
#    for ndofs = 2 could just plot loss vs. params?
using FlexiBasicLearning
using CairoMakie

function gd_tracking(result, gt)
    norms = LinearAlgebra.norm.(result.gradient_history)
    dots = dot.(result.gradient_history, Ref(gt))
    dists = dist.(result.parameter_history, Ref(gt))
    gd_tracker = (norms=norms, dots = dots, dists = dists)
    return gd_tracker
end

# sol.u, loss_history, grad_norm_history, grads, params

function dist(u,v)
    return LinearAlgebra.norm(u-v)
end

function plot_gd_tracker(gd_tracker, savedir)
    iters = 1:length(gd_tracker.norms)

    fig = Figure(size = (900, 900))

    ax1 = Axis(fig[1, 1], ylabel = "‖grad‖", yscale = log10,
               title = "Gradient norm vs. iteration")
    lines!(ax1, iters, gd_tracker.norms)

    ax2 = Axis(fig[2, 1], ylabel = "grad ⋅ grad_true",
               title = "Gradient alignment with truth vs. iteration")
    lines!(ax2, iters, gd_tracker.dots)
    hlines!(ax2, [0.0], color = :gray, linestyle = :dash)

    ax3 = Axis(fig[3, 1], xlabel = "iteration", ylabel = "‖u - u_true‖",
               yscale = log10, title = "Distance to true parameters vs. iteration")
    lines!(ax3, iters, gd_tracker.dists)

    save(joinpath(savedir, "gd_tracker.png"), fig)
    return fig
end

function plot_param_history(result, savedir, gt, datafile; func_string = "y = f(x)", n_points = 100, n_intermediate = 10)
    @load datafile data true_func 
    x_data = data[:, 1]
    y_data = data[:, 2] # for second figure


    ig = result.parameter_history[1]
    best = result.parameter_history[end]
    n_best = length(result.parameter_history)
    # choose 10 evenly between
    inter_idxs = rount(Int, range(2, n_best - 1, length = n_intermediate))
    inter_idxs = uniqe(inter_idxs)
    intermediates = result.parameter_history[inter_idxs]

    params_list = vcat[[ig], intermediates, [best]]
    xs = range(0.0, 1.0, length = n_points)

    # label initial guess and best fit, others labeled by index in parameter_history
    # ig is orange solid
    # best fit is green solid
    # rest is gray dashed
    labels = vcat(["initial guess"],
                   ["iter $(i)" for i in inter_idxs],
                   ["best fit"])
    colors = vcat([:orange], fill(:gray, length(intermediates)), [:green])
    styles = vcat([:solid], fill(:dash, length(intermediates)), [:solid])


    fig1 = Figure(size =(800, 600))

    ax1 = CairoMakie.Axis(fig[1, 1], xlabel = "x", ylabel = "f(x)",
              title = "Flexifunction History")


    # save 
    for (params, label, color, style) in zip(params_list, labels, colors, styles)
        ys = [FlexiFunctions.evaluate_decompress(x, params) for x in xs]
        lines!(ax1, xs, ys; label = label, color = color, linestyle = style)
    end

    axislegend(ax, position = :rt)
    save(joinpath(savedir, "flexifunction_history.png"), fig1)

    
    
    fig2 = Figure(size =(800, 600))

    ax2 = CairoMakie.Axis(fig[1, 1], xlabel = "x", ylabel = func_string,
            title = "$func_string with Flexifunction f(x) History")


    # save 
    for (params, label, color, style) in zip(params_list, labels, colors, styles)
        ys = [true_func(x, params) for x in xs]
        lines!(ax, xs, ys; label = label, color = color, linestyle = style)
    end

    # plot data points on top

    CairoMakie.scatter!(ax, x_data, y_data, label = "data", markersize = 4, color = (:gray, 0.4))


    axislegend(ax2, position = :rt)
    save(joinpath(savedir, "fullfunction_history.png"), fig2)

    
    return fig1, fig2


    

end
