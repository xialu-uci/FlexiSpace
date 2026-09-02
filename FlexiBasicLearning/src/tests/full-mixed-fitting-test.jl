

using FlexiBasicLearning
using JLD2
using CairoMakie
using ComponentArrays

# -----------------------------------------------------------------------------
# 1. Fitting pipeline: bbo -> (simplex -> gd) x num_flips
# -----------------------------------------------------------------------------

function pipeline_classical_flexi(datafile, savedir, ig; num_flips = 3, optimizer = :gradient_descent)
    @load datafile data
    loss_strategy = "normalized"
    make_model = () -> FlexiBasicLearning.make_ModelMixedLV(;flexi_dofs = 4) # hardcoded for now
    my_prob, my_model = FlexiBasicLearning.set_up_prob(data, make_model, loss_strategy)


    bbo_result = FlexiBasicLearning.bbo_learn(my_prob, ig)
    println(size(ig))

    guess_for_simplex = bbo_result.fit_params_repr


    simplex_results = []
    gd_results = []

    for i = 1:num_flips
        
        simplex_result = FlexiBasicLearning.simplex_learn(my_prob, guess_for_simplex)
        push!(simplex_results, simplex_result)

        
        guess_for_gd = simplex_result.fit_params_repr

        gd_result = FlexiBasicLearning.gradient_descent_learn(my_prob, guess_for_gd;
            optimizer = optimizer, maxiters = 10000, save_parameters = true)
        push!(gd_results, gd_result)

        
        guess_for_simplex = gd_result.fit_params
    end

    full_result = (bbo_result = bbo_result, simplex_results = simplex_results, gd_results = gd_results)
    return full_result
end

# -----------------------------------------------------------------------------
# 2. History concatenation
# -----------------------------------------------------------------------------

"""
Flatten the pipeline's loss histories into two series for plotting: the bbo
warm-start, and the full simplex/gd refinement trace (stages back-to-back,
in the order they actually ran).
"""
function concat_pipeline_loss_histories(full_result)
    bbo_loss = full_result.bbo_result.loss_history

    refinement_loss = Float64[]
    simplex_boundaries = Float64[]
    for (simplex_result, gd_result) in zip(full_result.simplex_results, full_result.gd_results)
        append!(refinement_loss, simplex_result.loss_history)
        append!(simplex_boundaries, length(refinement_loss))
        append!(refinement_loss, gd_result.loss_history)
        append!(simplex_boundaries, length(refinement_loss))
    end

    return bbo_loss, refinement_loss, simplex_boundaries
end

"""
Stitch the per-flip gd `parameter_history` / `gradient_history` into one
continuous trajectory. Each entry is the FULL param struct (p_classical +
flex1_params) since that's what gd jointly optimizes -- simplex is
derivative-free and doesn't produce gradient/parameter histories at all.
`flip_boundaries[i]` is the index in the combined vectors at which flip i's
gd stage ends, in case you want to mark them on a plot later.
"""
function concat_gd_histories(full_result)
    parameter_history = ComponentArray[]
    gradient_history = ComponentArray[]
    flip_boundaries = Int[]

    for gd_result in full_result.gd_results
        append!(parameter_history, gd_result.parameter_history)
        append!(gradient_history, gd_result.gradient_history)
        push!(flip_boundaries, length(parameter_history))
    end

    return (parameter_history = parameter_history,
            gradient_history  = gradient_history,
            flip_boundaries   = flip_boundaries)
end

# -----------------------------------------------------------------------------
# 3. Plotting: reuse fit_mult_alg.jl / gd_tracking.jl where the shapes line up
# -----------------------------------------------------------------------------

"""
ODE-model analogue of `plot_param_history`'s flexi-function-only panel.
(Its *second* panel needs a `func_form(params)(x)` closure, which doesn't
apply to ModelMixedLV -- that model needs a full ODE solve. Use
`FlexiBasicLearning.fw` for the full-model curve instead, as done in
`plot_pipeline_result` below.)
"""
function plot_flexi_history_ode(parameter_history, alg, savedir, datafile, true_flexi_params; n_intermediate = 10)
    # parameter_history entries are the full struct (p_classical + flex1_params);
    # this panel only cares about the flexi-function slice
    parameter_history = [p.flex1_params for p in parameter_history]

    @load datafile flexi_args

    ig = parameter_history[1]
    best = parameter_history[end]
    n_best = length(parameter_history)
    log_idxs = exp.(range(log(2), log(max(n_best - 1, 2)), length = n_intermediate))
    inter_idxs = unique(round.(Int, log_idxs))
    intermediates = parameter_history[inter_idxs]

    params_list = vcat([ig], intermediates, [best], [true_flexi_params])
    param_labels = vcat(["initial guess"], ["iter $(i)" for i in inter_idxs], ["best fit"], ["ground truth"])
    cmap = Makie.cgrad(:blues, max(length(intermediates), 1), categorical = true)
    colors = vcat([:green], [cmap[i] for i in 1:length(intermediates)], [:indigo], [:black])
    styles = vcat([:solid], fill(:dash, length(intermediates)), [:solid], [:dot])

    xs_flexi = range(0.0, 1.0, length = 500)
    fig = Figure(size = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "x", ylabel = "f(x)",
        title = "Pipeline flexi-function history ($alg)")

    for (params, label, color, style) in zip(params_list, param_labels, colors, styles)
        ys = [FlexiFunctions.evaluate_decompress(x, params) for x in xs_flexi]
        lines!(ax, xs_flexi, ys; label = label, color = color, linestyle = style)
    end
    
    CairoMakie.vlines!(ax, flexi_args, label = "flexi arg spacing", color = (:gray, 0.6))
    axislegend(ax, position = :rt)

    save(joinpath(savedir, "pipeline_flexi_history_$alg.png"), fig)
    return fig
end

function plot_pipeline_result(full_result, my_model, datafile, savedir)
    mkpath(savedir)
    @load datafile data
    @load datafile true_params

    # --- loss history: bbo and concatenated simplex+gd ---
    #TODO: TEST(add lines indicating where simplex is)
    bbo_loss, refinement_loss, simplex_boundaries = concat_pipeline_loss_histories(full_result)
    combined = concat_gd_histories(full_result)

    fig_loss = FlexiBasicLearning.make_loss_history_figs(
        [bbo_loss, refinement_loss],
        [NaN, NaN],  # swap in real elapsed times if you start tracking them per-stage
        ["bbo", "simplex + gd"]; simplex_boundaries = [nothing, simplex_boundaries])
    save(joinpath(savedir, "pipeline_loss_history.png"), fig_loss)

   # gd tracking 
    combined = concat_gd_histories(full_result)
    flex_param_history = [p.flex1_params for p in combined.parameter_history]
    flex_grad_history = [g.flex1_params for g in combined.gradient_history]
    gd_tracker = FlexiBasicLearning.gd_tracking(
        (gradient_history = flex_grad_history, parameter_history = flex_param_history),
        true_params.flex1_params)
    fig_tracker = FlexiBasicLearning.plot_gd_tracker(gd_tracker, "pipeline", savedir; flip_boundaries = combined.flip_boundaries)
    #TODO: combined.flip_boundaries has the iteration index of each flip's gd-stagekeys
    # TODO: vlines! them onto fig_tracker's axes afterward.

    # --- flexi-function-only snapshots across the whole pipeline ---
    fig_flexi = plot_flexi_history_ode(combined.parameter_history, "pipeline", savedir, datafile, true_params.flex1_params)

    # --- fit overlay: final pipeline fit vs. data vs. ground truth ---
    # gd_result.fit_params is already the full struct (p_classical + flex1_params),
    final_params_repr = full_result.gd_results[end].fit_params
    final_params_derepr = FlexiBasicLearning.derepresent_all(final_params_repr, my_model)

    x_data = data[:, 1]
    y_data = data[:, 2:end]
    x_grid = collect(LinRange(0.0, maximum(x_data), 500))

    y_final = FlexiBasicLearning.fw(x_grid, final_params_derepr, my_model) # TODO: must be derepr
    y_true  = FlexiBasicLearning.fw(x_grid, true_params, my_model)

    fig_overlay = FlexiBasicLearning.make_fit_overlay_fig(x_data, y_data, x_grid, y_true, [y_final];
        title = "Pipeline fit", xlabel = "t", labels = ["pipeline fit"])
    save(joinpath(savedir, "pipeline_fit_overlay.png"), fig_overlay)

    return (fig_loss = fig_loss, fig_tracker = fig_tracker, fig_flexi = fig_flexi,
            fig_overlay = fig_overlay, final_params_repr = final_params_repr)
end

# -----------------------------------------------------------------------------
# 4. Near-guess wrapper: choose_near_ig perturbs in derepresented space
# 
# -----------------------------------------------------------------------------

function choose_near_repr_ig(derepr_ig, model; kwargs...)
    near_derepr = FlexiBasicLearning.choose_near_ig(derepr_ig, model; kwargs...)
    return ComponentArray(
        p_classical  = FlexiBasicLearning.represent(near_derepr.p_classical, model),
        flex1_params = near_derepr.flex1_params,
    )
end

# -----------------------------------------------------------------------------
# 5. The near-guess loop
# -----------------------------------------------------------------------------
datafile = "../FlexiSpaceLocal/data/w_true_params_flexi_args/no-noise/flexi1lv2-4dof-32obs/sim_data_mixed_id.jld2"
my_model = FlexiBasicLearning.make_ModelMixedLV(;flexi_dofs = 4)
ig = my_model.params_derepresented_ig
println(size(ig))
dist_list = [0.0, 0.05, 0.5]
# dist_list = [0.0]
dir_list = ["cu", "cd"]
savedir_base = "../FlexiSpaceLocal/exp/09012026/mixed-lv"

all_near_results = []

dist_list = [0.0, 0.05, 0.5]
dir_list = ["cu", "cd"]
savedir_base = "../FlexiSpaceLocal/exp/09012026/mixed-lv"

all_near_results = []

for dist in dist_list
    for dir in dir_list
        # Skip combinations we don't want
        if dist == 0.0 && dir != "cu"
            continue
        end
        
        near_ig = choose_near_repr_ig(ig, my_model; cdist = dist, fdist = dist, fdir = dir)
        
        savedir = joinpath(savedir_base, "dist$(dist)_dir$(dir)")
        mkpath(savedir)
        
        full_result = pipeline_classical_flexi(datafile, savedir, near_ig; num_flips = 3, optimizer = :gradient_descent)
        @save joinpath(savedir, "full_result.jld2") full_result
        
        plots = plot_pipeline_result(full_result, my_model, datafile, savedir)
        
        push!(all_near_results, (dist = dist, dir = dir, savedir = savedir,
                                 full_result = full_result, final_params_repr = plots.final_params_repr))
    end
end

# -----------------------------------------------------------------------------
# 6. Overlay of fits from each near guess
# -----------------------------------------------------------------------------
@load datafile data
@load datafile true_params
x_data = data[:, 1]
y_data = data[:, 2:end]
x_grid = collect(LinRange(0.0, maximum(x_data), 500))

y_true = FlexiBasicLearning.fw(x_grid, true_params, my_model)
y_pred_list = [FlexiBasicLearning.fw(x_grid, FlexiBasicLearning.derepresent_all(r.final_params_repr, my_model), my_model) for r in all_near_results]
labels = ["dist=$(r.dist) dir=$(r.dir)" for r in all_near_results]

fig_all_overlay = FlexiBasicLearning.make_fit_overlay_fig(x_data, y_data, x_grid, y_true, y_pred_list;
    title = "Fits from all near guesses", xlabel = "t", labels = labels)
save(joinpath(savedir_base, "all_near_guess_overlay.png"), fig_all_overlay)