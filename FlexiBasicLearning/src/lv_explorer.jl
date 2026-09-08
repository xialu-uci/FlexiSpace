using FlexiBasicLearning
using JLD2 
using ComponentArrays
using CairoMakie
# (1) plot loss vs. a, flex1_params = id(4) (fig1) and identify non-optimal local minima

# choose ig = (a_local_min), flex1_params = id(4), plot p_classical (fig2)

# bfgs on flexi params, with a fixed at a_local_min, plot fit.flex1_params labeled with a_local_min (fig2)

# (2) plot loss vs. a with flex1_params = fit.flex1_params (fig1)

# helper function to find loss vs. a and identify local minima
function find_a_local_minima(flex1_params, learning_problem, savedir; step = 0.01)
    a_grid = range(0.0, 2.0, step=step)
    loss_values = Float64[]
    local_min_idxs = Int[]
    n = length(a_grid)
    lower = 0.0
    for (i, a) in enumerate(a_grid)
        p_classical_derepr = ComponentArray(a = a)
        p_classical_repr = FlexiBasicLearning.represent(p_classical_derepr, learning_problem.model)
        params_repr = ComponentArray(
            p_classical = p_classical_repr,
            flex1_params = flex1_params
        )
        loss = FlexiBasicLearning.get_loss(params_repr; learning_problem=learning_problem)
        # exclude endpoints from local minima consideration
        if i > 2 && i < (n+1)
            if loss > loss_values[end] && loss[end] < loss_values[end-1]
                push!(local_min_idxs, i-1)
            end
        end

        push!(loss_values, loss)
    end
    result = (a_grid = a_grid,
        loss_values = loss_values,
        local_min_idxs = local_min_idxs
    )
    return result
end


datafile = "../FlexiSpaceLocal/data/w_true_params_flexi_args/no-noise/flexi1lv2-4dof-32obs/sim_data_mixed_id.jld2"
savedir = "../FlexiSpaceLocal/exp/09042026/lv2_exploration/flexi1lv2-4dof-32obs/mixed_id"
mkpath(savedir)
@load datafile data
my_prob, my_model = FlexiBasicLearning.set_up_prob(data, () -> FlexiBasicLearning.make_ModelMixedLV(; flexi_dofs = 4), "normalized")
flex1_params = my_model.params_derepresented_ig.flex1_params

a_loss_landscape = find_a_local_minima(flex1_params, my_prob, savedir; step = 0.01)

a_global_min_min_idx = argmin(a_loss_landscape.loss_values[a_loss_landscape.local_min_idxs])
a_local_min_idx = a_loss_landscape.local_min_idxs[a_global_min_min_idx + 1]
a_local_min = a_loss_landscape.a_grid[a_local_min_idx] # because I want local minima closest to global minima on the right

a1_derepr_guess = ComponentArray(p_classical = ComponentArray(a = a_local_min), flex1_params = flex1_params)

a1_guess = FlexiBasicLearning.represent_all(a1_derepr_guess, my_model)

a1_result = FlexiBasicLearning.gradient_descent_learn(my_prob, a1_guess;
         maxiters = 10000, save_parameters = true)

a1_loss_landscape = find_a_local_minima(a1_result.fit_params.flex1_params, my_prob, savedir; step = 0.01)


# plotting

# --- Fig 1: loss vs a landscape(s) ---

fig1 = Figure(size = (900, 600))
ax1 = CairoMakie.Axis(fig1[1, 1],
    xlabel = "a",
    ylabel = "loss",
    yscale = log10,
    title = "Loss landscape vs. a"
)

# original landscape (flex1_params = id)
lines!(ax1, a_loss_landscape.a_grid, a_loss_landscape.loss_values,
    color = :steelblue, label = "flex1_params = id")

# mark all detected local minima on the original landscape
scatter!(ax1, a_loss_landscape.a_grid[a_loss_landscape.local_min_idxs],
    a_loss_landscape.loss_values[a_loss_landscape.local_min_idxs],
    color = :orange, markersize = 12, marker = :circle,
    label = "local minima")

# mark global minimum
scatter!(ax1, [a_loss_landscape.a_grid[a_global_min_idx]],
    [a_loss_landscape.loss_values[a_global_min_idx]],
    color = :red, markersize = 16, marker = :star5,
    label = "global minimum")

# mark chosen a_local_min explicitly
vlines!(ax1, [a_local_min], color = :orange, linestyle = :dash,
    label = "chosen a_local_min = $(round(a_local_min, digits=3))")

# refit landscape (flex1_params = fit.flex1_params)
lines!(ax1, a1_loss_landscape.a_grid, a1_loss_landscape.loss_values,
    color = :green, label = "flex1_params = fit")

a1_global_min_idx = argmin(a1_loss_landscape.loss_values)
scatter!(ax1, [a1_loss_landscape.a_grid[a1_global_min_idx]],
    [a1_loss_landscape.loss_values[a1_global_min_idx]],
    color = :purple, markersize = 16, marker = :star5,
    label = "global minimum (fit)")

axislegend(ax1, position = :rb)

save(joinpath(savedir, "loss_vs_a.png"), fig1)

# gd tracking

flex_param_history = [p.flex1_params for p in a1_result.parameter_history]
flex_grad_history = [g.flex1_params for g in a1_result.gradient_history]

gd_tracker = FlexiBasicLearning.gd_tracking( (gradient_history = flex_grad_history, parameter_history = flex_param_history),
     flex1_params)
gd_tracker_fig = FlexiBasicLearning.plot_gd_tracker(gd_tracker, "gradient_descent", savedir)

fig_flexi = FlexiBasicLearning.plot_flexi_history_ode(a1_result.parameter_history, "a1", savedir, datafile, flex1_params)


# make fig

# plot a_loss_landscape.a_grid vs. a_loss_landscape.loss_values, mark global minimum and chosen a_local_min

# overlay a1_loss_landscape.a_grid vs. a1_loss_landscape.loss_values, mark global minimum
