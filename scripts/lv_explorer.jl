using FlexiBasicLearning
using JLD2 
using ComponentArrays
using CairoMakie
using Printf

# (1) plot loss vs. a, flex1_params = id(4) (fig1) and identify non-optimal local minima

# choose ig = (a_local_min), flex1_params = id(4), plot p_classical (fig2)

# bfgs on flexi params, with a fixed at a_local_min, plot fit.flex1_params labeled with a_local_min (fig2)

# (2) plot loss vs. a with flex1_params = fit.flex1_params (fig1)



# debugged and parallel
function find_a_local_minima(flex1_params, learning_problem, savedir; step = 0.01)
    a_grid = range(0.0, 2.0, step = step)
    n = length(a_grid)
    loss_values = Vector{Float64}(undef, n)

    Threads.@threads for i in 1:n
        a = a_grid[i]
        p_classical_derepr = ComponentArray(a = a)
        p_classical_repr = FlexiBasicLearning.represent(p_classical_derepr, learning_problem.model)
        params_repr = ComponentArray(
            p_classical = p_classical_repr,
            flex1_params = flex1_params
        )
        loss_values[i] = FlexiBasicLearning.get_loss(params_repr; learning_problem = learning_problem)
    end

    # find local minima as a separate vectorized pass (excluding endpoints)
    local_min_idxs = Int[]
    for i in 2:(n - 1)
        if loss_values[i] < loss_values[i - 1] && loss_values[i] < loss_values[i + 1]
            push!(local_min_idxs, i)
        end
    end

    result = (a_grid = a_grid,
        loss_values = loss_values,
        local_min_idxs = local_min_idxs
    )
    return result
end

datafile = joinpath(FlexiBasicLearning.asset_dir, "data/w_true_params_flexi_args/no-noise/flexi1lv2-4dof-32obs/sim_data_mixed_id.jld2")
savedir = joinpath(FlexiBasicLearning.asset_dir, "exp/09082026/lv2_exploration/flexi1lv2-4dof-32obs/mixed_id")
mkpath(savedir)
@load datafile data
my_prob, my_model = FlexiBasicLearning.set_up_prob(data, () -> FlexiBasicLearning.make_ModelMixedLV(; flexi_dofs = 4), "normalized")
flex1_params = my_model.params_derepresented_ig.flex1_params

a_loss_landscape = find_a_local_minima(flex1_params, my_prob, savedir; step = 0.05)

a_global_min_min_idx = argmin(a_loss_landscape.loss_values[a_loss_landscape.local_min_idxs])
a_local_min_idx = a_loss_landscape.local_min_idxs[a_global_min_min_idx + 1]
a_local_min = a_loss_landscape.a_grid[a_local_min_idx] # because I want local minima closest to global minima on the right


a1_derepr_guess = ComponentArray(p_classical = ComponentArray(a = a_local_min), flex1_params = flex1_params)

# a1_guess = FlexiBasicLearning.represent_all(a1_derepr_guess, my_model)

# TODO: CHECK repeat 5 rounds of simplex (find a_guess_for_gd) and bfgs (a_result) and find loss landscapes for each a_result.fit.flex1_params
# first guess chosen as a1_guess, subsequent a_guesses chosen with simplex 

# a1_result = FlexiBasicLearning.gradient_descent_learn(my_prob, a1_guess; optimizer = :bfgs,
#          maxiters = 10000, save_parameters = true)

# a1_loss_landscape = find_a_local_minima(a1_result.fit_params.flex1_params, my_prob, savedir; step = 0.01)

# --- simplex + bfgs loop, tracking landscapes per round ---

n_rounds = 5

results        = Vector{Any}(undef, n_rounds)   # bfgs (a_result) per round
loss_landscapes = Vector{Any}(undef, n_rounds)  # loss-vs-a landscape per round's fit
a_guesses      = Vector{Float64}(undef, n_rounds)  # scalar a used to seed each round's bfgs


# --- Round 1: seed with a_local_min found from the initial grid search ---
a_guesses[1] = a_local_min 
a1_guess = FlexiBasicLearning.represent_all(a1_derepr_guess, my_model) # guess needs to be repr component array with P_classical and flex1_params for learning
results[1] = FlexiBasicLearning.gradient_descent_learn(my_prob, a1_guess; optimizer = :adam,
                 maxiters = 10000, save_parameters = true)
loss_landscapes[1] = find_a_local_minima(results[1].fit_params.flex1_params, my_prob, savedir; step = 0.01)

# --- Rounds 2-5: simplex finds next a_guess, then bfgs refines it ---
for i in 2:n_rounds
    prev_fit = results[i - 1].fit_params  # derepresented params from previous round's bfgs fit

   
    simplex_result = FlexiBasicLearning.simplex_learn(my_prob, prev_fit)

    guess_derepr = FlexiBasicLearning.derepresent_all(simplex_result.fit_params_repr, my_model)
    a_guesses[i] = guess_derepr.p_classical.a
    # a_guess_repr = FlexiBasicLearning.represent_all(simplex_result.fit_params_repr, my_model)
    # simplex_result.fit_params_repr is guess_repr
    results[i] = FlexiBasicLearning.gradient_descent_learn(my_prob, simplex_result.fit_params_repr; optimizer = :adam,
                     maxiters = 10000, save_parameters = true)

    loss_landscapes[i] = find_a_local_minima(results[i].fit_params.flex1_params, my_prob, savedir; step = 0.01)
end
# simplex_result = FlexiBasicLearning.simplex_learn(my_prob, a1_result.fit_params; maxiters = 10000, save_parameters = true)
# a_guess_for_gd = simplex_result.fit_params_repr.flex1_params

# save results, loss_landscapes, a_guesses to savedir to reload and plot later
@save joinpath(savedir, "results_landscapes_guesses.jld2") results loss_landscapes a_guesses

# --- Plotting: baseline (id) landscape + 5 round landscapes overlaid ---

fig1 = Figure(size = (1000, 700))
ax1 = CairoMakie.Axis(fig1[1, 1],
    xlabel = "a",
    ylabel = "loss",
    yscale = log10,
    title = "Loss landscape vs. a across $(n_rounds) rounds"
)

# baseline: flex1_params = id landscape, for reference
a_global_min_idx = argmin(a_loss_landscape.loss_values)
lines!(ax1, a_loss_landscape.a_grid, a_loss_landscape.loss_values,
    color = :gray, linestyle = :dot, label = "flex1_params = id (baseline)")
scatter!(ax1, [a_loss_landscape.a_grid[a_global_min_idx]],
    [a_loss_landscape.loss_values[a_global_min_idx]],
    color = :gray, markersize = 14, marker = :star5)

# distinct colors per round
round_colors = Makie.wong_colors()[1:n_rounds]

for i in 1:n_rounds
    landscape = loss_landscapes[i]
    color = round_colors[i]

    # main landscape curve
    lines!(ax1, landscape.a_grid, landscape.loss_values,
        color = color, label = "round $i fit")

    # local minima: faded circles
    scatter!(ax1, landscape.a_grid[landscape.local_min_idxs],
        landscape.loss_values[landscape.local_min_idxs],
        color = (color, 0.4), markersize = 8, marker = :circle)

    # global minimum: solid star
    gmin_idx = argmin(landscape.loss_values)
    scatter!(ax1, [landscape.a_grid[gmin_idx]], [landscape.loss_values[gmin_idx]],
        color = color, markersize = 16, marker = :star5)

    # a_guess used to seed this round's bfgs: faded vertical dashed line
    vlines!(ax1, [a_guesses[i]], color = (color, 0.4), linestyle = :dash)
end

axislegend(ax1, position = :rb, labelsize = 11)
save(joinpath(savedir, "loss_vs_a_5rounds.png"), fig1)

# gd tracking
all_param_history = []
all_grad_history = []
flips = []

for result in results
    a_str = @sprintf("%.3e", FlexiBasicLearning.derepresent(result.fit_params.p_classical, my_model).a)
    fig_flexi = FlexiBasicLearning.plot_flexi_history_ode(result.parameter_history, "adam_a$(a_str)", savedir, datafile, flex1_params)
    flex1_param_history = [p for p in result.parameter_history] #TODO: should update grad_desc history collection to store a as well
    flex1_grad_history = [g for g in result.gradient_history]
    append!(all_param_history, flex1_param_history)
    append!(all_grad_history, flex1_grad_history)
    push!(flips, length(all_param_history))
end
# flex_param_history = [p.flex1_params for p in a1_result.parameter_history]
# flex_grad_history = [g.flex1_params for g in a1_result.gradient_history]

gd_tracker = FlexiBasicLearning.gd_tracking( (gradient_history = all_grad_history, parameter_history = all_param_history), 
     flex1_params)
gd_tracker_fig = FlexiBasicLearning.plot_gd_tracker(gd_tracker, "adam", savedir; flip_boundaries = flips)

#fig_flexi = FlexiBasicLearning.plot_flexi_history_ode(a1_result.parameter_history, "a1", savedir, datafile, flex1_params)


# make fig

# plot a_loss_landscape.a_grid vs. a_loss_landscape.loss_values, mark global minimum and chosen a_local_min

# overlay a1_loss_landscape.a_grid vs. a1_loss_landscape.loss_values, mark global minimum
