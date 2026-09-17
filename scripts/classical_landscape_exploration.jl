using FlexiBasicLearning
using ComponentArrays
using JLD2
using Printf
using Zygote
# goal: explore classical loss landscape for lv2 (nondim). 
# sim data: ground truth is a = 1.0, dofs = 8, shape = crooked, num_points = 32
# train with 2 dofs, [8, 64]. 

# (1) save loss landscape with flexi = id

# (2) train simplex starting with initial guess a =1.5, flexi = id

    # simplex should get to global min

# (3) train bfgs or cmaes to get flexi = flexi_it1

# (4) save loss landscape with flexi = flexi_it1

# (5) repeat (3) -> (4) for num_rounds

# plot landscape.

# helper function computes loss landscape
function classical_loss_landscape(flex1_params, learning_problem, savedir; step = 0.01)
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

# fit a mixed model

function fit_mixed_alg(datafile, savedir, make_model; 
    ig_derepr = nothing, optimizers = :bfgs, differ = Zygote.gradient, maxiters = 10000, 
    save_parameters = false, time_grads = false,
    loss_strategy = "normalized", n_rounds = 3)
    @load datafile data
    
    
    my_prob, my_model = FlexiBasicLearning.set_up_prob(data, make_model, loss_strategy)
    if isnothing(ig_derepr)
        guess = deepcopy(my_model.params_repr_ig)
    else
        guess = FlexiBasicLearning.represent_all(ig_derepr, my_model)
    end
    
    mkpath(savedir) # creates the directory only if it doesn't already exist
    
    
    results        = Vector{Any}(undef, n_rounds)   # bfgs (a_result) per round
     loss_landscapes = Vector{Any}(undef, n_rounds)  # loss-vs-a landscape per round's fit
    
    # --- Rounds 1-n: simplex finds next a_guess, then bfgs finds flexi ---
    for i in 1:n_rounds
        # prev_fit = results[i - 1].fit_params  # derepresented params from previous round's bfgs fit
        # global guess
    
        simplex_result = FlexiBasicLearning.simplex_learn(my_prob, guess)

        # guess_derepr = FlexiBasicLearning.derepresent_all(simplex_result.fit_params_repr, my_model)
        # a_guesses[i] = guess_derepr.p_classical.a
        # a_guess_repr = FlexiBasicLearning.represent_all(simplex_result.fit_params_repr, my_model)
        # simplex_result.fit_params_repr is guess_repr
        results[i] = FlexiBasicLearning.gradient_descent_learn(my_prob, simplex_result.fit_params_repr; optimizer = :adam,
                        maxiters = 10000, save_parameters = true)
        guess = results[i].fit_params_repr
        println("saved result $i")
        loss_landscapes[i] = classical_loss_landscape(guess.flex1_params, my_prob, savedir; step = 0.01)
        println("saved landscape $i")

    end
    # simplex_result = FlexiBasicLearning.simplex_learn(my_prob, a1_result.fit_params; maxiters = 10000, save_parameters = true)
    # a_guess_for_gd = simplex_result.fit_params_repr.flex1_params
    println("out of for loop")
    return my_prob, results, loss_landscapes
    # save results, loss_landscapes, a_guesses to savedir to reload and plot later
    @save joinpath(savedir, "results_landscapes_guesses.jld2") my_prob results loss_landscapes


end

function concat_gd_result(results,)
    all_param_history = []
    all_grad_history = []
    flips = []

    for result in results
        a_str = @sprintf("%.3e", result.fit_params_derepr.p_classical.a)
        # fig_flexi = FlexiBasicLearning.plot_flexi_history_ode(result.parameter_history, "adam_a$(a_str)", savedir, datafile, flex1_params)
        flex1_param_history = [p for p in result.parameter_history] #TODO: should update grad_desc history collection to store a as well
        flex1_grad_history = [g for g in result.gradient_history]
        append!(all_param_history, flex1_param_history)
        append!(all_grad_history, flex1_grad_history)
        push!(flips, length(all_param_history))
    end
    return all_param_history, all_grad_history, flips
end

function plot_landscapes(my_prob, results, loss_landscapes)
    ig_derepr = my_prob.model.params_derepresented_ig
    a_ig = ig_derepr.p_classical.a
    og_loss_landscape = classical_loss_landscape(ig_derepr.flex1_params, my_prob, savedir; step = 0.01)
    n_rounds = length(results)
    fig1 = Figure(size = (1000, 700))
    ax1 = CairoMakie.Axis(fig1[1, 1],
        xlabel = "a",
        ylabel = "loss",
        yscale = log10,
        title = "Loss landscape vs. a across $(n_rounds) rounds"
    )

    # baseline: flex1_params = id landscape, for reference
    # a_global_min_idx = argmin(a_loss_landscape.loss_values)
    lines!(ax1, og_loss_landscape.a_grid, og_loss_landscape.loss_values,
        color = :gray, linestyle = :dot, label = "flex1_params = id (baseline)")
    scatter!(ax1, [og_loss_landscape.a_grid[argmin(og_loss_landscape.loss_values)]],
        [og_loss_landscape.loss_values[argmin(og_loss_landscape.loss_values)]],
        color = :gray, markersize = 14, marker = :star5)

    # distinct colors per round
    round_colors = Makie.wong_colors()[1:n_rounds]

    # a_guesses
    a_guesses = [result.fit_params_derepr.p_classical.a for result in results]
    
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
    save(joinpath(savedir, "loss_vs_a_$(n_rounds)rounds.png"), fig1)


end


datafile = "../FlexiSpaceLocal/data/mixed_true_params/no-noise/a1.0/flexi1lv2-4dof-32obs/sim_data_crooked.jld2"
savedir_base = "../FlexiSpaceLocal/exp/09172026/lv2_classical_landscape_exploration/gt-a1-crooked4"
# mkpath(savedir)

@load datafile true_params



make_model = () -> FlexiBasicLearning.make_ModelMixedLV(;flexi_dofs = 4)

# ig_derepr= make_model().params_derepresented_ig

savedir = joinpath(savedir_base, "fit_w_mixed_flexi1_lv2_crooked4")

my_prob, results, loss_landscapes = fit_mixed_alg(datafile, savedir, make_model)

plot_landscapes(my_prob, results, loss_landscapes)

all_param_history, all_grad_history, flips = concat_gd_result(results)

gd_tracker = FlexiBasicLearning.gd_tracking( (gradient_history = all_grad_history, parameter_history = all_param_history), 
     true_params.flex1_params)
gd_tracker_fig = FlexiBasicLearning.plot_gd_tracker(gd_tracker, "adam", savedir; flip_boundaries = flips)


# add plotting of the flexi and full function histories