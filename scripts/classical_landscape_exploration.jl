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