using FlexiBasicLearning
using JLD2 
using ComponentArrays
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
@load datafile data
my_prob, my_model = FlexiBasicLearning.set_up_prob(data, () -> FlexiBasicLearning.make_ModelMixedLV(; flexi_dofs = 4), "normalized")
flex1_params = my_model.params_derepresented_ig.flex1_params

a_loss_landscape = find_a_local_minima(flex1_params, my_prob, savedir; step = 0.01)

a_global_min_idx = argmin(a_loss_landscape.loss_values)
a_local_min = a_loss_landscape.a_grid[a_global_min_idx + 1] # because I want local minima closest to global minima on the right

