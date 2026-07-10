using FlexiBasicLearning
using JLD2
using CairoMakie

# load simplest sim data
datafile = "../FlexiSpaceLocal/data/flexi1alg1-5dof-20obs/sim_data_crooked.jld2"
savedir = "../FlexiSpaceLocal/exp/07062026/flexi1alg1-5dof-20obs/crooked"
make_model = () -> FlexiBasicLearning.make_ModelFlexiAlg(flexi_dofs=5)
mkpath(savedir)  # creates the directory if it doesn't already exist

@load datafile data
num_points = size(data)[1]
# no mask for now
full = Vector{Bool}(trues(num_points))
learning_problem = LearningProblem(
    data = data,
    model = make_model(),
    mask = full
)

true_params = FlexiBasicLearning.crooked_flexi(5)

print(true_params)
# fit with cmaes and gd

function loss_arc(i, true_params, learning_problem, savedir; plot = true)
    # plot loss as a function of param i, with param j determined by
    # the constraint norm(params) == 1, holding all other params fixed at true_params
    num_points = 100
    max_val = 1.0
    param_range = range(0.0, max_val, length=num_points)
    loss_values = zeros(num_points)

    for (idx_i, val_i) in enumerate(param_range)
        params = deepcopy(true_params)
        params[i] = val_i
        # params[j] = sqrt(max(0.0, max_val^2 - val_i^2))
        loss_values[idx_i] = FlexiBasicLearning.get_loss(params; learning_problem=learning_problem)
    end

    if plot
        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1], xlabel = "Param $i", ylabel = "Loss",
                   title = "Loss Arc: Param $i")
        lines!(ax, param_range, loss_values)
        save(joinpath(savedir, "loss_arc_$(i).png"), fig)
    end
    return param_range, loss_values
end

function p_loss(ig, true_params, learning_problem, savedir; num_points = 100, plot = true)
    # plot loss as a function of param i, with param j determined by
    # the constraint norm(params) == 1, holding all other params fixed at true_params
    between(p) = p*true_params + (1.0 - p)*ig
    p_range = range(0.0, 2.0, length=num_points)
    for p in p_range
        params = between(p)
        loss = FlexiBasicLearning.get_loss(params; learning_problem=learning_problem)
        # println("p=$p, loss=$loss")
    end
    # param_range = range(0.0, max_val, length=num_points)
    # loss_values = zeros(num_points)

    # for (idx_i, val_i) in enumerate(param_range)
    #     params = deepcopy(true_params)
    #     params[i] = val_i
    #     # params[j] = sqrt(max(0.0, max_val^2 - val_i^2))
    #     loss_values[idx_i] = FlexiBasicLearning.get_loss(params; learning_problem=learning_problem)
    # end
    # save ig and true_params to savedir
    @save joinpath(savedir, "ig_and_true_params.jld2") ig true_params 
    if plot
        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1], xlabel = "p, where f(p) = p*θ_true + (1-p)*θ_ig", ylabel = "Loss",
                   title = "Loss vs. p, ig, truth saved to $savedir")
        lines!(ax, p_range, loss_values)
        save(joinpath(savedir, "loss_arc_$(i).png"), fig)
    end
    return p_range, loss_values
end

function loss_from_ig_to_true(igs, true_params, learning_problem, savedir; num_points = 100, plot = true)
    # plot loss as a function of param i, with param j determined by
    # the constraint norm(params) == 1, holding all other params fixed at true_params
    for ig in igs
        dist = norm(ig - true_params)
        println("ig = $ig, dist from true_params = $dist, max ig = $(round(max(ig), digits=3))")

        subdir = joinpath(savedir, "ig_dist$(round(dist, digits=3))_max$(round(max(ig), digits=3))")
        p_loss(ig, true_params, learning_problem, subdir; num_points=num_points, plot=plot)
    end
end

function loss_slice(i, j, dofs, true_params, learning_problem, savedir; plot = true)
    num_points = 100
    zeroed_ij = deepcopy(true_params)
    zeroed_ij[i] = 0.0
    zeroed_ij[j] = 0.0
    max_val = 1.0
    param_range = range(0.0, max_val, length=num_points)
    loss_values = zeros(num_points, num_points)

    for (idx_i, val_i) in enumerate(param_range)
        for (idx_j, val_j) in enumerate(param_range)
            params = deepcopy(true_params)
            params[i] = val_i
            params[j] = val_j
            loss_values[idx_i, idx_j] = FlexiBasicLearning.get_loss(params; learning_problem=learning_problem)
        end
    end

    if plot
        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1], xlabel = "Param $i", ylabel = "Param $j", title = "Loss Slice")
        hm = heatmap!(ax, param_range, param_range, loss_values', colormap = :viridis)
        Colorbar(fig[1, 2], hm)
        save(joinpath(savedir, "loss_slice_$(i)_$(j).png"), fig)
    end
    return loss_values
end


# all_loss_arc = []
# for i in 1:5
#     param_range, loss_values = loss_arc(i, true_params, learning_problem; plot = true)
#     push!(all_loss_arc, (i, param_range, loss_values))
# end

# all_loss_slice = []
# for i in 1:5, j in i+1:5
#     loss_values = loss_slice(i, j, 5, true_params, learning_problem; plot = true)
#     push!(all_loss_slice, (i, j, loss_values))
# end


