# other loss slices tested directly in loss_slicing.jl, see comments
# test 1: loss vs. p, where p is a linear interpolation between ig and true_params
using FlexiBasicLearning
using JLD2
using CairoMakie

datafile = "../FlexiSpaceLocal/data/flexi1-5dof-20obs/sim_data_crooked.jld2"
savedir = "../FlexiSpaceLocal/exp/07102026/p-loss-tests/flexi1-5dof-20obs/crooked"
make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=5)
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

ig = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(5)
igs = [ig, ig .+ 0.1*randn(5), ig .+ 0.2*randn(5)] # all igs must be non-negative, so we will floor them at 0.0
igs = [max.(ig, 0.0) for ig in igs] #

arcs = FlexiBasicLearning.loss_from_ig_to_true(igs, true_params, learning_problem, savedir; step = 0.02, plot=true);

# want to print each change in direction of loss for each result
# print (p,loss)_i if (p,loss)(i-1) > (p,loss)i AND (p<loss)(i+1)< (p,loss)i or vice versa

for arc in arcs
    println(result.ig)
    p_range = arc.p_range
    loss_values = arc.loss_values
    for i in 2:(length(loss_values)-1)
        pre = loss_values[i] -loss_values[i-1]
        post = loss_values[i+1]-loss_values[i]
        if !(signbit(pre)==signbit(post))
            println("caught!")
            println(p_range[i])
            println(loss_values[i])
        end
    end
end


