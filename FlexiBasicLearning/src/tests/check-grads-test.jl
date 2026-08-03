using FlexiBasicLearning
using Random

Random.seed!(1234)

# 
datafile = "../FlexiSpaceLocal/data/flexi1-5dof-20obs/sim_data_crooked.jld2"
savedir = "../FlexiSpaceLocal/exp/07132026-test-check-grads/flexi1-5dof-20obs/crooked"
make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=5)

result1 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, joinpath(savedir, "test1"), make_model)
# test 1:  gd_grad_norm_history in result1 and non-empt
grads = result1[1]["gd_grad_norm_history"]
println(length(grads))
println(maximum(grads))
println(minimum(grads))
# FlexiBasicLearning.plot_loss_and_fits(result1, datafile)

# test 2: let's try adding plotting
FlexiBasicLearning.plot_loss_and_fits(result1, datafile)