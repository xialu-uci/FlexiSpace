# tests for overloaded fit_cmaes_and_gd()
using FlexiBasicLearning

# test 1: not entering ig should use default ig from model
datafile = "../FlexiSpaceLocal/data/flexi1-5dof-20obs/sim_data_crooked.jld2"
savedir = "../FlexiSpaceLocal/exp/07092026-test-mult-ig/flexi1-5dof-20obs/crooked"
make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=5)
result1 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, joinpath(savedir, "test1"), make_model)
FlexiBasicLearning.plot_loss_and_fits(result1, datafile)

# test 2: entering ig should use the provided ig
ig1 = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(5)  # this should be EXACTLY the same as default ig in test 1.
result2 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, joinpath(savedir, "test2"), make_model; igs=[ig1])
FlexiBasicLearning.plot_loss_and_fits(result2, datafile)

# test 3: entering ig that is different from default ig should use the provided ig
ig_list =[]
for i in 1:3
    ig = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(5) .+ 0.1*randn(5)  # this should be different from default ig in test 1.
    # floor it at 0.0 to avoid negative values
    ig = max.(ig, 0.0)
    push!(ig_list, ig) # i still expect same result? maybe
end
result3 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, joinpath(savedir, "test3"), make_model; igs=ig_list)

FlexiBasicLearning.plot_loss_and_fits(result3, datafile)
