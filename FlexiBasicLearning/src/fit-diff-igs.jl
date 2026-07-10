# tests for overloaded fit_cmaes_and_gd()

# test 1: not entering ig should use default ig from model
datafile = "../FlexiSpaceLocal/data/flexi1-5dof-20obs/sim_data_crooked_flexi.jld2"
savedir = "../FlexiSpaceLocal/exp/07092026-test-mult-ig/flexi1-5dof-20obs/crooked"
make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=5)
result1 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, savedir, make_model)
FlexiBasicLearning.plot_loss_and_fits(result1, datafile, savedir)

# test 2: entering ig should use the provided ig
ig = FlexiBasicLearning.generate_flexi_ig(5)  # this should be EXACTLY the same as default ig in test 1.
result2 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, savedir, make_model; ig=ig)

# test 3: entering ig that is different from default ig should use the provided ig
ig_list =[]
for i in 1:3
    ig = FlexiBasicLearning.generate_flexi_ig(5) .+ 0.1*randn(5)  # this should be different from default ig in test 1.
    # floor it at 0.0 to avoid negative values
    ig = max.(ig, 0.0)
    push!(ig_list, ig) # i still expect same result? maybe
    result3 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, savedir, make_model; ig=ig)
    FlexiBasicLearning.plot_loss_and_fits(result3, datafile, savedir)
end