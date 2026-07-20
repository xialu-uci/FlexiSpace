using FlexiBasicLearning

Random.seed!(1234)

# make sure all my stuff still runs after refactoring
datafile = "../FlexiSpaceLocal/data/no-noise/flexi1-5dof-20obs/sim_data_crooked.jld2"
savedir = "../FlexiSpaceLocal/exp/07202026-test-grad-tracking/no-noise/flexi1-5dof-20obs/crooked"
make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=5)
result1 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, joinpath(savedir, "test1"), make_model; save_parameters = true)
FlexiBasicLearning.plot_loss_and_fits(result1, datafile)


# does gd_tracking() yield a result
gd_result1 = (result1[1])["gd_result"]
gt = FlexiBasicLearning.crooked_flexi(5)
result1_gd_tracker = FlexiBasicLearning.gd_tracking(gd_result1, gt)

# issue: why are all grad norms and distances the same FIXED

# plot stuff
FlexiBasicLearning.plot_gd_tracker(result1_gd_tracker, savedir)

FlexiBasicLearning.plot_param_history(result1_gd, savedir, gt, datafile)
