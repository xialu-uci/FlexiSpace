
# datafile = "../FlexiSpaceLocal/data/flexi1-2dof-20obs/sim_data_cu.jld2"
# savedir = "../FlexiSpaceLocal/exp/07142026/p-loss-tests/flexi1-2dof-20obs/cu"
# make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=2)
# mkpath(savedir)  # creates the directory if it doesn't already exist

# @load datafile data
# true_params = FlexiBasicLearning.cu_flexi(2)

# num_points = size(data)[1]
# # no mask for now
# full = Vector{Bool}(trues(num_points))
# learning_problem = LearningProblem(
#     data = data,
#     model = make_model(),
#     mask = full
# )

# ig = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(2)
# igs = [ig, ig .+ 0.1*randn(2), ig .+ 0.2*randn(2)] # all igs must be non-negative, so we will floor them at 0.0
# igs = [max.(ig, 0.0) for ig in igs] #

# arcs = FlexiBasicLearning.loss_from_ig_to_true(igs, true_params, learning_problem, savedir; step = 0.02, plot=true);

# # function find_local_minima_params(local_min_idxs)

# datafile = "../FlexiSpaceLocal/data/flexi1-5dof-20obs/sim_data_crooked.jld2"
# savedir = "../FlexiSpaceLocal/exp/07142026/p-loss-tests/flexi1-5dof-20obs/crooked"
# make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=5)
# mkpath(savedir)  # creates the directory if it doesn't already exist

# @load datafile data
# true_params = FlexiBasicLearning.crooked_flexi(5)

# num_points = size(data)[1]
# # no mask for now
# full = Vector{Bool}(trues(num_points))
# learning_problem = LearningProblem(
#     data = data,
#     model = make_model(),
#     mask = full
# )

# ig = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(5)
# igs = [ig, ig .+ 0.1*randn(5), ig .+ 0.2*randn(5)] # all igs must be non-negative, so we will floor them at 0.0
# igs = [max.(ig, 0.0) for ig in igs] #

# arcs = FlexiBasicLearning.loss_from_ig_to_true(igs, true_params, learning_problem, savedir; step = 0.02, plot=true);
using FlexiBasicLearning
using Random

Random.seed!(1234)

function look_for_nonconvexity(datafile, savedir, flexi_dofs, shape)

    make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=flexi_dofs)
    mkpath(savedir)  # creates the directory if it doesn't already exist

    @load datafile data
    true_params = shape(flexi_dofs)

    num_points = size(data)[1]
    # no mask for now
    full = Vector{Bool}(trues(num_points))
    learning_problem = LearningProblem(
        data = data,
        model = make_model(),
        mask = full
    )

    ig = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(flexi_dofs)
    igs = [ig, ig .+ 0.1*randn(flexi_dofs), ig .+ 0.2*randn(flexi_dofs)] # all igs must be non-negative, so we will floor them at 0.0
    igs = [max.(ig, 0.0) for ig in igs] #

    arcs = FlexiBasicLearning.loss_from_ig_to_true(igs, true_params, learning_problem, savedir; step = 0.02, plot=true);

end

# dof_list = [2,3,4,5]
dof_list = [20, 50]
shapes = [FlexiBasicLearning.crooked_flexi, FlexiBasicLearning.cu_flexi, FlexiBasicLearning.cd_flexi]

for dof in dof_list
    for shape in shapes
        sname = FlexiBasicLearning.shape_name(shape)

        datafile = "../FlexiSpaceLocal/data/flexi1-$(dof)dof-20obs/sim_data_$(sname).jld2"
        savedir  = "../FlexiSpaceLocal/exp/07142026/p-loss-tests/flexi1-$(dof)dof-20obs/$(sname)"

        println("Running dof=$(dof), shape=$(sname)")
        look_for_nonconvexity(datafile, savedir, dof, shape)
    end
end


