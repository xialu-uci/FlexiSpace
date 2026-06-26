using JLD2
using FlexiBasicLearning
#load crooked sim data

datafile = "../FlexiSpaceLocal/data/sim_data_crooked_5seg.jld2"

@load datafile data
num_points = size(data)[1]
# no mask for now
full = Vector{Bool}(trues(num_points))

# set up model, LearningProblem
my_model = FlexiBasicLearning.make_ModelFlexi1()
my_prob = LearningProblem(
    data = data,
    model = my_model,
    mask = full
)

# fit with cmaes
ig = my_model.params
cmaes_fit_params, cmaes_loss_history = FlexiBasicLearning.cmaes_learn(my_prob, ig)