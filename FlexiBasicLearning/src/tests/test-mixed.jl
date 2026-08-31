# pick some lv data file
using FlexiBasicLearning
using JLD2
using Optimization, OptimizationBBO

datafile = "../FlexiSpaceLocal/data/w_true_params_flexi_args/no-noise/flexi1lv2-4dof-32obs/sim_data_crooked.jld2"
@load datafile data
loss_strategy = "normalized"
make_model = FlexiBasicLearning.make_ModelMixedLV
my_prob, my_model = FlexiBasicLearning.set_up_prob(data, make_model, loss_strategy)

# 1: does it have everything it should have?
 
ig = my_model.params_repr_ig

result = FlexiBasicLearning.bbo_learn(my_prob, ig) # ok yay loss goes down  

#