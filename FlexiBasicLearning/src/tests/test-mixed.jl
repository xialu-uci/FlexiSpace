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
 
ig = my_model.params_derepresented_ig

println(ig)

near_ig = FlexiBasicLearning.choose_near_ig(ig, my_model; cdist = 0.05, fdist = 0.05, fdir = "cu" )

println(near_ig)

result = FlexiBasicLearning.bbo_learn(my_prob, ig) # ok yay loss goes down  

# result = FlexiBasicLearning.simplex_learn(my_prob, ig) # changes happen (not enough iterations for loss to be printed)

# generate a couple guesses from different distances away

# pipe through bbo > simplex > gd

function pipeline_classical_flexi(datafile, savedir, ig; num_flips = 3, optimizer = :gradient_descent)
    @load datafile data
    loss_strategy = "normalized"
    make_model = FlexiBasicLearning.make_ModelMixedLV
    my_prob, my_model = FlexiBasicLearning.set_up_prob(data, make_model, loss_strategy)

    bbo_result = bbo_learn(my_prob, ig)
    guess_for_simplex = bbo_result.fit_params_repr
    guess_for_gd = bbo_result.fit_params_repr.flex1_params # just to instantiate it

    # num_flips = 3
    simplex_results = []
    gd_results = []

    for i = 1:num_flips
        global guess_for_simplex, guess_for_gd
        simplex_result = simplex_learn(my_prob, ig)
        push!(simplex_results, simplex_result)
        guess_for_gd= simplex_result.fit_params_repr.flex1_params

        gd_result = FlexiBasicLearning.gradient_descent_learn(my_prob, ig; 
        optimizer=optimizer, 
        maxiters=10000, save_parameters = true)= 
        push!(gd_results, gd_result)
        guess_for_simplex = ComponentArray(p_classical = simplex_result.fit_params_repr.p_classical, flex1_params = gd_result.fit_params)
    end

    full_result = (bbo_result=bbo_result, simplex_results = simplex_results, gd_results = gd_results)

    return full_result
end

dist_list = [0.5]
dir_list = ["cu", "cd"]
savedir_base = "../FlexiSpaceLocal/exp/08312026/mixed-lv"


# helper function for plotting

for dist in dist_list
    for dir in dir_list
        # make near guess
        # make savedir named for near guess
        # pipeline
        # plots 
        # for plotting: I want to plot the bbo_loss and the concatenated simplex and gd_losses
        # plot gt (this is the actual ig from the saved model) and fit overlays
        # plot all of my gradient descent tracking plots from gd_tracking.jl
    end
end

# plot overlay of fits from each near guess (labeled)

