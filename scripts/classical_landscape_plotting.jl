datafile = "../FlexiSpaceLocal/data/mixed_true_params/no-noise/a1.0/flexi1lv2-4dof-32obs/sim_data_crooked.jld2"
savedir_base = "../FlexiSpaceLocal/tests/09232026/lv2_classical_landscape_exploration/gt-mixed_flexi1_lv2_a1.0-crooked4/debug-loss-history-plots-cmaes"
# mkpath(savedir)

@load datafile true_params

# dofs = [4, 32, 64]
d = 4

optimizer = :cmaes
opt_str = "cmaes"

#for d in dofs

    make_model = () -> FlexiBasicLearning.make_ModelMixedLV(;flexi_dofs = d)

    # ig_derepr= make_model().params_derepresented_ig

    savedir = joinpath(savedir_base, "fit_w_mixed_flexi1_lv2_crooked$d")

    results_file = joinpath(savedir, "results_$opt_str.jld2")

    @load results_file  results_dict
    my_prob = results_dict["my_prob"]
    results = results_dict["intermediate_results"]
    loss_landscapes = results_dict["loss_landscapes"]
    # my_prob results loss_landscapes

    # my_prob, results, loss_landscapes = fit_mixed_alg(datafile, savedir, make_model; n_rounds = 3, optimizer = optimizer)
    
   # push!(results_list, results)
    
   plot_landscapes(my_prob, results, loss_landscapes, savedir)

    if optimizer != :cmaes
        all_param_history, all_grad_history, all_loss_history, flips = concat_gd_result(results, my_prob)

        gd_tracker = FlexiBasicLearning.gd_tracking( (gradient_history = all_grad_history, parameter_history = all_param_history), 
            true_params.flex1_params)
        gd_tracker_fig = FlexiBasicLearning.plot_gd_tracker(gd_tracker, opt_str, savedir; flip_boundaries = flips)
        
        flexi_history_fig, full_history_fig = FlexiBasicLearning.plot_param_history( (gradient_history = all_grad_history, parameter_history = all_param_history), opt_str, savedir, datafile)
    else
        all_loss_history = collect(Iterators.flatten([result.loss_history for result in results]))
        # println(all_loss_history)
    end
    
    loss_history_fig = FlexiBasicLearning.make_loss_history_figs([all_loss_history], [0.0], ["simplex --> $opt_str"])

    save(joinpath(savedir, "loss_history_simplex_$opt_str.png"), loss_history_fig)


#end
# add plotting of the flexi and full function histories

