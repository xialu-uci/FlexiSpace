using FlexiBasicLearning
using ComponentArrays
using JLD2
using Printf
using Zygote
using CairoMakie

datafile = joinpath(FlexiBasicLearning.asset_dir, "data/mixed_true_params/no-noise/a1.0/flexi1lv2-4dof-32obs/sim_data_crooked_tmax50.0.jld2")
savedir_base = joinpath(FlexiBasicLearning.asset_dir, "tests/10122026/lv2_classical_landscape_exploration/gt-mixed_flexi1_lv2_a1.0-crooked4_tmax50.0/")
# mkpath(savedir)

@load datafile true_params

dofs = [4,8,16,32,64]
#d = 4

optimizers = [:cmaes, :bfgs]
opt_strs = ["cmaes", "bfgs"]

for d in dofs, (opt_str, optimizer) in zip(opt_strs, optimizers)

    make_model = () -> FlexiBasicLearning.make_ModelMixedLV(;flexi_dofs = d)

    # ig_derepr= make_model().params_derepresented_ig

    savedir = joinpath(savedir_base, "$opt_str/fit_w_mixed_flexi1_lv2_crooked$d")

    results_file = joinpath(savedir, "results_$opt_str.jld2")

    @load results_file  results_dict
    my_prob = results_dict["my_prob"]
    results = results_dict["intermediate_results"]
    loss_landscapes = results_dict["loss_landscapes"]
    final_result = results_dict["final_result"]
    flips = results_dict["flips"]
    flexi_flips = results_dict["flexi_flips"]
    # my_prob results loss_landscapes

    # my_prob, results, loss_landscapes = fit_mixed_alg(datafile, savedir, make_model; n_rounds = 3, optimizer = optimizer)
    
   # push!(results_list, results)
    
   FlexiBasicLearning.plot_landscapes(my_prob, results, loss_landscapes, savedir)

    if optimizer != :cmaes
        #  all_param_history, all_grad_history, all_loss_history, flips = FlexiBasicLearning.concat_gd_result(results, my_prob, optimizer)

        gd_tracker = FlexiBasicLearning.gd_tracking( (gradient_history = final_result.gradient_history, parameter_history = final_result.parameter_history), 
            true_params.flex1_params)
        gd_tracker_fig = FlexiBasicLearning.plot_gd_tracker(gd_tracker, opt_str, savedir; flip_boundaries = flexi_flips)
        
        flexi_history_fig, full_history_fig = FlexiBasicLearning.plot_param_history( (gradient_history = final_result.gradient_history, parameter_history = final_result.parameter_history), opt_str, savedir, datafile)
    # else
    #     all_loss_history = collect(Iterators.flatten([result.loss_history for result in results]))
    #     # println(all_loss_history)
    end
    
    loss_history_fig = FlexiBasicLearning.make_loss_history_figs([final_result.loss_history], [final_result.time], ["simplex --> $opt_str"], simplex_flexi_boundaries = [flips])

    save(joinpath(savedir, "loss_history_simplex_$opt_str.png"), loss_history_fig)


end
# add plotting of the flexi and full function histories

