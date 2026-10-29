using CairoMakie

# TODO: split into a function that fits and a function that makes the result/cleans
# fit a mixed model



function make_mixed_result(datafile, savedir, num_dofs, make_model_type; save_landscapes = true, exit_condition = nothing,
    ig_derepr = nothing, optimizer = :bfgs, back_up_optimizer = nothing, differ = Zygote.gradient, maxiters = 10000, 
    save_parameters = false, loss_strategy = "normalized", n_rounds = 3)

    @load datafile data

    make_model = () -> make_model_type(;flexi_dofs = num_dofs)

    my_prob, my_model = set_up_prob(data, make_model, loss_strategy) # FlexiBasicLearning.set_up_prob(data, make_model, loss_strategy)
    if isnothing(ig_derepr)
        guess = deepcopy(my_model.params_repr_ig)
    else
        guess = represent_all(ig_derepr, my_model) # FlexiBasicLearning.represent_all(ig_derepr, my_model)
    end
    
    mkpath(savedir) # creates the directory only if it doesn't already exist
    
    results_dict = Dict(
        "datafile" => datafile,
        "save_dir" => savedir,
        "my_prob" => my_prob,
        "ig_derepr" => ig_derepr,
        "exit_condition" => exit_condition
    )

    
    flexi_results, simplex_results, loss_landscapes, exit_condition_met = fit_mixed_loop(my_prob, guess, optimizer, n_rounds, num_dofs, make_model_type, savedir; exit_condition = exit_condition)

    results_dict["exit_condition_met"] = exit_condition_met
    results_dict["intermediate_results"] = flexi_results
    results_dict["loss_landscapes"] = loss_landscapes
    
    
    
    if optimizer == :cmaes
        all_loss_history, all_flips, total_time = concat_final_result(flexi_results, simplex_results, my_prob, optimizer)
       
    else
        all_param_history, all_grad_history, all_loss_history, all_flips, flexi_flips, total_time, back_up_opt_endpts, back_up_opt_usage = concat_final_result(flexi_results, simplex_results, my_prob, optimizer)
        
    end
   
    final_result = (fit_params_repr = flexi_results[end].fit_params_repr, fit_params_derepr = derepresent_all(flexi_results[end].fit_params_repr, my_model),
     loss_history = all_loss_history, optimizer = flexi_results[end].optimizer, time = total_time,
            gradient_history = (optimizer == :cmaes) ? nothing : all_grad_history,
            #parameter_history = config.save_parameters ? parameter_history : nothing,
            parameter_history = (optimizer == :cmaes) ? nothing : all_param_history,
            back_up_opt_endpts = (optimizer == :cmaes) ? nothing : back_up_opt_endpts,
            back_up_opt_usage = (optimizer == :cmaes) ? nothing : back_up_opt_usage
            )

    results_dict["final_result"] = final_result
    results_dict["flips"] = all_flips
    results_dict["flexi_flips"] = (optimizer == :cmaes) ? nothing : flexi_flips

    @save joinpath(savedir, "results_$optimizer.jld2") results_dict
    
    return results_dict 

end

function fit_mixed_loop(my_prob, guess, optimizer, n_rounds, num_dofs, make_model_type, savedir; save_landscapes = true, exit_condition = nothing)
    flexi_results        = Vector{Any}(nothing, n_rounds)   # bfgs (a_result) per round
    simplex_results = Vector{Any}(nothing, n_rounds)   # simplex (a_guess) per round
    loss_landscapes = Vector{Any}(nothing, n_rounds)  # loss-vs-a landscape per round's fit
    exit_condition_met = false
    n = size(my_prob.data)[1]
    k = div(n,2)
    
    # --- Rounds 1-n: simplex finds next a_guess, then bfgs finds flexi ---
    
    for i in 1:n_rounds
    
        new_mask = random_mask(n,k)
        println(length(new_mask))

        my_prob.mask .= new_mask
        simplex_result = simplex_learn(my_prob, guess) # FlexiBasicLearning.simplex_learn(my_prob, guess)
        simplex_results[i] = simplex_result
        
        if optimizer == :cmaes
            flexi_results[i] = cmaes_learn(my_prob, simplex_result.fit_params_repr) 
        else 
            # flexi_results[i] = gradient_descent_learn(my_prob, simplex_result.fit_params_repr; optimizer = optimizer, maxiters = 10000, save_parameters = true)
            flexi_results[i] = refined_flexi_opt(my_prob, simplex_result.fit_params_repr, optimizer, make_model_type, num_dofs; start_dofs = nothing)

            # result_pt1 = gradient_descent_learn(my_prob, simplex_result.fit_params_repr; optimizer = optimizer, maxiters = 10000, save_parameters = true)
            # final_grad_norm = LinearAlgebra.norm(result_pt1.gradient_history[end])
            # if !isnothing(back_up_optimizer) && final_grad_norm > 1e-4
            #     println("Warning: final gradient norm $(final_grad_norm) exceeds threshold 1e-4")
            #     println("Attempting to refine solution using backup optimizer: $back_up_optimizer")
            #     result_pt2 = gradient_descent_learn(my_prob, simplex_result.fit_params_repr; optimizer = back_up_optimizer, maxiters = 10000, save_parameters = true)
            #     flexi_results[i] = concat_gd_result(result_pt1, result_pt2)
            # else
            #     flexi_results[i] = result_pt1
                
            
            # end
            
        end
        guess = flexi_results[i].fit_params_repr
        println("saved result $i")
        if save_landscapes
            loss_landscapes[i] = classical_loss_landscape(guess.flex1_params, my_prob, savedir; step = 0.01, specific_a_value = flexi_results[i].fit_params_derepr.p_classical.a)
            println("saved landscape $i")
        end
    
        # here check for exit condition
        if !isnothing(exit_condition)
            if exit_condition(flexi_results)
                # escape exit with wiggle here.
                println("exit condition met at round $i")
                simplex_results = simplex_results[1:i]
                flexi_results = flexi_results[1:i]
                loss_landscapes = loss_landscapes[1:i]
                # converged = true
                exit_condition_met = true
                break
            end
        end
        
    end
    return flexi_results, simplex_results, loss_landscapes, exit_condition_met
end

function random_mask(n, k)
    mask = zeros(Bool, n)
    mask[randperm(n)[1:k]] .= true
    return mask
end


#TODO: untested
# WIP: refinement

function refined_flexi_opt(my_prob, guess, optimizer, make_model_type, num_dofs; start_dofs = 2)

    if !isnothing(start_dofs)
        dofs_list = [start_dofs,num_dofs]
    else
        dofs_list = [num_dofs]
    end
    maxiters = 100
    results = Vector{Any}(nothing, length(dofs_list))    
    for (i, d) in enumerate(dofs_list)
        # reconstruct guess
        # flex1_params = flexi_equiv(guess.flex1_params, d)
        # guess.flex1_params = flex1_params
        guess = full_equiv(guess,d)
        # reconstruct problem
        make_model = () -> make_model_type(;flexi_dofs = d)
        if d == num_dofs
            maxiters = 10000
        end
        my_prob, my_model = set_up_prob(my_prob.data, make_model, my_prob.loss_strategy)
        result = gradient_descent_learn(my_prob, guess; optimizer = optimizer, maxiters = maxiters, save_parameters = true)
        results[i] = result
        guess = result.fit_params_repr
    end

    
    return  concat_gd_result(results)# use concat_gd_result.
    
end

function concat_gd_result(result_pts)

    # did back up improve on result
    # back_up_improves = false
    # if result_pt2.loss_history[end] < result_pt1.loss_history[end]
    #     back_up_improves = true
    # end

    fit_params_repr = result_pts[end].fit_params_repr
    fit_params_derepr = result_pts[end].fit_params_derepr

    # to do want this to work for any length list of result_pts
    # loss_history = vcat(result_pts[1].loss_history, result_pts[2].loss_history)  
    # splits = length(result_pt1.loss_history) 
    loss_history = reduce(vcat, (r.loss_history for r in result_pts))
    splits = cumsum(length(r.loss_history) for r in result_pts)

    optimizer = result_pts[1].optimizer
    # back_up_optimizer = result_pt2.optimizer
    time = sum(r.time for r in result_pts)# result_pt1.time + result_pt2.time

    save_parameters = !isnothing(result_pts[1].gradient_history)

    if save_parameters
        gradient_history = reduce(vcat, r.gradient_history for r in result_pts) # vcat(result_pt1.gradient_history, result_pt2.gradient_history)
        parameter_history = reduce(vcat, r.parameter_history for r in result_pts) # vcat(result_pt1.parameter_history, result_pt2.parameter_history)
    end




    result = (fit_params_repr = fit_params_repr, fit_params_derepr = fit_params_derepr,
        loss_history = loss_history, optimizer = optimizer, time = time, splits = splits, 
        # back_up_improves = back_up_improves,
        # back_up_optimizer = back_up_optimizer, back_up_start = opt_swap ? back_up_start : nothing,
        # back_up_used = back_up_used,
        gradient_history = save_parameters ? gradient_history : nothing,
        #parameter_history = config.save_parameters ? parameter_history : nothing,
        parameter_history = save_parameters ? parameter_history : nothing)

    return result

end
