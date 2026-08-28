
# for objective function, need to define function that takes in parameter array and returns loss (pass get_loss with reconstructed params)
# function obj_func(x, p_repr_ig)
#     p_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(x, p_repr_ig,learning_problem.model) # this is where params are updated # the trick is the x is the actual params we want. 
#     # only pass through p_repr_ig for the keys
#     return CombiCellModelLearning.get_loss(p_repr; learning_problem=learning_problem)
# end

# let's make this whole section a function bbo_learn(learning_problem, p_repr_ig)
# takes and returns p_repr_ig (and only trains/computes loss using p_repr_ig.p_classical)
function bbo_learn(learning_problem, p_repr_ig)

    function obj_func(x, p)
        p_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(x, p_repr_ig,learning_problem.model) # this is where params are updated # the trick is the x is the actual params we want. 
        # only pass through p_repr_ig for the keys
        return CombiCellModelLearning.get_loss(p_repr; learning_problem=learning_problem)
    end
    # initial guess params array
    classical_params_array = collect(values(copy(p_repr_ig.p_classical)))

    # p
    p = [1.0, 100.0] # not used in obj_func, honestly I think i could delete this but will leave in for now and see if it changes anything later

    # algo is the optimization algorithm (here bbo)
    # maxiters is maximum iterations
    maxiters = 300000 # not reduced for testing?
    # callback is a function called at each iteration, s.t. optimzation stops if it returns true
    config = CallbackConfig() # just stores info for callback function in fields
    callback, loss_history = CombiCellModelLearning.create_bbo_callback_with_early_termination(
    config, maxiters)



    # create optimization problem
    prob = Optimization.OptimizationProblem(
        obj_func,
        classical_params_array,
        p;
        lb= learning_problem.p_repr_lb,
        ub=learning_problem.p_repr_ub)

    # solve optimization problem
    sol = solve(prob, BBO_adaptive_de_rand_1_bin(); callback=callback, maxiters=maxiters)
    final_params_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(sol.minimizer, p_repr_ig, learning_problem.model)
    # final_params_derepr = CombiCellModelLearning.derepresent_all(final_params_repr, intPoints, learning_problem.model)

    result = (fit_params_repr = final_params_repr, loss_history = loss_history) # returns p_repr

    return result
end


# function bbo_learn_single(learning_problem, p_repr_ig, intPoints)

#     function obj_func(x, p)
#         p_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(x, p_repr_ig,learning_problem.model) # this is where params are updated # the trick is the x is the actual params we want. 
#         # only pass through p_repr_ig for the keys
#         return CombiCellModelLearning.get_single_loss(p_repr, intPoints; learning_problem=learning_problem)
#     end
#     # initial guess params array
#     classical_params_array = collect(values(copy(p_repr_ig)))

#     # p
#     p = [1.0, 100.0] # not used in obj_func, honestly I think i could delete this but will leave in for now and see if it changes anything later

#     # algo is the optimization algorithm (here bbo)
#     # maxiters is maximum iterations
#     maxiters = 300000 # not reduced for testing?
#     # callback is a function called at each iteration, s.t. optimzation stops if it returns true
#     config = CallbackConfig() # just stores info for callback function in fields
#     callback, loss_history = CombiCellModelLearning.create_bbo_callback_with_early_termination(
#     config, maxiters)



#     # create optimization problem
#     prob = Optimization.OptimizationProblem(
#         obj_func,
#         classical_params_array,
#         p;
#         lb= learning_problem.p_repr_lb,
#         ub=learning_problem.p_repr_ub)

#     # solve optimization problem
#     sol = solve(prob, BBO_adaptive_de_rand_1_bin(); callback=callback, maxiters=maxiters)
#     final_params_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(sol.minimizer, p_repr_ig, learning_problem.model)
#     # final_params_derepr = CombiCellModelLearning.derepresent_all(final_params_repr, intPoints, learning_problem.model)

#     return final_params_repr, loss_history
# end
