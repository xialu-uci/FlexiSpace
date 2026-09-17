function simplex_learn(learning_problem, p_repr_ig)

    function obj_func(x, p)
        p_repr = FlexiBasicLearning.reconstruct_learning_params_from_array(x, p_repr_ig, learning_problem.model) # this is where params are updated # the trick is the x is the actual params we want. 
        # only pass through p_repr_ig for the keys
        return FlexiBasicLearning.get_loss(p_repr; learning_problem=learning_problem)
    end
    # initial guess params array
    classical_params_array = collect(values(copy(p_repr_ig.p_classical)))

    # p
   #  p = [1.0, 100.0] # not used in obj_func, honestly I think i could delete this but will leave in for now and see if it changes anything later

    # algo is the optimization algorithm (here bbo)
    # maxiters is maximum iterations
    maxiters = 300000 # not reduced for testing?
    # callback is a function called at each iteration, s.t. optimzation stops if it returns true
    config = CallbackConfig() # just stores info for callback function in fields

    # parameter_history = logging ? [] : nothing # generally not loging
    
    callback, loss_history = FlexiBasicLearning.create_standard_callback("simplex", config)
    
    prob = Optimization.OptimizationProblem(
        obj_func,
        classical_params_array;
        
        lb= ComponentArray(FlexiBasicLearning.represent(learning_problem.model.p_derepresented_lowerbounds, learning_problem.model)),
        ub=ComponentArray(FlexiBasicLearning.represent(learning_problem.model.p_derepresented_upperbounds, learning_problem.model)))

    opt = NLopt.LN_SBPLX()

    # hardcoding solve options for now, can make more flexible later if needed
    solve_options = Dict{Symbol, Any}()
    solve_options[:reltol] = 1e-6 
    solve_options[:abstol] = 1e-6

    sol = solve(prob, opt; callback=callback, maxiters=maxiters, solve_options...)

    final_params_repr = FlexiBasicLearning.reconstruct_learning_params_from_array(sol.u, p_repr_ig, learning_problem.model)
    #final_params_derepr = CombiCellModelLearning.derepresent_all(final_params_repr, intPoints, learning_problem.model)

    result = (fit_params_repr = final_params_repr, fit_params_derepr = FlexiBasicLearning.derepresent_all(final_params_repr, learning_problem.model), loss_history = loss_history) # returns p_repr

    return result


end