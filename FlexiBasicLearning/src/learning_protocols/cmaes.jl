using Optimization, OptimizationEvolutionary
using Statistics

# example cmaes usage
# rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
# x0 = zeros(2)
# p = [1.0, 100.0]
# f = OptimizationFunction(rosenbrock)
# prob = SciMLBase.OptimizationProblem(f, x0, p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
# sol = solve(prob, Evolutionary.CMAES(μ = 40, λ = 100))

# as a kindness to future me, it may be useful to construct a learning problem type situation

function compute_adaptive_sigma0(ig; upper_bound_multiplier = 10.0)
    # Method 1: Scale based on parameter magnitudes (robust to parameter scales)
    # Use a fraction of the typical parameter magnitude
    nonzero_params = ig[abs.(ig) .> 1e-12]  # Exclude near-zero parameters
    if length(nonzero_params) > 0
        typical_magnitude = mean(abs.(nonzero_params))
        magnitude_based = typical_magnitude * 0.1  # 10% of typical parameter size
    else
        magnitude_based = 0.1  # Fallback if all parameters are near zero
    end
    
    # Method 2: Scale based on current bound setup
    flexi_bound = maximum(abs.(ig)) * upper_bound_multiplier
    bound_based = flexi_bound * 0.02  # 2% of the dynamic bound
    
    # Method 3: Scale based on parameter variance
    param_std = std(ig)
    variance_based = param_std * 0.5  # Half the parameter standard deviation
    
    # Combine methods with weights favoring magnitude-based approach
    combined_sigma0 = 0.6 * magnitude_based + 0.3 * bound_based + 0.1 * variance_based
    
    # Apply reasonable bounds
    min_sigma0 = 1e-4
    max_sigma0 = 1.0
    
    return clamp(combined_sigma0, min_sigma0, max_sigma0)
    # return 1e-5 # for now
end

# CMA-ES implementation
function cmaes_learn(learning_problem, ig; upper_bound_multiplier=10.0)
    
    
    # Track best solution during optimization
    best_loss = Inf
    best_flexi_params = nothing
    

    function flexi_loss(params, p)
       return FlexiBasicLearning.get_loss(params; learning_problem=learning_problem)
    end

    # Evaluate initial guess
    # ig = deepcopy(collect(values(copy((flex_all)))))
    initial_loss = flexi_loss(ig, nothing)

    loss_history = Float64[]
    config = CallbackConfig() # just stores info for callback function in fields
    function callback(p, lossval)
        if length(loss_history) < 5
            println("iter 1: loss=$lossval")
            println("p.u = ", p.u)
            println("dist from ig = ", norm(p.u - ig))
            
        end
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        # Track best solution encountered - use fresh loss evaluation instead of lossval
        if !hasfield(typeof(p), :u) # u is obj func val
            error("DEBUGGING: CMA-ES callback parameter missing :u field, type: $(typeof(p))")
        end
        
        # # Evaluate fresh loss at current parameters
        fresh_loss = flexi_loss(p.u, nothing) # u is params
        # println("DEBUGGING: best loss so far: $best_loss, current loss: $fresh_loss at iter $current_iter")
        
        if fresh_loss < best_loss
            best_loss = fresh_loss
            best_flexi_params = deepcopy(p.u)
            # println("New best solution found at iter $current_iter with loss $best_loss")
        end
        
        if config.verbose && current_iter % config.print_frequency == 0
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In cmaes-on-flexi, iteration $current_iter: loss=$lossval, qdrms=$qdrms at $(now())")
            println("sigma0: $cmaes_options")
            flush(stdout)
        end
        
        
        return false
    end

    optf = Optimization.OptimizationFunction(flexi_loss)

    # Set up bounds for flexi parametersnumDof
    # flexi_bound = maximum(abs.(ig)) .* upper_bound_multiplier 
    flexi_bound = 0.5 # for cu, true max is always dof/srt(sum(1 to dof)(x^2)), but initial guess is 1/sqrt(dof)

    lb = 0.0*fill(flexi_bound, length(ig));#-1.0*fill(flexi_bound, length(ig))
    ub = +1.0*fill(flexi_bound, length(ig))

    prob = Optimization.OptimizationProblem(optf, ig, [1.0, 100.0]; lb=lb, ub=ub)

    # BELOW: protocol dependencies
    # Build CMAES options with hyperparameters
    cmaes_options = Dict{Symbol, Any}()
    cmaes_options[:μ] = 40 # from jun protocol.mu
    cmaes_options[:λ] =100  # from jun protocol.lambda
    # cmaes_options[:λ] = max(100, 4 * length(ig))  # λ = 4 * number of parameters, at least 100
    # cmaes_options[:μ] = div(cmaes_options[:λ], 2)

# use default hyperparameters
 
    
    # Handle sigma0: use adaptive computation if sigma0 = -1, otherwise use specified value
    # if protocol.sigma0 == -1.0upper_bound_multiplier =upper_bound_multiplier = 10.0 10.0
    adaptive_sigma0 = compute_adaptive_sigma0(ig; upper_bound_multiplier=upper_bound_multiplier)
    cmaes_options[:sigma0] = adaptive_sigma0
    println("Using adaptive sigma0 = $adaptive_sigma0 (computed from initial guess)")
    
    
    sol = solve(prob, Evolutionary.CMAES(; cmaes_options...); 
                callback=callback, maxiters=3e7) # num iteration fed to here. also track best
    println("Ran $(length(loss_history)) iterations (maxiters was $(3e7))")
    println("Final retcode: $(sol.retcode)")
    println("CMAES fields: ", fieldnames(typeof(Evolutionary.CMAES())))    # println(sol)
    println(sol.original)
   #  println(sol.minimizer)
    # Determine which solution to return: initial guess vs best found vs final solution
    final_loss_from_sol = flexi_loss(collect(sol.u), nothing)
    
    # Choose the best among: initial guess, best during optimization, final solution
    candidates = [
        (initial_loss, ig, "initial guess"),
        (best_loss, best_flexi_params, "best during optimization"), 
        (final_loss_from_sol, collect(sol.u), "final solution")
    ]
    
    
    best_candidate_idx = argmin([x[1] for x in candidates])
    chosen_loss, chosen_flexi_params, chosen_source = candidates[best_candidate_idx]
    
    
    println("CMA-ES: Chose $chosen_source with loss $chosen_loss")
    println("  Initial guess loss: $initial_loss")
    println("  Best during optimization: $best_loss") 
    println("  Final solution loss: $final_loss_from_sol")


    # convergence_status = sol.retcode == :success ? :converged : :max_iterations
    
    return chosen_flexi_params, loss_history
end# CMA-ES Learning Protocol Implementation


