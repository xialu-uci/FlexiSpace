
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using SciMLSensitivity
using LineSearches
using Zygote
# using FlexiBasicLearning
using FiniteDiff
using ForwardDiff



# takes p_repr.flex1_params
function gradient_descent_learn(learning_problem, ig; 
    optimizer = :gradient_descent, 
    back_up_optimizer = nothing, # use after checking if final gradient norm is small enough
    differ = :zygote, # also try ForwardDiff.gradient, FiniteDiff.grad, FiniteDiff.finite_difference_gradient
    learning_rate=0.01, # for adam
    linesearch=nothing, # for gd or bfgs
    maxiters=10000, 
    print_frequency = 1000, 
    save_parameters = false, 
    time_grads = false)

    # timing
    t0 = Base.time()

    p_classical_fixed = copy(ig.p_classical)   # outside, once

    function flexi_loss(flexi_params, p)
        # TODO: make this compatible with the unmixed models too
        params = ComponentArray(
            p_classical = p_classical_fixed,
            flex1_params = flexi_params
        )
        # println("params in flexi_loss = $params")
        loss = get_loss(params; learning_problem=learning_problem, gradient_mode=true)
    
       return loss
    end

    config = CallbackConfig(;print_frequency = print_frequency, save_parameters = save_parameters, time_grads = time_grads)


   

    if config.time_grads

        n = length(ig.flex1_params)

        grad_eval_count = Ref(0)

        # build once — outside the loop — so it's non-allocating on every call
        fd_cache = FiniteDiff.GradientCache(zeros(n), zeros(n))

        # a single unified gradient! function, chosen by `differ`
        grad_fn! = if differ == :zygote
            (G, params, p) -> begin
                #println(length(G))
                #println(Zygote.gradient(θ -> flexi_loss(θ, p), params))
                g = Zygote.gradient(θ -> flexi_loss(θ, p), params)[1]
                G .= g
            end
            elseif differ == :forwarddiff
                (G, params, p) -> begin
                    G .= ForwardDiff.gradient(θ -> flexi_loss(θ, p), params)
                end
            elseif differ == :finitediff
                (G, params, p) -> begin
                    FiniteDiff.finite_difference_gradient!(G, θ -> flexi_loss(θ, p), params, fd_cache)
                end
            else
                error("Unknown differ: $differ")
            end

        function counted_grad!(G, params, p)
            grad_eval_count[] += 1
            grad_fn!(G, params, p)
            return nothing
        end
    end


    loss_history = Float64[]
    best_loss = Ref(Inf)
    best_params = Ref{Any}(nothing)
    opt_swap = false # flag to indicate if we have swapped to backup optimizer
    # grad_norm_history = Float64[]
    

    parameter_history = config.save_parameters ? [] : nothing
    gradient_history = config.save_parameters ? [] : nothing
    grad_time_history = config.time_grads ? [] : nothing
    grad_alloc_history = config.time_grads ? [] : nothing # new
    
    timing_buf = config.time_grads ? zeros(n) : nothing 

    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        if lossval < best_loss[]
            best_loss[] = lossval
            best_params[] = copy(p.u)
        end

        if config.save_parameters
            # if current_iter % config.print_frequency == 0
            push!(gradient_history, copy(p.grad))
            push!(parameter_history, copy(p.u))
                # push!(save_its, )
            # end   
        end

        if config.time_grads
            grad_fn!(timing_buf, p.u, nothing)  # warm-up / compilation
            t0 = time_ns() # TODO: add memory
            grad_fn!(timing_buf, p.u, nothing)
            elapsed = (time_ns() - t0) / 1e9
            push!(grad_time_history, elapsed)

            allocs = @allocated grad_fn!(timing_buf, p.u, nothing) # new
            push!(grad_alloc_history, allocs) # new
        end


        if config.verbose && current_iter % config.print_frequency == 0 
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In grad_descent, iteration $current_iter: loss=$lossval, qdrms=$qdrms at $(now())")
            flush(stdout)
        end

        # early stopping: if within sqrt(eps()) of the minimum loss, stop early (this is consistent with hager zhang 2006)
        if lossval < config.min_loss + sqrt(eps())
            println("Early stopping: loss $lossval is within sqrt(eps()) of the minimum loss $(config.min_loss) at iteration $current_iter")
            return true
        end
        return false 
    end



    if config.time_grads
        optf = Optimization.OptimizationFunction(flexi_loss; grad = counted_grad!)
    else
        optf = Optimization.OptimizationFunction(flexi_loss, Optimization.AutoForwardDiff())
    end

    flexi_ig = collect(ig.flex1_params) 

    prob = Optimization.OptimizationProblem(optf, flexi_ig) 

    opt_alg = build_optimizer(optimizer; learning_rate=learning_rate, linesearch=linesearch)

    
    sol = solve(prob, opt_alg; callback=callback, maxiters=maxiters)
    
    
    println("Final loss: $(sol.objective)")
    println("Best loss found during optimization: $(best_loss[])")

    if config.time_grads
        println("Total gradient evaluations: $(grad_eval_count[])")
    end

    if config.save_parameters && length(loss_history) < maxiters && !isnothing(back_up_optimizer)
        final_grad_norm = LinearAlgebra.norm(gradient_history[end])
        if final_grad_norm > 1e-4
            println("Warning: final gradient norm $(final_grad_norm) exceeds threshold 1e-4")
            println("Attempting to refine solution using backup optimizer: $back_up_optimizer")
            println("loss history length: $(length(loss_history)), best loss: $(best_loss[])")
            # Use the best parameters found so far as the initial guess for the backup optimizer
            opt_swap = true
            backup_start = length(loss_history) + 1  # Start from the iteration where the best loss was found
            backup_prob = Optimization.OptimizationProblem(optf, best_params[])
            backup_alg = build_optimizer(back_up_optimizer; learning_rate=learning_rate, linesearch=linesearch)
            backup_sol = solve(backup_prob, backup_alg; callback=callback, maxiters=maxiters)
            println("Backup optimizer final loss: $(backup_sol.objective)")
            if backup_sol.objective < best_loss[]
                println("Backup optimizer improved the solution. Updating best parameters.")
                best_loss[] = backup_sol.objective
                best_params[] = copy(backup_sol.u)
            else
                println("Backup optimizer did not improve the solution.")
            end
        end
    end
    println("after backup loss history length: $(length(loss_history)), best loss: $(best_loss[])")
  
    time = Base.time() - t0
    println("Gradient descent ($optimizer) time: $time")

    fit_params_repr = ComponentArray(
        p_classical = ig.p_classical,
        flex1_params = best_params[]
    ) # TODO: make this compatible with unmixed models too

    #TODO: add full_parameter_history which stores full ComponenetArray of p_classical and flex1_params
    # p_classical does not change within a single gd so just add p_classical to the history of flex1_params
    full_parameter_history = config.save_parameters ? [ComponentArray(p_classical = ig.p_classical, flex1_params = p) for p in parameter_history] : nothing
    result = (fit_params_repr = fit_params_repr, fit_params_derepr = derepresent_all(fit_params_repr, learning_problem.model),
     loss_history = loss_history, optimizer = optimizer, time = time,
            back_up_optimizer = back_up_optimizer, backup_start = opt_swap ? backup_start : nothing,
            gradient_history = config.save_parameters ? gradient_history : nothing,
            #parameter_history = config.save_parameters ? parameter_history : nothing,
            parameter_history = config.save_parameters ? full_parameter_history : nothing,
            num_grad_evals = config.time_grads ? grad_eval_count[] : nothing,
            grad_time_history = config.time_grads ? grad_time_history : nothing,
            grad_alloc_history = config.time_grads ? grad_alloc_history : nothing)

    # fit_params_derepr = FlexiBasicLearning.derepresent_all(fit_params_repr, learning_problem.model)

    return result # returns p_repr.flex1_params

    
end

# helper builds optimizer
function build_optimizer(name::Symbol; learning_rate=0.01, linesearch=nothing)
    if name == :gradient_descent
        return linesearch === nothing ?
            OptimizationOptimJL.GradientDescent() :
            OptimizationOptimJL.GradientDescent(linesearch=linesearch)
    elseif name == :bfgs
        return linesearch === nothing ?
            OptimizationOptimJL.BFGS() :
            OptimizationOptimJL.BFGS(linesearch=linesearch)
    elseif name == :adam
        return OptimizationOptimisers.Adam(learning_rate)
    else
        error("Unknown optimizer :$name. Supported: :gradient_descent, :bfgs, :adam")
    end
end
