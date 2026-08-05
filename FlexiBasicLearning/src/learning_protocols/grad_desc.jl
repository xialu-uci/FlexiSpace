
using Optimization
using OptimizationOptimJL
using SciMLSensitivity
using LineSearches
using Zygote

function gradient_descent_learn(learning_problem, ig; maxiters=10000, print_frequency = 1000, save_parameters = false, time_grads = false)

    function flexi_loss(params, p)
       #println("USING TRY/CATCH VERSION")
       #loss = try
        loss = get_loss(params; learning_problem=learning_problem, gradient_mode=true)
    #    catch e
    #        println("CAUGHT: $(typeof(e))")
    #        if e isa InexactError || e isa DomainError
    #            return 1e12
    #        else
    #            rethrow(e)
    #        end
    #    end
       return loss
   end

    loss_history = Float64[]
    # grad_norm_history = Float64[]
    
    config = CallbackConfig(;print_frequency = print_frequency, save_parameters = save_parameters, time_grads = time_grads)

    parameter_history = config.save_parameters ? [] : nothing
    gradient_history = config.save_parameters ? [] : nothing
    grad_time_history = config.time_grads ? [] : nothing

    
    
    # TODO: modify to use config
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        # println("callback iter $current_iter: loss=$lossval, norm(u)=$(sqrt(sum(p.u.^2))), norm(grad)=$(sqrt(sum(p.grad.^2)))")
        # in callback: compute gradient again? time it and print it here.

        if config.save_parameters
            # if current_iter % config.print_frequency == 0
            push!(gradient_history, copy(p.grad))
            push!(parameter_history, copy(p.u))
                # push!(save_its, )
            # end   
        end

        if config.time_grads
            t0 = time_ns()
            Zygote.gradient(θ -> flexi_loss(θ, nothing), p.u)
            elapsed = (time_ns() - t0) / 1e9
            push!(grad_time_history, elapsed)
        end


        if config.verbose && current_iter % config.print_frequency == 0 
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In grad_descent, iteration $current_iter: loss=$lossval, qdrms=$qdrms at $(now())")
            flush(stdout)
        end
        return false
    end

    optf = Optimization.OptimizationFunction(flexi_loss, Optimization.AutoZygote())

    # flexi_bound = 1.0 # for cu, true max is always dof/srt(sum(1 to dof)(x^2)), but initial guess is 1/sqrt(dof)

    # lb = 0.0*fill(flexi_bound, length(ig));#-1.0*fill(flexi_bound, length(ig))
    # ub = +1.0*fill(flexi_bound, length(ig)) # how kosher is it for me to do this teehee

    prob = Optimization.OptimizationProblem(optf, ig) 
    # sol = solve(prob, OptimizationOptimJL.GradientDescent(linesearch = LineSearches.BackTracking(),
    #                 alphaguess = LineSearches.InitialStatic(alpha = 1e-2)); callback=callback, maxiters=maxiters)
    if config.time_grads
        Zygote.gradient(θ -> flexi_loss(θ, nothing), ig)  # this compiles it
    end
    sol = solve(prob, OptimizationOptimJL.GradientDescent(); callback=callback, maxiters=maxiters)
    # # could try other gradient descent optimizers (BFGS)

    # function flexi_grad!(G, params, p)
    #     g = Zygote.gradient(θ -> flexi_loss(θ, p), params)[1]
    #     gnorm = sqrt(sum(g.^2))
    #     max_norm = 1e3
    #     if gnorm > max_norm
    #         g = g .* (max_norm / gnorm)
    #     end
    #     G .= g
    #     return nothing
    # end

    # optf = Optimization.OptimizationFunction(flexi_loss, Optimization.AutoZygote(); grad = flexi_grad!)
    # prob = Optimization.OptimizationProblem(optf, ig)
    # sol = solve(prob, OptimizationOptimJL.GradientDescent(); callback=callback, maxiters=maxiters)

    println("Final loss: $(sol.objective)")

    # TODO: modify to be a result with fields
    if config.save_parameters
        result = (fit_params = sol.u, loss_history = loss_history, gradient_history = gradient_history,
                  parameter_history = parameter_history,
                  grad_time_history = config.time_grads ? grad_time_history : nothing)
    else
        result = (fit_params = sol.u, loss_history = loss_history,
                  grad_time_history = config.time_grads ? grad_time_history : nothing)
    end
    return result
    
end