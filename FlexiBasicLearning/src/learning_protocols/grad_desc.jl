
using Optimization
using OptimizationOptimJL
using SciMLSensitivity

function gradient_descent_learn(learning_problem, ig; maxiters=10000, print_frequency = 1000, save_parameters = false)

    function flexi_loss(params, p)
        return get_loss(params; learning_problem=learning_problem, gradient_mode=true)
    end

    loss_history = Float64[]
    # grad_norm_history = Float64[]
    
    config = CallbackConfig(;print_frequency = print_frequency, save_parameters = save_parameters)

    parameter_history = config.save_parameters ? [] : nothing
    gradient_history = config.save_parameters ? [] : nothing

    
    
    # TODO: modify to use config
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        

        if config.save_parameters
            # if current_iter % config.print_frequency == 0
            push!(gradient_history, copy(p.grad))
            push!(parameter_history, copy(p.u))
                # push!(save_its, )
            # end   
        end

        if config.verbose && current_iter % config.print_frequency == 0
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In grad_descent, iteration $current_iter: loss=$lossval, qdrms=$qdrms at $(now())")
            flush(stdout)
        end
        return false
    end

    optf = Optimization.OptimizationFunction(flexi_loss, Optimization.AutoZygote())

    flexi_bound = 1.0 # for cu, true max is always dof/srt(sum(1 to dof)(x^2)), but initial guess is 1/sqrt(dof)

    lb = 0.0*fill(flexi_bound, length(ig));#-1.0*fill(flexi_bound, length(ig))
    ub = +1.0*fill(flexi_bound, length(ig)) # how kosher is it for me to do this teehee

    prob = Optimization.OptimizationProblem(optf, ig; lb=lb, ub = ub) # TODO: add ub, lb?
    sol = solve(prob, OptimizationOptimJL.GradientDescent(); callback=callback, maxiters=maxiters)
    # could try other gradient descent optimizers (BFGS)

    println("Final loss: $(sol.objective)")

    # TODO: modify to be a result with fields
    if config.save_parameters
        # return sol.u, loss_history, grad_norm_history, grads, params, config.print_frequency
        result =  (fit_params = sol.u, loss_history = loss_history, gradient_history = gradient_history, parameter_history = parameter_history) # save_freq = config.print_frequency
    else
        result = (fit_params = sol.u, loss_history = loss_history)
    end
    return result
    
end