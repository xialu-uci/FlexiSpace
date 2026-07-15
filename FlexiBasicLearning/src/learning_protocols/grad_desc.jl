
using Optimization
using OptimizationOptimJL

function gradient_descent_learn(learning_problem, ig; maxiters=10000, print_frequency = 100, save_parameters = false)

    function flexi_loss(params, p)
        return get_loss(params; learning_problem=learning_problem, gradient_mode=true)
    end

    loss_history = Float64[]
    # grad_norm_history = Float64[]
    if slow
        grads = []
        params = []
        # save_its = []
    end
    
    config = CallbackConfig(;print_frequency = print_frequency, save_parameters = save_parameters)
    
    # TODO: modify to use config
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        parameter_history = config.save_parameters ? [] : nothing
        gradient_history = config.save_parameters ? [] : nothing

        if config.save_parameters
            if current_iter % config.print_frequency == 0
                push!(grads, p.grad)
                push!(params, p.u)
                # push!(save_its, )
            end   
        end

        if config.verbose && current_iter % config.print_frequency == 0
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In grad_descent, iteration $current_iter: loss=$lossval, qdrms=$qdrms at $(now())")
            flush(stdout)
        end
        return false
    end

    optf = Optimization.OptimizationFunction(flexi_loss, Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(optf, ig)
    sol = solve(prob, OptimizationOptimJL.GradientDescent(); callback=callback, maxiters=maxiters)
    # could try other gradient descent optimizers (BFGS)

    println("Final loss: $(sol.objective)")

    # TODO: modify to be a result with fields
    if slow
        # return sol.u, loss_history, grad_norm_history, grads, params, config.print_frequency
        result =  (fit_params = sol.u, loss_history = loss_history, grads = grads, grads = params, save_it = config.print_frequency)
    else
        result = (fit_params = sol.u, loss_history = loss_history)
    end
    return result
    
end