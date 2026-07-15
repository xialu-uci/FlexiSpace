
using Optimization
using OptimizationOptimJL

function gradient_descent_learn(learning_problem, ig; maxiters=10000, slow = false)

    function flexi_loss(params, p)
        return get_loss(params; learning_problem=learning_problem, gradient_mode=true)
    end

    loss_history = Float64[]
    grad_norm_history = Float64[]
    if slow
        grads = []
        params = []
    end
    config = CallbackConfig()
    
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)

        if hasproperty(p, :grad) && !isnothing(p.grad)
            push!(grad_norm_history, norm(p.grad))
            if slow && current_iter % config.print_frequency == 0
                push!(grads, p.grad)
                push!(params, p.u)
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

    if slow
        return sol.u, loss_history, grad_norm_history, grads, params
    end
    return sol.u, loss_history, grad_norm_history
    
end