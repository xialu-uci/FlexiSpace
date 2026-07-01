
using Optimization
using OptimizationOptimJL

function gradient_descent_learn(learning_problem, ig; maxiters=10000)

    function flexi_loss(params, p)
        return get_loss(params; learning_problem=learning_problem, gradient_mode=true)
    end

    loss_history = Float64[]
    config = CallbackConfig()

    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
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

    println("Final loss: $(sol.objective)")
    return sol.u, loss_history
end