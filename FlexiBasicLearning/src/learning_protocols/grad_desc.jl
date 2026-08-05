
using Optimization
using OptimizationOptimJL
using SciMLSensitivity
using LineSearches
using Zygote

function gradient_descent_learn(learning_problem, ig; maxiters=10000, print_frequency = 1000, save_parameters = false, time_grads = false)

    function flexi_loss(params, p)
       
        loss = get_loss(params; learning_problem=learning_problem, gradient_mode=true)
    
       return loss
    end

    grad_eval_count = Ref(0)

    function counted_grad!(G, params, p)
        grad_eval_count[] += 1
        g = Zygote.gradient(θ -> flexi_loss(θ, p), params)[1]
        G .= g
        return nothing
    end

    loss_history = Float64[]
    # grad_norm_history = Float64[]
    
    config = CallbackConfig(;print_frequency = print_frequency, save_parameters = save_parameters, time_grads = time_grads)

    parameter_history = config.save_parameters ? [] : nothing
    gradient_history = config.save_parameters ? [] : nothing
    grad_time_history = config.time_grads ? [] : nothing
    
    
    
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

        if config.time_grads
            # TODO: retrieve number of times that gradient was evaluated (not here but in the optimization function) and store it in config.num_grad_evals
            Zygote.gradient(θ -> flexi_loss(θ, nothing), p.u) # compilation
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

    if config.time_grads
        optf = Optimization.OptimizationFunction(flexi_loss; grad = counted_grad!)
    else
        optf = Optimization.OptimizationFunction(flexi_loss, Optimization.AutoZygote())
    end

    

    prob = Optimization.OptimizationProblem(optf, ig) 
    
    sol = solve(prob, OptimizationOptimJL.GradientDescent(); callback=callback, maxiters=maxiters)
    

    
    println("Final loss: $(sol.objective)")

    if config.time_grads
        println("Total gradient evaluations: $(grad_eval_count[])")
    end

    
    base = (fit_params = sol.u, loss_history = loss_history,
            num_grad_evals = config.time_grads ? grad_eval_count[] : nothing,
            grad_time_history = config.time_grads ? grad_time_history : nothing)

    if config.save_parameters
        result = merge(base, (gradient_history = gradient_history, parameter_history = parameter_history))
    else
        result = base
    end
    return result
    return result
    
end