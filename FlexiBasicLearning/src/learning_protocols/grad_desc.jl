
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using SciMLSensitivity
using LineSearches
using Zygote
using FlexiBasicLearning

function gradient_descent_learn(learning_problem, ig; 
    optimizer = :gradient_descent, 
    differ = Zygote.gradient, # also try ForwardDiff.gradient, FiniteDiff.grad, FiniteDiff.finite_difference_gradient
    learning_rate=0.01, # for adam
    linesearch=nothing, # for gd or bfgs
    maxiters=10000, 
    print_frequency = 1000, 
    save_parameters = false, 
    time_grads = false)

    # timing
    t0 = Base.time()

    function flexi_loss(params, p)
       
        loss = get_loss(params; learning_problem=learning_problem, gradient_mode=true)
    
       return loss
    end

    grad_eval_count = Ref(0)

    n = length(ig)
    # build once — outside the loop — so it's non-allocating on every call
    fd_cache = FiniteDiff.GradientCache(zeros(n), zeros(n))

    # a single unified gradient! function, chosen by `differ`
    grad_fn! = if differ == :zygote
        (G, params, p) -> begin
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
            grad_fn!(θ -> flexi_loss(θ, nothing), p.u) # compilation
            t0 = time_ns()
            grad_fn!(θ -> flexi_loss(θ, nothing), p.u)
            elapsed = (time_ns() - t0) / 1e9
            push!(grad_time_history, elapsed)
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
        optf = Optimization.OptimizationFunction(flexi_loss, Optimization.AutoZygote())
    end

    

    prob = Optimization.OptimizationProblem(optf, ig) 

    opt_alg = build_optimizer(optimizer; learning_rate=learning_rate, linesearch=linesearch)

    
    sol = solve(prob, opt_alg; callback=callback, maxiters=maxiters)
    
    
    println("Final loss: $(sol.objective)")

    if config.time_grads
        println("Total gradient evaluations: $(grad_eval_count[])")
    end

    time = Base.time() - t0
    println("Gradient descent ($optimizer) time: $time")
    result = (fit_params = sol.u, loss_history = loss_history, optimizer = optimizer, time = time,
            gradient_history = config.save_parameters ? gradient_history : nothing,
            parameter_history = config.save_parameters ? parameter_history : nothing,
            num_grad_evals = config.time_grads ? grad_eval_count[] : nothing,
            grad_time_history = config.time_grads ? grad_time_history : nothing)



    return result
    
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