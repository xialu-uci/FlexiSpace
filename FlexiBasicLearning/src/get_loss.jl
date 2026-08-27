function get_loss(params; learning_problem::LearningProblem{M}, gradient_mode = false) where {M<:AbstractModel}
    x = learning_problem.data[:,1]
    y = learning_problem.data[:,2:end]


    # println("y: $y")
    # println(size(y))
    # println("params $params")
    # println("is y_pred >1")
    y_pred = fw(x, params, learning_problem.model; gradient_mode = gradient_mode)
    # println(y_pred === nothing)
    # println(y_pred === nothing ? "nothing" : size(y_pred))
    # println(size(learning_problem.data[:,2:end]))

    if y_pred === nothing
        println("warn fw returned nothing")
        return 1e6 + sum(abs2, params)
    end

    y_pred_vec = vec(y_pred)
    y_vec = vec(y)

    if length(y_pred_vec) != length(y_vec)
        println("warn size mismatch")
        return 1e6 + sum(abs2, params)
    end
    # println("pred: $y_pred")
    # println(size(y_pred))
    # println(maximum(y_pred))

    ssr = sum(abs2, y - y_pred) # default p =2 (frobenius)
    
    # make a map of what I need to backprop through for grad computation to figure out where to put the timer

    if learning_problem.loss_strategy == "vanilla"
        return ssr # = sum(abs2, y - y_pred) default p =2 (frobenius)
    elseif learning_problem.loss_strategy == "normalized"
        return ssr / length(y)
    elseif learning_problem.loss_strategy == "RMSE"
        return sqrt(ssr / length(y))
    end
    return ssr
end

