function get_loss(params; learning_problem::LearningProblem{M}, gradient_mode = false) where {M<:AbstractModel}
    x = learning_problem.data[:,1]
    y = learning_problem.data[:,2]
    # println("params $params")
    y_pred = fw(x, params, learning_problem.model; gradient_mode = gradient_mode)

    if learning_problem.loss_strategy == "vanilla"
        ssr = sum(abs2, y - y_pred) # default p =2 (frobenius)
    end
    return ssr
end

