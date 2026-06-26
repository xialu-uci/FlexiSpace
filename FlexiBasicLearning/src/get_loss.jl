function get_loss(params; learning_problem::LearningProblem{M}) where {M<:AbstractModel}
    x = learning_problem.data[:,1]
    y = learning_problem.data[:,2]
    # println("params $params")
    y_pred = fw(x, params, learning_problem.model)

    if learning_problem.loss_strategy == "vanilla"
        ssr = LinearAlgebra.norm(y - y_pred) # default p =2 (frobenius)
    end
    return ssr
end

