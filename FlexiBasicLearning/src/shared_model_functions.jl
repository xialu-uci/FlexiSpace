# abstract type AbstractModel end

abstract type AbstractFlexiBasicModel <: AbstractModel end  

# struct LearningProblem{M<:AbstractModel}
#     data::Matrix{Float64} # modified to match data structure
#     model::M
#     mask::Vector{Bool} # modified to match data structure
#     loss_strategy::String = "vanilla"
# end

