module FlexiBasicLearning
using Statistics
using Dates
using LinearAlgebra
using ComponentArrays
using Random
using JLD2
using Optim, Optimization, OptimizationEvolutionary
using OptimizationOptimJL
using Optimisers, Zygote
using ChainRulesCore # can prob get rid of ChainRulesCore

include("FlexiFunctions.jl")

using .FlexiFunctions


export LinearAlgebra, ComponentArrays, FlexiFunctions, Random, JLD2


abstract type AbstractModel end # stores model type (for fw and loss) and params
abstract type AbstractFlexiBasicModel <: AbstractModel end  

# include("shared_model_functions.jl")


@kwdef struct LearningProblem{M<:AbstractModel}
    data::Matrix{Float64} # modified to match data structure
    model::M
    mask::Vector{Bool} # modified to match data structure
    loss_strategy::String = "vanilla"
end


export LearningProblem
export CallbackConfig, LearningConstants


# include models
include("models/def_flexi_basic.jl")

# include helpers
include("get_loss.jl")


# include learning protocols
include("learning_protocols/base.jl")
include("learning_protocols/cmaes.jl")
include("learning_protocols/grad_desc.jl")



end # module FlexiBasicLearning
