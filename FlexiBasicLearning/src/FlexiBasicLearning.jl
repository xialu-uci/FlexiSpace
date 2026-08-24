module FlexiBasicLearning
using Statistics
using Dates
using LinearAlgebra
using ComponentArrays
using Random
using JLD2
using Printf
using Optim, Optimization, OptimizationEvolutionary
using OptimizationOptimJL
using Optimisers, Zygote
using ChainRulesCore # can prob get rid of ChainRulesCore
using SciMLSensitivity
using LineSearches
using FiniteDiff
using ForwardDiff
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
    loss_strategy::String = "RMSE"
end


export LearningProblem
export CallbackConfig, LearningConstants


# include models
include("models/def_flexi_basic.jl")
include("models/def_flexi_alg.jl")
include("models/def_flexi_ode.jl")
include("models/def_flexi_lv.jl")

# include helpers
include("get_loss.jl")


# include learning protocols
include("learning_protocols/base.jl")
include("learning_protocols/cmaes.jl")
include("learning_protocols/grad_desc.jl")

# include functions for fitting and plotting
include("fit_flexi.jl")
include("fit_mult_algs.jl")
include("sim_data.jl") # use naming functions in other files
include("loss_slicing.jl") # for looking at loss vs. params and other stuff.
include("gd_tracking.jl") # for looking at where gd goes


# for my for loops
const func_info = Dict(
    "flexi1"      => (FlexiBasicLearning.make_flexi1_func,      "y = f(t)"),
    "flexi1_alg1" => (FlexiBasicLearning.make_flexi1_alg1_func, "y = t \u22c5 f(t)"),
    "flexi1_ode1" => (FlexiBasicLearning.make_flexi1_ode1_func, "y' = f(y)"),
    "flexi1_lv2" => (FlexiBasicLearning.make_flexi1_lv_func, "y1 = (y1+1)f(y1/(y1+1)) -y1 \u22c5 y2, y2 = α \u22c5 y2(y1-1)")
)

const shapes = Dict(
    "crooked" => FlexiBasicLearning.crooked_flexi,
    "cu"      => FlexiBasicLearning.cu_flexi,
    "cd"      => FlexiBasicLearning.cd_flexi,
)

# maps func_key -> (dof -> make_model closure)
const model_makers = Dict(
    "flexi1"      => d -> () -> FlexiBasicLearning.make_ModelFlexi1(;flexi_dofs=d),
    "flexi1_alg1" => d -> () -> FlexiBasicLearning.make_ModelFlexiAlg(;flexi_dofs=d),  # same structure for now
    "flexi1_ode1" => d -> () -> FlexiBasicLearning.make_ModelFlexiODE(;flexi_dofs=d),
    "flexi1_lv2" => (d, tol) -> () -> FlexiBasicLearning.make_ModelFlexiLV(;flexi_dofs = d, reltol = tol)
)


end # module FlexiBasicLearning
