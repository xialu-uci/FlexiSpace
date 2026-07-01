# main reason for making a structure is so that I can have fw(..., model) be diff for each model
struct ModelFlexiAlg <: AbstractFlexiBasicModel
    # u0::Vector{Float64}  # Not used for algebraic model, but kept for compatibility for now...
    # params::ComponentArray{Float64}'
    params::AbstractVector{Float64} 
    
end

function make_ModelFlexiAlg(;flexi_dofs=5)
   
    # params = ComponentArray(
    #     flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    #     # this is in case I want to do multiple flexifunctions
    # )
    
    params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    
    


    return ModelFlexiAlg(
       params
    )
end

function fw(x::AbstractVector, params, model::ModelFlexiAlg; gradient_mode = false)
    return x .* FlexiFunctions.evaluate_decompress.(x, Ref(params); gradient_mode=gradient_mode)
end