# main reason for making a structure is so that I can have fw(..., model) be diff for each model
struct ModelFlexiAlg <: AbstractFlexiBasicModel
    # u0::Vector{Float64}  # Not used for algebraic model, but kept for compatibility for now...
    # params::ComponentArray{Float64}'
    # params::AbstractVector{Float64} 
    # p_classical_derepresented_ig::ComponentArray{Float64} # 
    # p_derepresented_lowerbounds::ComponentArray{Float64} # lower bounds for derepresented parameters
    # p_derepresented_upperbounds::ComponentArray{Float64} # upper bounds for derepresented parameters
    params_repr_ig::ComponentArray{Float64} # biophysical parameters mapped to spaces suitable for optimization # log, logit, sqrt transforms
    params_derepresented_ig::ComponentArray{Float64}

end

function make_ModelFlexiAlg(;flexi_dofs=5, reltol = 1e-3, abstol = 1e-8) # for call consistency
   
    # params = ComponentArray(
    #     flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    #     # this is in case I want to do multiple flexifunctions
    # )
    
    flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    params_repr_ig = ComponentArray(
        flex1_params = flex1_params
    )
    params_derepresented_ig = ComponentArray(
        flex1_params = deepcopy(flex1_params)
    ) # for compatibility with other models, but not really used for this model
    
    


    return ModelFlexiAlg(
       params_repr_ig,
       params_derepresented_ig
    )
end

function fw(x::AbstractVector, params, model::ModelFlexiAlg; gradient_mode = false)
    return x .* FlexiFunctions.evaluate_decompress.(x, Ref(params); gradient_mode=gradient_mode)
end