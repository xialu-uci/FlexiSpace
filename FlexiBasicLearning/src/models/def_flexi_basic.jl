
# main reason for making a structure is so that I can have fw(..., model) be diff for each model
struct ModelFlexi1 <: AbstractFlexiBasicModel
    # u0::Vector{Float64}  # Not used for algebraic model, but kept for compatibility for now...
    # params::ComponentArray{Float64}'
    # params::AbstractVector{Float64} 
    params_repr_ig::ComponentArray{Float64} # biophysical parameters mapped to spaces suitable for optimization # log, logit, sqrt transforms
    params_derepresented_ig::ComponentArray{Float64}

end

function make_ModelFlexi1(;flexi_dofs=5, reltol = 1e-3, abstol = 1e-8) # for call consistency
   
    # params = ComponentArray(
    #     flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    #     # this is in case I want to do multiple flexifunctions
    # )
    
    # params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    
    flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    params_repr_ig = ComponentArray(
        p_classical = nothing,
        flex1_params = flex1_params
    )
    params_derepresented_ig = ComponentArray(
        p_classical = nothing,
        flex1_params = deepcopy(flex1_params)
    ) # for compatibility with other models, but not really used for this model
    
    


    return ModelFlexi1(
       params_repr_ig,
       params_derepresented_ig
    )
end

function fw(x::AbstractVector, params, model::ModelFlexi1; gradient_mode = false)
    return FlexiFunctions.evaluate_decompress.(x, Ref(params.flex1_params); gradient_mode=gradient_mode)
end

