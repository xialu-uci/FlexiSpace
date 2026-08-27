struct ModelMixedLV <: AbstractFlexiBasicModel
    # u0::Vector{Float64}  # Not used for algebraic model, but kept for compatibility for now...
    # params::ComponentArray{Float64}'
    p_classical_derepresented_ig::ComponentArray{Float64} # classical parameters
    p_derepresented_lowerbounds::ComponentArray{Float64} # lower bounds for derepresented parameters
    p_derepresented_upperbounds::ComponentArray{Float64} # upper bounds for derepresented parameters
    params_repr_ig::ComponentArray{Float64} # biophysical parameters mapped to spaces suitable for optimization # log, logit, sqrt transforms
    params_derepresented_ig::ComponentArray{Float64}
    # params::AbstractVector{Float64}
    u0::AbstractVector{Float64}
    reltol::Float64
    abstol::Float64
    
end

function make_ModelMixedLV(;flexi_dofs=5, reltol = 1e-3, abstol = 1e-8)
   
    p_classical_derepresented_ig = ComponentArray(
        a = 1.0
    )

    p_derepresented_lowerbounds = ComponentArray(
        a = 5e-2
    )

    p_derepresented_upperbounds = ComponentArray(
        a = 10.0
    )

    
    
    params_derepresented_ig = ComponentArray(
        p_classical=deepcopy(p_classical_derepresented_ig),
        flexi1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    )

    params_repr_ig = ComponentArray(
        p_classical=represent_on_type(p_classical_derepresented_ig, ModelMixedLV),
        # flex
        flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs),
        # flex2_params = FlexiFunctions.generate_flexi_ig(flexi_dofs) 
    )
    
    u0 = [1.0, 2.5]


    return ModelMixedLV(
       p_classical_derepresented_ig,
       p_derepresented_lowerbounds,
       p_derepresented_upperbounds,
       params_derepresented_ig,
       params_repr_ig,
       u0,
       reltol,
       abstol)

end


function make_rhs(model::ModelMixedLV; gradient_mode = false)
    function rhs(du, u, params, t)
        x, y = u
        # println(x)
        x_mod = x/(x+1)
        a = 1.0 #TODO: use p_classical here
        # du .= FlexiFunctions.evaluate_decompress.(u, Ref(params); gradient_mode=gradient_mode)
        du[1] = (x+1) * FlexiFunctions.evaluate_decompress(x_mod, params; gradient_mode = gradient_mode) - x*y
        du[2] = -a*y + x*y

        return nothing
    end
    return rhs
end

# using FiniteDiff # comment out later
# can probably be shared for ODE models
function fw(x::AbstractVector, params, model::ModelMixedLV; gradient_mode = false)
    # get ODE solution
    rhs = make_rhs(model; gradient_mode = gradient_mode)
    tspan = (0.0,maximum(x))
    prob = ODEProblem(rhs, model.u0,tspan, params)
   
    
    sol = solve(prob, Tsit5(); reltol = model.reltol, abstol = model.abstol,
        saveat = x, 
        sensealg = ReverseDiffAdjoint()) # could reduce tolerance

    if sol.retcode != SciMLBase.ReturnCode.Success
        return nothing   # signal failure upstream
    end

    y = permutedims(Array(sol))
    return y
end

function represent_on_type(p_classical_derepresented, model::ModelMixedLV)
    # initial transformations, subject to change
   return  ComponentArray(
        a=log(p_classical_derepresented.a),  # log
   )
end
    


function derepresent(p_classical_represented, model::ModelMixedLV)
    return  ComponentArray(
        a=exp(p_classical_represented.a),  # log
   )
end