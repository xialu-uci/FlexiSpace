struct ModelFlexiODE <: AbstractFlexiBasicModel
    # u0::Vector{Float64}  # Not used for algebraic model, but kept for compatibility for now...
    # params::ComponentArray{Float64}'
    params::AbstractVector{Float64}
    u0::AbstractVector{Float64}
    
end

function make_ModelFlexiODE(;flexi_dofs=5)
   
    # params = ComponentArray(
    #     flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    #     # this is in case I want to do multiple flexifunctions
    # )
    
    params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    
    u0 = [0.1]


    return ModelFlexiODE(
       params,
       u0)

end


function make_rhs(model::ModelFlexiODE; gradient_mode = false)
    function rhs(du, u, params, t)
        du .= FlexiFunctions.evaluate_decompress.(u, Ref(params); gradient_mode=gradient_mode)
        return nothing
    end
    return rhs
end

# can probably be shared for ODE models
function fw(x::AbstractVector, params, model::ModelFlexiODE; gradient_mode = false)
    # get ODE solution
    rhs = make_rhs(model; gradient_mode = gradient_mode)
    tspan = (0.0,1.0)
    prob = ODEProblem(rhs, model.u0,tspan, params)
    sol = solve(prob, Tsit5();
        saveat = x,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))

    y = vec(Array(sol))   # states x length(x), then transpose -> length(x) x states
    return y
end