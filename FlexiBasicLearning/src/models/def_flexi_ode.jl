struct ModelFlexiODE <: AbstractFlexiBasicModel
    # u0::Vector{Float64}  # Not used for algebraic model, but kept for compatibility for now...
    # params::ComponentArray{Float64}
    params_repr_ig::ComponentArray{Float64} # for compatibiility
    params_derepresented_ig::ComponentArray{Float64}
    u0::AbstractVector{Float64}
    reltol::Float64
    abstol::Float64
end

function make_ModelFlexiODE(;flexi_dofs=5, reltol = 1e-3, abstol = 1e-8)
   
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
    
    
    u0 = [0.1]


    return ModelFlexiODE(
       params_repr_ig,
       params_derepresented_ig,
       u0,
       reltol,
       abstol)

end


function make_rhs(model::ModelFlexiODE; gradient_mode = false)
    function rhs(du, u, params, t)
        du .= FlexiFunctions.evaluate_decompress.(u, Ref(params); gradient_mode=gradient_mode)
        return nothing
    end
    return rhs
end

# using FiniteDiff # comment out later
# can probably be shared for ODE models
function fw(x::AbstractVector, params, model::ModelFlexiODE; gradient_mode = false)
    # get ODE solution
    rhs = make_rhs(model; gradient_mode = gradient_mode)
    tspan = (0.0,maximum(x))
    prob = ODEProblem(rhs, model.u0,tspan, params)
    # sol = solve(prob, Tsit5();
    #     saveat = x,
    #     sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())) 
    sol = solve(prob, Tsit5();
        saveat = x, 
        sensealg = ReverseDiffAdjoint()) # could reduce tolerance
    # println("g_fd:$()")
    y = vec(Array(sol))   # states x length(x), then transpose -> length(x) x states
    # println(size())
    return y
end