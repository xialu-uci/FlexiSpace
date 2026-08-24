struct ModelFlexiLV <: AbstractFlexiBasicModel
    # u0::Vector{Float64}  # Not used for algebraic model, but kept for compatibility for now...
    # params::ComponentArray{Float64}'
    params::AbstractVector{Float64}
    u0::AbstractVector{Float64}
    reltol::Float64
    abstol::Float64
    
end

function make_ModelFlexiLV(;flexi_dofs=5, reltol = 1e-3, abstol = 1e-8)
   
 
    
    params = FlexiFunctions.generate_flexi_ig(flexi_dofs)
    
    u0 = [1.0, 2.5]


    return ModelFlexiLV(
       params,
       u0,
       reltol,
       abstol)

end


function make_rhs(model::ModelFlexiLV; gradient_mode = false)
    function rhs(du, u, params, t)
        x, y = u
        # println(x)
        x_mod = x/(x+1)
        # du .= FlexiFunctions.evaluate_decompress.(u, Ref(params); gradient_mode=gradient_mode)
        du[1] = (x+1) * FlexiFunctions.evaluate_decompress(x_mod, params; gradient_mode = gradient_mode) - x*y
        du[2] = 2.0*y*(x-1)

        return nothing
    end
    return rhs
end

# using FiniteDiff # comment out later
# can probably be shared for ODE models
function fw(x::AbstractVector, params, model::ModelFlexiLV; gradient_mode = false)
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