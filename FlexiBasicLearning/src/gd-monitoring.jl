# what plots are useful to make for monitoring gd slow?
# already checking the grad norms in not slow
#    for ndofs = 2 could just plot loss vs. params?
function scan_slow_gd(result, gt)
    norms = LinearAlgebra.norm.(result.gradient_history)
    dots = dot.(result.gradient_history, Ref(gt))
    dists = dist.(result.parameter_history, Ref(gt))
    gd_tracker = (norms=norms, dots = dots, dists = dists)
    return gd_tracker
end

# sol.u, loss_history, grad_norm_history, grads, params