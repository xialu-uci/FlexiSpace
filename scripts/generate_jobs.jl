# Generates jobs.txt: one line per (func_key, dof, shape_key) combo,
# in the exact same order as the original nested for-loop.
# (func_string is 1-1 with func_key, so it doesn't need its own dimension here.)

#func_keys  = ["flexi1", "flexi1_alg1", "flexi1_ode1"]
dofs = [4, 5, 8, 16, 32, 64]
optimizers = ["cmaes", "bfgs"]
# opt_strings = ["cmaes"]
# shape_keys = ["crooked", "cu", "cd"]

open("landscape_jobs.txt", "w") do io
    for d in dofs, opt in optimizers
        println(io, "$d $opt")
    end
end

println("Wrote $(length(dofs) * length(optimizers)) jobs to landscape_jobs.txt")