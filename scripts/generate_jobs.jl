# Generates jobs.txt: one line per (func_key, dof, shape_key) combo,
# in the exact same order as the original nested for-loop.
# (func_string is 1-1 with func_key, so it doesn't need its own dimension here.)

#func_keys  = ["flexi1", "flexi1_alg1", "flexi1_ode1"]
diffnames = ["rv", "fw","fd"]
fnames = ["flexi1", "flexi1alg1", "flexi1ode1"]
snames = ["crooked", "cu", "cd"]
num_points = [4, 8, 16, 32, 64, 128, 254, 512, 1024, 2048]
dofs = [4, 8, 16, 32, 64, 128, 254, 512, 1024, 2048]
# dofs = [4, 5, 8, 16, 32, 64]
optimizers = ["cmaes", "bfgs"]
# opt_strings = ["cmaes"]
# shape_keys = ["crooked", "cu", "cd"]

open("timing_grads_tasks.txt", "w") do io
    for diffname in diffnames, fname in fnames, sname in snames, np in num_points
        println(io, "$diffname $fname $sname 32 $np")
    end
    for diffname in diffnames, fname in fnames, sname in snames, d in dofs
        println(io, "$diffname $fname $sname $d 32")
    end
end

println("Wrote $(length(diffnames) * length(fnames) * length(snames) * ( length(dofs) + length(num_points))) jobs to timing_grads_tasks.txt")