# Generates jobs.txt: one line per (func_key, dof, shape_key) combo,
# in the exact same order as the original nested for-loop.
# (func_string is 1-1 with func_key, so it doesn't need its own dimension here.)

func_keys  = ["flexi1", "flexi1_alg1", "flexi1_ode1"]
dofs       = [3, 4, 5, 20, 50]
shape_keys = ["crooked", "cu", "cd"]

open("jobs.txt", "w") do io
    for f in func_keys, d in dofs, s in shape_keys
        println(io, "$f $d $s")
    end
end

println("Wrote $(length(func_keys) * length(dofs) * length(shape_keys)) jobs to jobs.txt")