# 00_generate_tasks.jl
using FlexiBasicLearning

gt = 4
num_points_list = [4, 8, 16, 32, 64, 128, 254, 512, 1024, 2048, 4096, 8182, 16364]
dofs = [4, 8, 16, 32, 64, 128, 254, 512, 1024, 2048, 4096, 8182, 16364]
const FIXED_NUM = 32

dof_np_combos = unique(vcat(
    [(d, FIXED_NUM) for d in dofs],
    [(FIXED_NUM, np) for np in num_points_list]
))

skeys = ["crooked", "cu", "cd"]
diff_name_conv = ["rv", "fw", "fd"]   # zygote, forwarddiff, finitediff
func_names = ["flexi1", "flexi1alg1", "flexi1ode1"]  

datadir_base = "../FlexiSpaceLocal/data/w_true_params_flexi_args/no-noise"

n_written = 0
n_skipped = 0
skipped_files = String[]

open("./scripts/tasks.txt", "w") do io
    for diffname in diff_name_conv, fname in func_names, sname in skeys, (d, np) in dof_np_combos
        datafile = joinpath(datadir_base, "$(fname)-$(gt)dof-$(np)obs", "sim_data_$(sname).jld2")
        if !isfile(datafile)
            @warn "Missing datafile, skipping" datafile
            n_skipped += 1
            push!(skipped_files, datafile)
            continue
        end
        println(io, "$diffname $fname $sname $d $np")
        global n_written
        n_written += 1
    end
end

n = countlines("./scripts/tasks.txt")
println("Wrote $n tasks to tasks.txt")
println("Skipped $n_skipped combos due to missing datafiles")

# unique missing files (same datafile gets checked once per diffname, so dedupe for a clean summary)
unique_missing = unique(skipped_files)
println("$(length(unique_missing)) unique datafiles missing:")
for f in unique_missing
    println("  $f")
end