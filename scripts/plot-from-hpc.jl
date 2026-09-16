using JLD2
using FlexiBasicLearning
using CairoMakie
using LinearAlgebra

# ------------------------------------------------------------------
# Static config -- must match fit_single.jl exactly
# ------------------------------------------------------------------
# const num_points = 20
num_points = 20
expdir  = "../FlexiSpaceLocal/tests/08102025-apple/"
datadir = "../FlexiSpaceLocal/data/w_true_params/no-noise"

# func_info = Dict(
#     "flexi1"      => (FlexiBasicLearning.make_flexi1_func,      "y = f(t)"),
#     "flexi1_alg1" => (FlexiBasicLearning.make_flexi1_alg1_func, "y = t \u22c5 f(t)"),
#     "flexi1_ode1" => (FlexiBasicLearning.make_flexi1_ode1_func, "y' = f(y)"),
# )

# shapes = Dict(
#     "crooked" => FlexiBasicLearning.crooked_flexi,
#     "cu"      => FlexiBasicLearning.cu_flexi,
#     "cd"      => FlexiBasicLearning.cd_flexi,
# )

dofs = [3, 4, 5, 20, 50]

# dofs = [3]

# ------------------------------------------------------------------
# Walk the same grid fit_single.jl was run over
# ------------------------------------------------------------------
n_ok = 0
n_missing = 0
n_failed = 0

for func_key in keys(FlexiBasicLearning.func_info), d in dofs, shape_key in keys(FlexiBasicLearning.shapes)
    f, f_str = FlexiBasicLearning.func_info[func_key]
    s = FlexiBasicLearning.shapes[shape_key]

    fname = FlexiBasicLearning.func_name(f)
    sname = FlexiBasicLearning.shape_name(s)

    subfolder = "$(fname)-$(d)dof-$(num_points)obs"
    savedir   = joinpath(expdir, subfolder, sname)
    datafile  = joinpath(datadir, subfolder, "sim_data_$(sname).jld2")

    results_file = joinpath(savedir, "results_all_ig.jld2")

    if !isfile(results_file)
        println("  [skip] no results found at $results_file")
        global n_missing += 1
        continue
    end

    println("Plotting: func=$func_key dof=$d shape=$shape_key")
    try
        @load results_file results
        # println(results)
        FlexiBasicLearning.make_fitting_figs(results) #TODO: Modify plotting calls, test if new structure works for plotting
        FlexiBasicLearning.end_to_end_gd_tracking(results; func_form = f, func_string = f_str)
        global n_ok += 1
    catch e
        println("  [error] failed on $savedir: $e")
        global n_failed += 1
    end
end

println("\nDone. ok=$n_ok  missing=$n_missing  failed=$n_failed")