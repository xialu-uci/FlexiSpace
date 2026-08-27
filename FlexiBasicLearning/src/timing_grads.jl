# 01_run_fits.jl
# Sweeps over function form, dof, shape, and num_points; runs fit_cmaes_and_gd
# for each combo; collects timing info into a DataFrame; saves to JLD2.
#
# max_iters = 10. time_grads = true

using FlexiBasicLearning
using JLD2
using DataFrames
using Statistics
using Zygote
using FiniteDiff
using ForwardDiff

# ------------------------------------------------------------------
# setup
# ------------------------------------------------------------------
expdir_base = "../FlexiSpaceLocal/exp/08262026/fixed-gts"
datadir_base = "../FlexiSpaceLocal/data/w_true_params_flexi_args/no-noise"

gt = 4
num_points_list = [4, 8, 16, 32, 64, 128, 254, 512, 1024, 2048, 4096, 8182, 16364]
dofs = [4, 8, 16, 32, 64, 128, 254, 512, 1024, 2048, 4096, 8182, 16364]

const FIXED_NUM = 32

# build the two slices: (dof varies, num_points=32) and (dof=32, num_points varies)
# dedupe the (32, 32) point so it isn't fit twice
dof_np_combos = unique(vcat(
    [(d, FIXED_NUM) for d in dofs],           # sweep dof, np fixed at 32
    [(FIXED_NUM, np) for np in num_points_list] # sweep np, dof fixed at 32
))

skeys = ["crooked", "cu", "cd"]

differs = [:zygote, :forwarddiff, :finitediff]
diff_name_conv = ["rv", "fw", "fd"]
funcs = [FlexiBasicLearning.make_flexi1_func, FlexiBasicLearning.make_flexi1_alg1_func, FlexiBasicLearning.make_flexi1_ode1_func]

func_strings = Dict(
    FlexiBasicLearning.make_flexi1_func      => "y = f(t)",
    FlexiBasicLearning.make_flexi1_alg1_func => "y = t ⋅ f(t)",
    FlexiBasicLearning.make_flexi1_ode1_func => "y' = f(y)",
)

function make_model_for(f, d)
    if f == FlexiBasicLearning.make_flexi1_func
        return () -> FlexiBasicLearning.make_ModelFlexi1(; flexi_dofs = d)
    elseif f == FlexiBasicLearning.make_flexi1_alg1_func
        return () -> FlexiBasicLearning.make_ModelFlexiAlg(; flexi_dofs = d)
    elseif f == FlexiBasicLearning.make_flexi1_ode1_func
        return () -> FlexiBasicLearning.make_ModelFlexiODE(; flexi_dofs = d)
    else
        error("Unrecognized func: $f")
    end
end

# ------------------------------------------------------------------
# main sweep
# ------------------------------------------------------------------
for (differ, name) in zip(differs, diff_name_conv)
    expdir = joinpath(expdir_base, name)
    rows = NamedTuple[]
    grad_time_check = []
    for f in funcs, sname in skeys, (d, np) in dof_np_combos
        fname = FlexiBasicLearning.func_name(f)
        s = FlexiBasicLearning.shapes[sname]
        f_str = func_strings[f]

        datafile = joinpath(datadir_base, "$(fname)-$(gt)dof-$(np)obs", "sim_data_$(sname).jld2")

        if !isfile(datafile)
            @warn "Missing datafile, skipping" datafile
            continue
        end

        exp_subfolder = "gt-$(gt)dof/$(fname)-$(d)dof-$(np)obs"
        savedir  = joinpath(expdir, exp_subfolder, sname)

        make_model = make_model_for(f, d)

        println("Fitting: func=$fname shape=$sname dof=$d num_points=$np. Ground truth is $datafile")
        result = FlexiBasicLearning.fit_all_algs(
            datafile, savedir, make_model;  differ = differ,
            maxiters = 10, save_parameters = true, time_grads = true,
        )
        println("Finished: func=$fname shape=$sname dof=$d num_points=$np")

        grad_times = result[1]["gd_gradient_descent_result"].grad_time_history
        grad_allocs = result[1]["gd_gradient_descent_result"].grad_alloc_history
        push!(grad_time_check, grad_times)
        push!(rows, (
            func_form        = fname,
            func_string      = f_str,
            shape            = sname,
            dof              = d,
            num_points       = np,
            mean_grad_time   = mean(grad_times),
            median_grad_time = median(grad_times),
            total_grad_time  = sum(grad_times),
            mean_grad_alloc   = mean(grad_allocs),
            median_grad_alloc = median(grad_allocs),
            total_grad_alloc  = sum(grad_allocs),
            grad_time_history = grad_times,
            n_grad_calls     = result[1]["gd_gradient_descent_result"].num_grad_evals,
            gd_time          = result[1]["gd_gradient_descent_result"].time,
            save_dir         = savedir,
        ))
    end
    df = DataFrame(rows)

    mkpath(expdir)
    outfile = joinpath(expdir, "$(name)_grad_time_results.jld2")
    JLD2.save(outfile, "df", df)
    println("Saved results table ($(nrow(df)) rows) to $outfile")
end