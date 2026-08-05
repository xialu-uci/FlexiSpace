# 01_run_fits.jl
# Sweeps over function form, dof, shape, and num_points; runs fit_cmaes_and_gd
# for each combo; collects timing info into a DataFrame; saves to JLD2.
#
# max_iters = 10. time_grads = true

using FlexiBasicLearning
using JLD2
using DataFrames
using Statistics

# ------------------------------------------------------------------
# setup
# ------------------------------------------------------------------
expdir = "../FlexiSpaceLocal/tests/08052026/added-grad-comp-counter"
datadir_base = "../FlexiSpaceLocal/data/w_true_params/no-noise"

# dofs = [3, 4, 5, 20, 50]
# num_points_list = [3, 5, 10, 20, 50, 100]
# test smaller sweep
# dofs = [3,4]
# num_points_list = [3,5]
shapes = [FlexiBasicLearning.crooked_flexi, FlexiBasicLearning.cu_flexi, FlexiBasicLearning.cd_flexi]
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
rows = NamedTuple[]
gt_check = []
for f in funcs, d in dofs, s in shapes, np in num_points_list
    fname = FlexiBasicLearning.func_name(f)
    sname = FlexiBasicLearning.shape_name(s)
    f_str = func_strings[f]

    subfolder = "$(fname)-$(d)dof-$(np)obs"
    savedir  = joinpath(expdir, subfolder, sname)
    datafile = joinpath(datadir_base, subfolder, "sim_data_$(sname).jld2")

    if !isfile(datafile)
        @warn "Missing datafile, skipping" datafile
        continue
    end

    make_model = make_model_for(f, d)

    println("Fitting: func=$fname shape=$sname dof=$d num_points=$np")
    result = FlexiBasicLearning.fit_cmaes_and_gd(
        datafile, savedir, make_model;
        maxiters = 10, save_parameters = true, time_grads = true,
    ) # TODO: make it fit only gd for some args
    println("Finished: func=$fname shape=$sname dof=$d num_points=$np")

    println(result[1])
    grad_times = result[1]["gd_result"].grad_time_history
    push!(gt_check, grad_times)
    push!(rows, (
        func_form        = fname,
        func_string      = f_str,
        shape            = sname,
        dof              = d,
        num_points       = np,
        mean_grad_time   = mean(grad_times),
        median_grad_time = median(grad_times),
        total_grad_time  = sum(grad_times),
        grad_time_history = grad_times,   # full per-iteration vector
        n_grad_calls     = result[1]["gd_result"].num_grad_evals,  # number of times gradient was evaluated
        gd_time          = result[1]["gd_time"],
        # cmaes_time       = result[1]["cmaes_time"],
        save_dir         = savedir,
    ))
end

df = DataFrame(rows)

mkpath(expdir)
outfile = joinpath(expdir, "grad_time_results.jld2")
JLD2.save(outfile, "df", df)
println("Saved results table ($(nrow(df)) rows) to $outfile")