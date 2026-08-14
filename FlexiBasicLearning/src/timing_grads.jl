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
expdir = "../FlexiSpaceLocal/exp/08132026/fixed-gts"
datadir_base = "../FlexiSpaceLocal/data/w_true_params/no-noise"

# dofs = [3, 4, 5, 20, 50]
# num_points_list = [3, 5, 10, 20, 50, 100]
# test smaller sweep
# dofs = [3,4]
# num_points_list = [3,5]
gt = 4
dofs = [4, 8, 16, 32, 64, 128, 254, 512]

num_points_list = [2, 4, 8, 16, 32, 64, 128, 254, 512]

skeys = ["crooked", "cu", "cd"]

#shapes = [FlexiBasicLearning.crooked_flexi, FlexiBasicLearning.cu_flexi, FlexiBasicLearning.cd_flexi]
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
grad_time_check = []
for f in funcs, sname in skeys, np in num_points_list
    fname = FlexiBasicLearning.func_name(f)
    s = FlexiBasicLearning.shapes[sname]
    f_str = func_strings[f]

    # data_subfolder = "$(fname)-$(np)obs"
    datafile = joinpath(datadir_base, "$(fname)-$(gt)dof-$(np)obs", "sim_data_$(sname).jld2")

    if !isfile(datafile)
        @warn "Missing datafile, skipping" datafile
        continue
    end
    

for d in dofs

    exp_subfolder = "gt-$(gt)dof/$(fname)-$(d)dof-$(np)obs"
    savedir  = joinpath(expdir, exp_subfolder, sname)
    # datafile = joinpath(datadir_base, subfolder, "sim_data_$(sname).jld2") # TODO: this has to be the same ground truth for all.

    

    make_model = make_model_for(f, d)

    println("Fitting: func=$fname shape=$sname dof=$d num_points=$np. Ground truth is $datafile")
    result = FlexiBasicLearning.fit_all_algs(
        datafile, savedir, make_model;
        maxiters = 10, save_parameters = true, time_grads = true,
    ) # TODO: make it fit only gd for some args
    println("Finished: func=$fname shape=$sname dof=$d num_points=$np")

    # println(result[1])
    grad_times = result[1]["gd_gradient_descent_result"].grad_time_history
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
        grad_time_history = grad_times,   # full per-iteration vector
        n_grad_calls     = result[1]["gd_gradient_descent_result"].num_grad_evals,  # number of times gradient was evaluated
        gd_time          = result[1]["gd_gradient_descent_result"].time,
        # cmaes_time       = result[1]["cmaes_time"],
        save_dir         = savedir,
    ))
end
end
df = DataFrame(rows)

mkpath(expdir)
outfile = joinpath(expdir, "grad_time_results.jld2")
JLD2.save(outfile, "df", df)
println("Saved results table ($(nrow(df)) rows) to $outfile")