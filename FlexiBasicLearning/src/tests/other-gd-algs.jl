# for a 1 model types (try simple one): do bfgs, gd, adam, cmaes. 
    # want to see that: (1) each gd alg runs, (2) loss history is decreasing.

using JLD2
using FlexiBasicLearning
using CairoMakie
using LinearAlgebra

# just edited remote to have "main" as default branch. Edit to check if push/pulling to intended places.
# ------------------------------------------------------------------
# Static config (shared across all jobs)
# ------------------------------------------------------------------
num_points = 20
expdir  = "../FlexiSpaceLocal/tests/0807026/other-algs-test-1"
datadir = "../FlexiSpaceLocal/data/w_true_params/no-noise"

# ------------------------------------------------------------------
# CLI args: func_key  dof  shape_key
# ------------------------------------------------------------------
# function usage_and_exit()
#     println(stderr, """
#     Usage: julia fit_single.jl <func_key> <dof> <shape_key>

#       func_key   one of: $(join(sort(collect(keys(func_info))), ", "))
#       dof        integer, e.g. 3, 4, 5, 20, 50
#       shape_key  one of: $(join(sort(collect(keys(shapes))), ", "))

#     Example:
#       julia fit_single.jl flexi1 20 crooked
#     """)
#     exit(1)
# end

# length(ARGS) < 3 && usage_and_exit()

# # uncomment for HPC
# func_key  = ARGS[1]
# d         = parse(Int, ARGS[2])
# shape_key = ARGS[3]


#TODO: Test if refactoring still works for fitting.func_key  = ARGS[1]
# uncomment for local testing
func_key  = "flexi1_ode1"
d         = 3
shape_key = "crooked"


haskey(FlexiBasicLearning.func_info, func_key) || error("Unknown func_key '$func_key'. Options: $(collect(keys(FlexiBasicLearning.func_info)))")
haskey(FlexiBasicLearning.shapes, shape_key)   || error("Unknown shape_key '$shape_key'. Options: $(collect(keys(FlexiBasicLearning.shapes)))")

f, f_str = FlexiBasicLearning.func_info[func_key]
s        = FlexiBasicLearning.shapes[shape_key]

# ------------------------------------------------------------------
# Reconstruct paths / model 
# ------------------------------------------------------------------
fname = FlexiBasicLearning.func_name(f)
sname = FlexiBasicLearning.shape_name(s)

subfolder = "$(fname)-$(d)dof-$(num_points)obs"
savedir   = joinpath(expdir, subfolder, sname)
datafile  = joinpath(datadir, subfolder, "sim_data_$(sname).jld2")

make_model = FlexiBasicLearning.model_makers[func_key](d)

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
println("Fitting for datafile: $datafile")
results = FlexiBasicLearning.fit_all_algs(datafile, savedir, make_model; optimizers = [:gradient_descent, :bfgs, :adam], save_parameters = true)
println("Finished fitting for datafile: $datafile")

# uncomment for local testing
FlexiBasicLearning.make_fitting_figs(results)
FlexiBasicLearning.end_to_end_gd_tracking(results; func_form = f, func_string = f_str)

# for 1 model types (try difficult ones): 
    # do bfgs, gd, adam, cmaes. (save_parmameters = true)
        # this is more like a preliminary result than a test.
        # hope to see: (1) new gd algs are speedier than vanilla gd. do they perform on par with cmaes? does the gradient seem near zero at the end of optimization?
   
# for all model types: 
    # do bfgs, gd, adam (low maxiter, time_grads = true, save_parameters = true)
    # this is more like a preliminary result than a test.
    # hope to see: (1) new gd algs are speedier than vanilla gd. how do the number of gradient eval calls compare across gd algs?


# function fit_all_algs() <-- should be similar to fit_cmaes_and_gd, but takes a list of gd algs and runs them all, saving results in a single file. 

