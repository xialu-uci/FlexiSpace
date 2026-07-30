using JLD2
using FlexiBasicLearning
using CairoMakie
using LinearAlgebra

# ------------------------------------------------------------------
# Static config (shared across all jobs)
# ------------------------------------------------------------------
const num_points = 20
const expdir  = "../FlexiSpaceLocal/exp/07292026/"
const datadir = "../FlexiSpaceLocal/data/w_true_params/no-noise"

# func_key -> (func, func_string) -- 1-1 correspondence, func used for file naming,
# func_string used for plot titles.
const func_info = Dict(
    "flexi1"      => (FlexiBasicLearning.make_flexi1_func,      "y = f(t)"),
    "flexi1_alg1" => (FlexiBasicLearning.make_flexi1_alg1_func, "y = t \u22c5 f(t)"),
    "flexi1_ode1" => (FlexiBasicLearning.make_flexi1_ode1_func, "y' = f(y)"),
)

const shapes = Dict(
    "crooked" => FlexiBasicLearning.crooked_flexi,
    "cu"      => FlexiBasicLearning.cu_flexi,
    "cd"      => FlexiBasicLearning.cd_flexi,
)

# maps func_key -> (dof -> make_model closure)
const model_makers = Dict(
    "flexi1"      => d -> () -> FlexiBasicLearning.make_ModelFlexi1(;flexi_dofs=d),
    "flexi1_alg1" => d -> () -> FlexiBasicLearning.make_ModelFlexiAlg(;flexi_dofs=d),  # same structure for now
    "flexi1_ode1" => d -> () -> FlexiBasicLearning.make_ModelFlexiODE(;flexi_dofs=d),
)

# ------------------------------------------------------------------
# CLI args: func_key  dof  shape_key
# ------------------------------------------------------------------
function usage_and_exit()
    println(stderr, """
    Usage: julia fit_single.jl <func_key> <dof> <shape_key>

      func_key   one of: $(join(sort(collect(keys(func_info))), ", "))
      dof        integer, e.g. 3, 4, 5, 20, 50
      shape_key  one of: $(join(sort(collect(keys(shapes))), ", "))

    Example:
      julia fit_single.jl flexi1 20 crooked
    """)
    exit(1)
end

length(ARGS) < 3 && usage_and_exit()

func_key  = ARGS[1]
d         = parse(Int, ARGS[2])
shape_key = ARGS[3]

haskey(func_info, func_key) || error("Unknown func_key '$func_key'. Options: $(collect(keys(func_info)))")
haskey(shapes, shape_key)   || error("Unknown shape_key '$shape_key'. Options: $(collect(keys(shapes)))")

f, f_str = func_info[func_key]
s        = shapes[shape_key]

# ------------------------------------------------------------------
# Reconstruct paths / model 
# ------------------------------------------------------------------
fname = FlexiBasicLearning.func_name(f)
sname = FlexiBasicLearning.shape_name(s)

subfolder = "$(fname)-$(d)dof-$(num_points)obs"
savedir   = joinpath(expdir, subfolder, sname)
datafile  = joinpath(datadir, subfolder, "sim_data_$(sname).jld2")

make_model = model_makers[func_key](d)

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
println("Fitting for datafile: $datafile")
result = FlexiBasicLearning.fit_cmaes_and_gd(datafile, savedir, make_model, save_parameters = true)
println("Finished fitting for datafile: $datafile")

# FlexiBasicLearning.plot_loss_and_fits(result, datafile)
# FlexiBasicLearning.end_to_end_gd_tracking(result, datafile; func_form = f, func_string = f_str)