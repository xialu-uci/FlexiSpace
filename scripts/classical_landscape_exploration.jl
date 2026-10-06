using FlexiBasicLearning
using ComponentArrays
using JLD2
using Printf
using Zygote
using CairoMakie



datafile = joinpath(FlexiBasicLearning.asset_dir, "mixed_true_params/no-noise/a1.0/flexi1lv2-4dof-32obs/sim_data_crooked_tmax50.0.jld2") # higher sampling freq
savedir_base = joinpath(FlexiBasicLearning.asset_dir, "exp/10062026/lv2_classical_landscape_exploration/gt-mixed_flexi1_lv2_a1.0-crooked4_tmax50.0/")
# mkpath(savedir)

@load datafile true_params

# uncomment for local testing
# d = 4
# optimizer = :cmaes
# opt_str = "cmaes"

str_to_opt = Dict("cmaes" => :cmaes, "bfgs" => :bfgs)

# results_list = []
# for d in dofs

# uncomment for HPC
d         = parse(Int, ARGS[1]) # fails in this line
opt_str = ARGS[2]
optimizer = str_to_opt[opt_str]


make_model = () -> FlexiBasicLearning.make_ModelMixedLV(;flexi_dofs = d)

# ig_derepr= make_model().params_derepresented_ig

savedir = joinpath(savedir_base, "$opt_str/fit_w_mixed_flexi1_lv2_crooked$d")

# my_prob, results, loss_landscapes = fit_mixed_alg(datafile, savedir, make_model; n_rounds = 3, optimizer = optimizer)
results_dict = FlexiBasicLearning.fit_mixed_alg(datafile, savedir, make_model; n_rounds = 3, optimizer = optimizer)

