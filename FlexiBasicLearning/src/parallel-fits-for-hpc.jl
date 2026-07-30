# using JLD2
# using FlexiBasicLearning
# using CairoMakie
# using LinearAlgebra

# setting up dirs
expdir = "../FlexiSpaceLocal/exp/07292026/"
datadir = "../FlexiSpaceLocal/data/w_true_params/no-noise"
savedirs = []
datafiles = []
make_models = []

# prefixes = []  # to hold the prefixes for each combination of func, dof, shape

num_points = 20
dofs = [3, 4, 5, 20, 50]
# dofs = [3]
# dofs = [4,5,20,50]
shapes = [FlexiBasicLearning.crooked_flexi, FlexiBasicLearning.cu_flexi, FlexiBasicLearning.cd_flexi]
# shapes = [FlexiBasicLearning.cu_flexi] #TODO: debug
funcs = [FlexiBasicLearning.make_flexi1_func, FlexiBasicLearning.make_flexi1_alg1_func]
# funcs = [FlexiBasicLearning.make_flexi1_ode1_func] # test for 0728: with full domain, does it get faster?
# func_strings = ["y' = f(y)"]
func_strings = ["y = f(t)", "y = t ⋅ f(t)"]

# more dirs
funcs_rpts = []
func_strings_rpts = []
for f in funcs, f_str in func_strings, d in dofs, s in shapes
    fname = FlexiBasicLearning.func_name(f)
    sname = FlexiBasicLearning.shape_name(s)
    subfolder = "$(fname)-$(d)dof-$(num_points)obs"
    savedir = joinpath(expdir, subfolder, "$sname" )
    datafile = joinpath(datadir, subfolder, "sim_data_$(sname).jld2")
    push!(savedirs, savedir)
    push!(datafiles, datafile)
    push!(funcs_rpts, f)
    push!(func_strings_rpts, f_str)
    if f == FlexiBasicLearning.make_flexi1_func
        push!(make_models, () -> FlexiBasicLearning.make_ModelFlexi1(;flexi_dofs=d))
    elseif f == FlexiBasicLearning.make_flexi1_alg1_func
        push!(make_models, () -> FlexiBasicLearning.make_ModelFlexiAlg(;flexi_dofs=d))  # For now, using the same model structure for both function types
    elseif f == FlexiBasicLearning.make_flexi1_ode1_func
        push!(make_models, () -> FlexiBasicLearning.make_ModelFlexiODE(;flexi_dofs=d))
    end
    # prefixes = 
end

println(funcs_rpts)
println(func_strings_rpts)
results = []
for (datafile, savedir, make_model, f, f_str) in zip(datafiles, savedirs, make_models, funcs_rpts, func_strings_rpts)
    println("Fitting for datafile: $datafile")
    result = FlexiBasicLearning.fit_cmaes_and_gd(datafile, savedir, make_model, save_parameters = true)
    push!(results, result)
    println("Finished fitting for datafile: $datafile")
    # plot using result Dict
    FlexiBasicLearning.plot_loss_and_fits(result, datafile) # TODO: plotting doesn't used savedir anymore
    FlexiBasicLearning.end_to_end_gd_tracking(result, datafile; func_form = f, func_string = f_str)
end
