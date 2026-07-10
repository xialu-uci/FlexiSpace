# using JLD2
using FlexiBasicLearning
using CairoMakie
# using LinearAlgebra

# setting up dirs
expdir = "../FlexiSpaceLocal/exp/07062026/"
datadir = "../FlexiSpaceLocal/data/"
savedirs = []
datafiles = []
make_models = []
# prefixes = []  # to hold the prefixes for each combination of func, dof, shape

num_points = 20
dofs = [5, 20, 50]
shapes = [FlexiBasicLearning.crooked_flexi, FlexiBasicLearning.cu_flexi, FlexiBasicLearning.cd_flexi]
funcs = [FlexiBasicLearning.make_flexi1_func, FlexiBasicLearning.make_flexi1_alg1_func]

for f in funcs, d in dofs, s in shapes
    fname = FlexiBasicLearning.func_name(f)
    sname = FlexiBasicLearning.shape_name(s)
    subfolder = "$(fname)-$(d)dof-$(num_points)obs"
    savedir = joinpath(expdir, subfolder, "$sname" )
    datafile = joinpath(datadir, subfolder, "sim_data_$(sname).jld2")
    push!(savedirs, savedir)
    push!(datafiles, datafile)
    if f == FlexiBasicLearning.make_flexi1_func
        push!(make_models, () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=d))
    elseif f == FlexiBasicLearning.make_flexi1_alg1_func
        push!(make_models, () -> FlexiBasicLearning.make_ModelFlexiAlg(flexi_dofs=d))  # For now, using the same model structure for both function types
    end
    # prefixes = 
end

for (datafile, savedir, make_model) in zip(datafiles, savedirs, make_models)
    println("Fitting for datafile: $datafile")
    result = FlexiBasicLearning.fit_cmaes_and_gd(datafile, savedir, make_model)
    println("Finished fitting for datafile: $datafile")
    # plot using result Dict
    FlexiBasicLearning.plot_loss_and_fits(result, datafile, savedir)
end

# savedir = savedirs[div(length(savedirs), 2)]
# datafile = datafiles[div(length(datafiles), 2)]
# make_model = make_models[div(length(make_models), 2)]
# result = FlexiBasicLearning.fit_cmaes_and_gd(datafile, savedir, make_model)
# FlexiBasicLearning.plot_loss_and_fits(result, datafile, savedir)