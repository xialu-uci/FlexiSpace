using JLD2
using FlexiBasicLearning
#load crooked sim data

datafile = "../FlexiSpaceLocal/data/sim_data_cu_5seg.jld2"
savedir = "../FlexiSpaceLocal/data/06302026-flexi-basic-cu"
mkpath(savedir)  # creates the directory if it doesn't already exist

@load datafile data
num_points = size(data)[1]
# no mask for now
full = Vector{Bool}(trues(num_points))

# set up model, LearningProblem
my_model = FlexiBasicLearning.make_ModelFlexi1()
my_prob = LearningProblem(
    data = data,
    model = my_model,
    mask = full
)

# fit with cmaes
ig = my_model.params
cmaes_fit_params, cmaes_loss_history = FlexiBasicLearning.cmaes_learn(my_prob, ig)

# fit with grad descent
gd_fit_params, gd_loss_history = FlexiBasicLearning.gradient_descent_learn(my_prob,ig)

using CairoMakie
using FlexiBasicLearning
using LinearAlgebra


# --- Plot 1: overlay of true function, data, and fits ---

x = data[:, 1]
y_data = data[:, 2]
x_grid = collect(LinRange(0.0, 1.0, 500))

# crooked
# true_dofs_params = let
#     p = zeros(5)
#     p[1:2:end] .= 1.0
#     p / norm(p)
# end

# cu
true_dofs_params = let
    p = collect(1:5)
    p / norm(p)
    
end

# cd 
# true_dofs_params = let 
#     p = collect(5:-1:1)
#     p / norm(p)  
# end

y_true  = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(true_dofs_params))
y_cmaes = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(cmaes_fit_params))
y_gd    = FlexiBasicLearning.FlexiFunctions.evaluate_decompress.(x_grid, Ref(gd_fit_params))

fig1 = Figure(size = (800, 500))
ax1 = Axis(fig1[1, 1], xlabel = "x", ylabel = "y",
           title = "Flexi fit comparison (cu, 5 segments)")

CairoMakie.scatter!(ax1, x, y_data, label = "noisy data", markersize = 4, color = (:gray, 0.4))
CairoMakie.lines!(ax1, x_grid, y_true,  label = "true",            linewidth = 2, color = :black, linestyle = :dash)
CairoMakie.lines!(ax1, x_grid, y_cmaes, label = "cmaes fit",        linewidth = 2, color = :red)
CairoMakie.lines!(ax1, x_grid, y_gd,    label = "grad descent fit", linewidth = 2, color = :blue)
axislegend(ax1, position = :rb)
save(joinpath(savedir, "cu_fit_overlay.png"), fig1)

# --- Plot 2: loss histories, side by side ---

fig2 = Figure(size = (900, 400))

ax_cmaes = Axis(fig2[1, 1], xlabel = "iteration", ylabel = "loss",
                title = "CMA-ES loss", yscale = log10)
lines!(ax_cmaes, cmaes_loss_history, color = :red, linewidth = 2)

ax_gd = Axis(fig2[1, 2], xlabel = "iteration", ylabel = "loss",
             title = "Gradient descent loss", yscale = log10)
lines!(ax_gd, gd_loss_history, color = :blue, linewidth = 2)

linkyaxes!(ax_cmaes, ax_gd)

save(joinpath(savedir, "cu_loss_history.png"), fig2)

println("Saved cu_fit_overlay.png and cu_loss_history.png to $savedir")