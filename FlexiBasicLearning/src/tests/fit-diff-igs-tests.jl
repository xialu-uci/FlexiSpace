# tests for overloaded fit_cmaes_and_gd()
using FlexiBasicLearning
using Random

Random.seed!(1234)

# # test 1: not entering ig should use default ig from model
# datafile = "../FlexiSpaceLocal/data/flexi1-50dof-20obs/sim_data_crooked.jld2"
# savedir = "../FlexiSpaceLocal/exp/07132026-test-mult-ig/flexi1-50dof-20obs/crooked"
# make_model = () -> FlexiBasicLearning.make_ModelFlexi1(flexi_dofs=50)
# # result1 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, joinpath(savedir, "test1"), make_model)
# # FlexiBasicLearning.plot_loss_and_fits(result1, datafile)

# # # test 2: entering ig should use the provided ig
# # ig1 = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(5)  # this should be EXACTLY the same as default ig in test 1.
# # result2 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, joinpath(savedir, "test2"), make_model; igs=[ig1])
# # FlexiBasicLearning.plot_loss_and_fits(result2, datafile)

# # test 3: entering ig that is different from default ig should use the provided ig

# ig = FlexiBasicLearning.FlexiFunctions.generate_flexi_ig(50)
# ig_list = [ig, ig .+ 0.1*randn(50), ig .+ 0.2*randn(50)] # all igs must be non-negative, so we will floor them at 0.0
# ig_list = [max.(ig, 0.0) for ig in ig_list]
# result3 = FlexiBasicLearning.fit_cmaes_and_gd(datafile, joinpath(savedir, "test3"), make_model; igs=ig_list)

# FlexiBasicLearning.plot_loss_and_fits(result3, datafile)

# # make table with the end losses for test 3, save to txt
# using Printf

# function save_end_loss_table(results, ig_list, outfile)
#     mkpath(dirname(outfile))
#     open(outfile, "w") do io
#         println(io, rpad("i", 4), rpad("cmaes_end_loss", 18), rpad("gd_end_loss", 16), "ig")
#         println(io, "-"^80)
#         for (i, r) in enumerate(results)
#             cmaes_end = r["cmaes_loss_history"][end]
#             gd_end    = r["gd_loss_history"][end]
#             ig_str    = string(round.(ig_list[i], digits=3))
#             println(io, rpad(i, 4),
#                         rpad(@sprintf("%.6f", cmaes_end), 18),
#                         rpad(@sprintf("%.6f", gd_end), 16),
#                         ig_str)
#         end
#     end
#     println("Saved end-loss table to $outfile")
# end

# save_end_loss_table(result3, ig_list, joinpath(savedir, "test3", "end_losses.txt"))

# let's loss slice these same exact ones to see if these plots look nonconvex. They should not.
true_params = FlexiBasicLearning.crooked_flexi(50)
arcs = FlexiBasicLearning.loss_from_ig_to_true(ig_list, true_params, learning_problem, joinpath(savedir, "test3"); step= 0.01, plot=true);

# want to print each change in direction of loss for each result
# print (p,loss)_i if (p,loss)(i-1) > (p,loss)i AND (p<loss)(i+1)< (p,loss)i or vice versa

for arc in arcs
    println(arc.ig)
    p_range = arc.p_range
    loss_values = arc.loss_values
    for i in 2:(length(loss_values)-1)
        pre = loss_values[i] -loss_values[i-1]
        post = loss_values[i+1]-loss_values[i]
        if !(signbit(pre)==signbit(post))
            println("caught!")
            println(p_range[i])
            println(loss_values[i])
        end
    end
end