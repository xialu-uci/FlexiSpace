using FlexiBasicLearning

a = 1.0
# funcs = [FlexiBasicLearning.make_flexi1_func, FlexiBasicLearning. make_flexi1_alg1_func, FlexiBasicLearning.make_flexi1_ode1_func, FlexiBasicLearning.make_flexi1_lv_func]
# # # num_points = [4, 8, 16, 32, 64, 128, 254, 512]
# # # dofs = [4, 8, 16, 32, 64, 128, 254, 512]
funcs = [FlexiBasicLearning.make_flexi1_lv_func]
x_max = 5e1
num_points = [32]
dofs = [8]
skeys = ["crooked"]


for n in num_points, f in funcs, d in dofs, sname in skeys
    fname = FlexiBasicLearning.func_name(f)
    s        = FlexiBasicLearning.shapes[sname]
    save_name = joinpath("mixed_true_params/no-noise/a$(a)/$(fname)-$(d)dof-$(n)obs", "sim_data_$(sname)_tmax$(x_max).jld2")
    FlexiBasicLearning.sim_data(n, d; x_max = x_max, std = 0.0, func_form = f, shape = s, save_name = save_name)
end