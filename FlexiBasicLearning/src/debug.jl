using Zygote

params = FlexiFunctions.generate_flexi_ig(5)
x = my_prob.data[:,1]

# Test 1: does evaluate_decompress differentiate?
loss1, grads1 = Zygote.withgradient(p -> sum(FlexiFunctions.evaluate_decompress.(x, Ref(p))), params)
println("Test 1 grads: $grads1")

# Test 2: does fw differentiate?
loss2, grads2 = Zygote.withgradient(p -> sum(FlexiBasicLearning.fw(x, p, my_prob.model)), params)
println("Test 2 grads: $grads2")

# Test 3: does get_loss differentiate?
loss3, grads3 = Zygote.withgradient(p -> FlexiBasicLearning.get_loss(p; learning_problem=my_prob), params)
println("Test 3 grads: $grads3")