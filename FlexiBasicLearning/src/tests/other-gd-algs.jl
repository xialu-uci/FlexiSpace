# for a 1 model types (try simple one): do bfgs, gd, adam, cmaes. 
    # want to see that: (1) each gd alg runs, (2) loss history is decreasing.



# for 1 model types (try difficult ones): 
    # do bfgs, gd, adam, cmaes. (save_parmameters = true)
        # this is more like a preliminary result than a test.
        # hope to see: (1) new gd algs are speedier than vanilla gd. do they perform on par with cmaes? does the gradient seem near zero at the end of optimization?
   
# for all model types: 
    # do bfgs, gd, adam (low maxiter, time_grads = true, save_parameters = true)
    # this is more like a preliminary result than a test.
    # hope to see: (1) new gd algs are speedier than vanilla gd. how do the number of gradient eval calls compare across gd algs?


# function fit_all_algs() <-- should be similar to fit_cmaes_and_gd, but takes a list of gd algs and runs them all, saving results in a single file. 

