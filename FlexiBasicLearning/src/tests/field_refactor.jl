# does adding p_classical and flex1_params to the ComponentArray break anything? I think it should be fine, but need to check.

# run fit single with:

# func_key  = "flexi1"
# d         = 3
# shape_key = "crooked"


# func_key  = "flexi1_alg1"
# d         = 3
# shape_key = "crooked"


# func_key  = "flexi1_ode1"
# d         = 3
# shape_key = "crooked"


# func_key  = "flexi1_lv2"
# d         = 3
# shape_key = "crooked"

# these all run and loss goes down for everything except for ode1 (weirdly enough)...