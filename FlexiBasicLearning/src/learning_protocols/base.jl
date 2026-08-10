
# Learning constants
struct LearningConstants
    qdrms_divisor::Float64
    convergence_threshold::Float64
    early_check_fraction::Float64
    min_early_iterations::Int
end

const DEFAULT_LEARNING_CONSTANTS = LearningConstants(14.0, 1/1000, 1/6, 20000)

struct CallbackConfig
    print_frequency::Int
    save_parameters::Bool
    time_grads::Bool
    verbose::Bool
    constants::LearningConstants
    min_loss::Float64
end

CallbackConfig(; print_frequency=100, save_parameters=false, time_grads=false, verbose=true, 
               constants=DEFAULT_LEARNING_CONSTANTS, min_loss=0.0) = 
    CallbackConfig(print_frequency, save_parameters, time_grads, verbose, constants, min_loss)

    # Standardized callback functions
function create_standard_callback(protocol_name::String, config::CallbackConfig)
    loss_history = Float32[]
    parameter_history = config.save_parameters ? [] : nothing
    
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        if config.verbose && current_iter % config.print_frequency == 0
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In $protocol_name, iteration $current_iter: loss=$lossval, qdrms=$qdrms at $(now())")
            flush(stdout)
        end
        
        if config.save_parameters
            push!(parameter_history, deepcopy(p))
        end
        
        return false
    end
    
    return callback, loss_history, parameter_history
end

function create_bbo_callback_with_early_termination(config::CallbackConfig, maxiters::Int, 
                                                  intermediate_save=nothing)
    loss_history = Float32[]
    
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        if config.verbose && current_iter % 1000 == 0
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In bb, Current loss after $current_iter iterations: $lossval, qd rms=$qdrms at $(now())")
            flush(stdout)
        end

        if !isnothing(intermediate_save) && current_iter % 10000 == 0
            # Note: intermediate save would need specific implementation
        end
        
        early_check_point = max(ceil(Int, maxiters * config.constants.early_check_fraction), 
                               config.constants.min_early_iterations)
        if current_iter >= early_check_point
            halfway_point = current_iter - ceil(Int, current_iter / 2)
            previous_loss = loss_history[halfway_point]
            current_loss = loss_history[end]
            relative_change = abs(current_loss - previous_loss) / current_loss
            
            if relative_change < config.constants.convergence_threshold
                println("Early termination at iteration $current_iter: Loss stabilized (change: $relative_change).")
                return true
            end
        end

        return false
    end
    
    return callback, loss_history
end