#' Deep VAR model setup (custom NLL loss)
#'
#' @param deepvar_data Processed data list from deepvars::prepare_deepvar_data
#' @param num_units   Number of LSTM units per layer
#' @param num_layers  Number of LSTM layers per series
#' @param p_drop_out  Dropout probability between layers
#' @param epsilon     Learning rate for optimizer
#'
#' @return A "deepvar_model" object containing compiled Keras models
#' @export
prepare_deepvar_model <- function(
    deepvar_data,
    num_units   = 50,
    num_layers  = 2,
    p_drop_out  = 0.5,
    epsilon     = 0.001
) {
  # Number of series and input dimensions
  K         <- deepvar_data$K
  dim_input <- dim(deepvar_data$X)[2:3]
  
  # Build one model per series
  model_list <- lapply(seq_len(K), function(k) {
    # Input layer
    input_layer <- keras::layer_input(shape = dim_input)
    x <- input_layer
    
    # Add LSTM + Dropout layers
    for (i in seq_len(num_layers)) {
      return_seq <- i < num_layers
      x <- keras::layer_lstm(
        units            = num_units,
        return_sequences = return_seq
      )(x)
      x <- keras::layer_dropout(rate = p_drop_out)(x)
    }
    
    # Final dense layer outputs 2 units: [mean, raw_scale]
    preds <- keras::layer_dense(
      units      = 2,
      activation = "linear"
    )(x)
    
    # Build the model
    model <- keras::keras_model(
      inputs  = input_layer,
      outputs = preds
    )
    
    # Custom Negative Log-Likelihood loss
    nll_loss <- function(y_true, y_pred) {
      mu        <- y_pred[, 1, drop = FALSE]
      sigma_raw <- y_pred[, 2, drop = FALSE]
      sigma     <- 1e-3 + tensorflow::tf$math$softplus(sigma_raw)
      dist      <- tfprobability::tfd_normal(loc = mu, scale = sigma)
      -tensorflow::tf$reduce_mean(dist$log_prob(y_true))
    }
    
    # ✅ Compile model (Keras 3 compatible)
    model$compile(
      optimizer = keras::optimizer_adam(learning_rate = epsilon),
      loss      = nll_loss
    )
    
    return(model)
  })
  
  # Package result
  deepvar_model <- list(
    model_list = model_list,
    model_data = deepvar_data
  )
  class(deepvar_model) <- c("deepvar_model", "dvars_model")
  return(deepvar_model)
}
