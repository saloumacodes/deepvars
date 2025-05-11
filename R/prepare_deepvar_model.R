# Combined model: ----
#' Deep VAR model setup
#'
#' @param deepvar_data
#' @param num_units Number of units per LSTM layer
#' @param num_layers Number of stacked LSTM layers
#' @param p_drop_out Dropout rate between LSTM layers
#' @param epsilon Small constant added to the scale (and used as the Adam LR)
#' @param optimizer Keras optimizer (defaults to Adam with lr = epsilon)
#'
#' @return A list of compiled Keras models (one per series) wrapped in class "deepvar_model"
#' @export
#'
#' @importFrom keras `%>%`
#'
#' @author Patrick Altmeyer (modified)
prepare_deepvar_model <- function(
    deepvar_data,
    num_units   = 50,
    num_layers  = 2,
    p_drop_out  = 0.5,
    epsilon     = 0.001,
    optimizer   = keras::optimizer_adam(learning_rate = epsilon)
) {

  K <- deepvar_data$K
  N <- deepvar_data$N
  dim_input <- dim(deepvar_data$X)[2:3]

  model_list <- lapply(
    1:K,
    function(k) {

      # Build list of LSTM + dropout layers
      list_of_layers <- lapply(
        1:num_layers,
        function(layer) {
          list(
            keras::layer_lstm(
              units           = num_units,
              return_sequences = ifelse(layer < num_layers, TRUE, FALSE),
              input_shape     = dim_input
            ),
            keras::layer_dropout(rate = p_drop_out)
          )
        }
      )
      list_of_layers <- do.call(c, list_of_layers)

      # 1) Initialize sequential with the LSTM stack
      model <- keras::keras_model_sequential(list_of_layers)

      # 2) Add final Dense layer (must pass `object` by name)
      model <- keras::layer_dense(
        object     = model,
        units      = 2,
        activation = "linear"
      )

      # 3) Wrap into a DistributionLambda (must pass `object` by name)
      model <- tfprobability::layer_distribution_lambda(
        object = model,
        function(x) {
          tfprobability::tfd_normal(
            loc   = x[, 1, drop = FALSE],
            scale = epsilon + tensorflow::tf$math$softplus(x[, 2, drop = FALSE])
          )
        }
      )

      # Define negative log-likelihood loss
      negloglik <- function(y, model_out) {
        - (model_out %>% tfprobability::tfd_log_prob(y))
      }

      # 4) Compile the model
      model <- model %>% keras::compile(
        loss      = negloglik,
        optimizer = optimizer
      )

      return(model)
    }
  )

  deepvar_model <- list(
    model_list = model_list,
    model_data = deepvar_data
  )
  class(deepvar_model) <- "deepvar_model"

  return(deepvar_model)
}
