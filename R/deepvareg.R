#' Deep VAR
#'
#' @param data Input dataset
#' @param lags Number of lags
#' @param num_units Units in LSTM layers
#' @param num_layers Number of LSTM layers
#' @param p_drop_out Dropout probability
#' @param horizon Forecast horizon
#' @param type Type of model ("var" or "other")
#' @param verbose Verbosity level
#' @param bayes Bayesian mode (not used in this function)
#' @param n_mc Number of Monte Carlo samples (not used in this function)
#' @param ... Additional arguments
#'
#' @return A trained deepvar_model object
#' @export
#'
#' @author Patrick Altmeyer
deepvareg <- function(
    data,
    lags = 1,
    num_units = 50,
    num_layers = 2,
    p_drop_out = 0.5,
    horizon = 1,
    type = "var",
    verbose = 0,
    bayes = TRUE,
    n_mc = 50,
    ...
) {

  # Prepare data:
  deepvar_data <- prepare_deepvar_data(data, lags, horizon, type)

  # Prepare model:
  deepvar_model <- prepare_deepvar_model(
    deepvar_data,
    num_units = num_units,
    num_layers = num_layers,
    p_drop_out = p_drop_out
  )

  # Fit the model:
  deepvar_model <- fit(deepvar_model, verbose = verbose, ...)

  # Posterior predictive:
  deepvar_model$posterior_predictive <- posterior_predictive(deepvar_model)

  # Fitted values:
  deepvar_model$y_hat <- deepvar_model$posterior_predictive$mean

  # Predictive uncertainty:
  deepvar_model$uncertainty <- deepvar_model$posterior_predictive$sd

  # Residuals:
  deepvar_model$res <- residuals(deepvar_model)

  # Assign class:
  class(deepvar_model) <- c("deepvar_model", "dvars_model")

  return(deepvar_model)
}

