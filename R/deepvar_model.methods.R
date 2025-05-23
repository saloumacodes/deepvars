#' @importFrom keras `%>%`
fit.deepvar_model <- function(deepvar_model, verbose = 0, epochs = 50, ...) {
  
  K <- deepvar_model$model_data$K
  X_train <- deepvar_model$model_data$X
  y_train <- deepvar_model$model_data$y
  time_steps <- dim(X_train)[2] # number of time_steps in the X_train dataset
  
  # Fit models:
  fitted_models <- lapply(
    1:K,
    function(k) {
      model <- deepvar_model$model_list[[k]]
      
      # Reshape y_train to (samples, time_steps, 1)
      y_reshaped <- array(y_train[,,k], dim = c(dim(y_train)[1], dim(y_train)[2], 1))
      
      # Create a NEW optimizer for each model:
      optimizer <- keras::optimizer_adam(learning_rate = 0.001)  # Or use epsilon from prepare_deepvar_model if needed
      model$compile(
        optimizer = optimizer,
        loss      = deepvar_model$model_list[[k]]$loss  # Keep the same loss function
      )
      
      history <- model$fit(
        x = X_train,
        y = y_reshaped,
        epochs = epochs, #Explicitly pass the epochs parameter
        verbose = verbose,
        ...
      )
      list(
        model = model,
        history = history
      )
    }
  )
  
  # Output:
  deepvar_model$model_list <- lapply(fitted_models, function(fitted_model) fitted_model[["model"]]) # update model list
  deepvar_model$model_histories <- lapply(fitted_models, function(fitted_model) fitted_model[["history"]]) # extract history
  deepvar_model$X_train <- X_train
  deepvar_model$y_train <- y_train
  
  return(deepvar_model)
}

fit <- function(deepvar_model, ...) {
  UseMethod("fit", deepvar_model)
}

## Predictions: ----
#' @export
posterior_predictive.deepvar_model <- function(deepvar_model, X = NULL) {
  
  if (is.null(X) & !is.null(deepvar_model$y_hat)) {
    y_hat <- deepvar_model$y_hat
  } else {
    
    # Preprocessing:
    if (is.null(X)) {
      X <- deepvar_model$X_train
    }
    if (length(dim(X)) < 3) {
      # ! If new data is not 3D tensor, assume that unscaled 2D tensor was supplied !
      # Get rid of constant:
      if (all(X[,1] == 1)) {
        X <- X[,-1]
      }
      # Apply scaling:
      scaler <- deepvar_model$model_data$scaler
      lags <- deepvar_model$model_data$lags
      K <- deepvar_model$model_data$K
      X <- apply_scaler_from_training(X, scaler, lags, K)
      # Reshape:
      X <- keras::array_reshape(X, dim = c(dim(X)[1], 1, dim(X)[2]))
    }
    
    # Compute fitted values:
    fitted <- lapply(
      1:length(deepvar_model$model_list),
      function(k) {
        mod <- deepvar_model$model_list[[k]]
        fitted <- mod(X)
        y_hat <- as.numeric(fitted %>% tfprobability::tfd_mean())
        sd <- as.numeric(fitted %>% tfprobability::tfd_stddev())
        # Rescale data:
        y_hat <- invert_scaling(y_hat, deepvar_model$model_data, k = k)
        sd <- invert_scaling(sd, deepvar_model$model_data, k = k)
        return(list(y_hat = unlist(y_hat), sd = unlist(sd)))
      }
    )
    # Posterior mean:
    y_hat <- matrix(sapply(fitted, function(i) i$y_hat), ncol = deepvar_model$model_data$K)
    rownames(y_hat) <- NULL
    colnames(y_hat) <- deepvar_model$model_data$var_names
    # Posterior variance:
    sd <- matrix(sapply(fitted, function(i) i$sd), ncol = deepvar_model$model_data$K)
    rownames(sd) <- NULL
    colnames(sd) <- deepvar_model$model_data$var_names
  }
  
  return(list(mean = y_hat, sd = sd))
}

#' @export
posterior_predictive <- function(deepvar_model, X = NULL) {
  UseMethod("posterior_predictive", deepvar_model)
}

#' @export
fitted.deepvar_model <- function(deepvar_model, X = NULL) {
  if (is.null(X)) {
    y_hat <- deepvar_model$posterior_predictive$mean
  } else {
    y_hat <- posterior_predictive(deepvar_model, X)$mean
  }
  return(y_hat)
}

#' @export
uncertainty.deepvar_model <- function(deepvar_model, X = NULL) {
  if (is.null(X)) {
    uncertainty <- deepvar_model$posterior_predictive$sd
  } else {
    uncertainty <- posterior_predictive(deepvar_model, X)$sd
  }
  return(uncertainty)
}

#' @export
uncertainty <- function(deepvar_model, X = NULL) {
  UseMethod("uncertainty", deepvar_model)
}

#' @export
residuals.deepvar_model <- function(deepvar_model, X = NULL, y = NULL) {
  
  new_data <- new_data_supplied(X = X, y = y)
  
  if (new_data | is.null(deepvar_model$res)) {
    if (!new_data) {
      # If no new data is supplied, training outputs are re-scaled:
      X <- deepvar_model$X_train
      y <- deepvar_model$y_train
      y <- keras::array_reshape(y, dim = c(dim(y)[1], dim(y)[3]))
      y <- invert_scaling(y, deepvar_model$model_data)
    }
    y_hat <- fitted(deepvar_model, X = X)
    res <- y - y_hat
  } else {
    res <- deepvar_model$res
  }
  
  return(res)
}

#' @export
prepare_predictors.deepvar_model <- function(deepvar_model, data) {
  
  lags <- deepvar_model$model_data$lags
  
  # Explanatory variables:
  X = as.matrix(
    data[
      (.N - (lags - 1)):.N, # take last p rows
      sapply(
        0:(lags - 1),
        function(lag) {
          data.table::shift(.SD, lag)
        }
      )
    ][.N,] # take last row of that
  )
  
  X <- keras::array_reshape(X, dim = c(1, 1, ncol(X)))
  
  return(X)
}

#' @export
prepare_predictors <- function(deepvar_model, data) {
  UseMethod("prepare_predictors", deepvar_model)
}
