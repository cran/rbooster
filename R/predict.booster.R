#' Prediction function for Adaboost framework
#'
#' Makes predictions based on booster function
#'
#' @param object booster object
#' @param newdata a factor class variable. Boosting algorithm allows for
#' @param type pre-ready or a custom classifier function.
#'
#' @details
#' Discrete Adaboost does not estimate class probabilities but here, "prob"
#' gives class probabilities. Probabilities are calculated using sum
#' of weighted class votes, that is votes/sum(votes). It has no theoretical
#' background. Use it on your own risk.
#'
#' @return A vector of class predictions or a matrix of class probabilities
#' depending of \code{type}
#'
#' @keywords internal
#' @export

predict.booster <- function(object, newdata, type = "pred", print_detail = FALSE, ...){
  if (class(object) != "booster") {
    stop("object class must be 'booster'")
  }

  n_train <- object$n_train
  w <- object$w
  alpha <- object$alpha
  models <- object$models
  x_classes <- object$x_classes
  n_classes <- object$n_classes
  k_classes <- object$k_classes
  class_names <- object$class_names
  predicter <- object$predicter

  x_test <- newdata
  n_test <- nrow(x_test)

  posteriors_all <- matrix(NA, nrow = n_test, ncol = k_classes)

  fit_test <- matrix(0, nrow = n_test, ncol = k_classes)

  for (i in 1:length(models)) {
    preds <- predicter(models[[i]], x_test)
    preds_num <- (sapply(class_names, function(m) as.numeric(preds == m)))

    fit_test <- fit_test + alpha[i]*preds_num
    if (print_detail) {
      cat("\r",
          "%",
          formatC(i/length(models)*100, digits = 2, format = "f"),
          " |",
          rep("=", floor(i/length(models)*20)),
          rep("-", 20 - ceiling(i/length(models)*20)),
          "|   ", sep = "")
    }
  }

  posteriors <- t(apply(fit_test, 1, function(m) m/sum(m)))
  posteriors[is.infinite(posteriors)] <- .Machine$double.xmax
  posteriors <- posteriors/apply(posteriors, 1, sum)

  colnames(posteriors) <- class_names

  if (type == "prob") {
    return(posteriors)
  }
  if (type == "pred") {
    predictions <- apply(posteriors, 1, function(m) class_names[which.max(m)])
    return(predictions)
  }
}
