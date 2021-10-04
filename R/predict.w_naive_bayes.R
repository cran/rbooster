#' Naive Bayes algorithm with case weights
#'
#' @description
#' Function for Naive Bayes algorithm prediction.
#'
#' @param object "naive_bayes" class object from naive_bayes function.
#' @param newdata new observations which predictions will be made on.
#' @param type "pred" or "prob".
#'
#' @details
#' Type "pred" will give class predictions. "prob" will give probabilities for
#' each class.
#'
#' @return A vector of class predictions or a matrix of class probabilities
#' depending of \code{type}
#'
#' @examples
#'data(iris)
#'
#'x <- iris[,1:4]
#'y <- iris[,5]
#'
#'# Without weights
#'m <- w_naive_bayes(x_train = x, y_train = y)
#'preds <- predict(object = m, newdata = x, type = "pred")
#'table(y, preds)
#'
#'# Using weights
#'weights <- ifelse(y == "setosa" | y == "versicolor", 1, 0.01)
#'
#'m <- w_naive_bayes(x_train = x, y_train = y, w = weights)
#'preds <- predict(object = m, newdata = x, type = "pred")
#'table(y, preds)
#'
#'# Using weights example 2
#'weights <- ifelse(y == "setosa" | y == "virginica", 1, 0.01)
#'
#'m <- w_naive_bayes(x_train = x, y_train = y, w = weights)
#'preds <- predict(object = m, newdata = x, type = "pred")
#'table(y, preds)
#'
#' @keywords internal
#' @export

predict.w_naive_bayes <- function(object, newdata = NULL, type = "prob", ...){

  n_train <- object$n_train
  p <- object$p
  x_classes <- object$x_classes
  n_classes <- object$n_classes
  k_classes <- object$k_classes
  priors <- object$priors
  class_names <- object$class_names
  means <- object$means
  stds <- object$stds

  x_test <- newdata
  n_test <- nrow(x_test)

  for (i in 1:p) {
    if (is.factor(x_test[,i])) {
      x_test[,i] <- as.numeric(x_test[,i])
    }
  }

  densities <- lapply(1:k_classes, function(m) sapply(1:p, function(m2) {
    d <- stats::dnorm(x_test[,m2], mean = means[[m]][m2], sd = stds[[m]][m2])
    d[is.infinite(d)] <- .Machine$double.xmax
    d[d == 0] <- 1e-20
    return(d)
  }))

  likelihoods <- sapply(1:k_classes, function(m) apply(densities[[m]], 1, prod))
  posteriors <- sapply(1:k_classes,
                       function(m) apply(cbind(priors[m], likelihoods[,m]), 1,
                                         prod))

  posteriors <- t(apply(posteriors, 1, function(m) {
    if(all(m == 0)){
      stats::runif(k_classes, min = 0, max = 1)
    } else{
      m
    }
  }))

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
