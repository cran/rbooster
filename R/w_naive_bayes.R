#' Naive Bayes algorithm with case weights
#'
#' @description
#' Function for Naive Bayes algorithm classification.
#'
#' @param x_train features.
#' @param y_train a factor class variable.
#' @param w a vector of case weights.
#'
#' @details
#' It uses Gaussian densities with case weights and allows
#' multiclass classification.
#'
#'
#' @return a w_naive_bayes object with below components.
#'  \item{n_train}{Number of cases in the input dataset.}
#'  \item{p}{Number of features.}
#'  \item{x_classes}{A list of datasets, which are \code{x_train} separated
#'  for each class.}
#'  \item{n_classes}{Number of cases for each class in input dataset.}
#'  \item{k_classes}{Number of classes in class variable.}
#'  \item{priors}{Prior probabilities.}
#'  \item{class_names}{Names of classes in class variable.}
#'  \item{means}{Weighted mean estimations for each variable.}
#'  \item{stds}{Weighted standart deviation estimations for each variable.}
#'
#' @examples
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
#' @export

w_naive_bayes <- function(x_train, y_train, w = NULL){
  n_train <- nrow(x_train)
  p <- ncol(x_train)

  for (i in 1:p) {
    if (is.factor(x_train[,i])) {
      x_train[,i] <- as.numeric(x_train[,i])
    }
  }

  if (is.null(w)) {
    w <- rep(1, n_train)
  }
  w <- w*n_train/sum(w)

  class_names <- unique(y_train)
  k_classes <- length(class_names)

  n_train <- nrow(x_train)
  n_classes <- sapply(class_names, function(m) sum(y_train == m))

  priors <- sapply(class_names, function(m) sum(w[y_train == m])/n_train)

  x_classes <- lapply(class_names, function(m) x_train[y_train == m,])
  w_classes <- lapply(class_names, function(m) w[y_train == m])

  means <- lapply(1:k_classes, function(m2) sapply(1:p, function(m) {
    ww <- w_classes[[m2]]/sum(w_classes[[m2]])*n_classes[m2]
    ms <- Hmisc::wtd.mean(x = x_classes[[m2]][,m], na.rm = TRUE, weights = ww)
    return(ms)
  }))

  stds <- lapply(1:k_classes, function(m2) sapply(1:p, function(m) {
    ww <- w_classes[[m2]]/sum(w_classes[[m2]])*n_classes[m2]
    vars <- Hmisc::wtd.var(x = x_classes[[m2]][,m], na.rm = TRUE, weights = ww)
    return(sqrt(vars))
  }))

  model <- structure(list(n_train = n_train,
                          p = p,
                          x_classes = x_classes,
                          n_classes = n_classes,
                          k_classes = k_classes,
                          priors = priors,
                          class_names = class_names,
                          means = means,
                          stds = stds),
                     class = "w_naive_bayes")

  return(model)
}
