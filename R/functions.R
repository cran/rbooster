#' Functions to be used internally
#'
#' @description for internal use
#'
#' @param x_train input features.
#' @param y_train factor class variable.
#' @param weights instance weights.
#' @param model model obtained from respective classifier.
#' @param x_new new features for prediction.
#' @param ... other control parameters.
#'
#' @return Classifiers produce an object which is appropriate
#' for respective predicter. Predicters returns class
#' predictions for \code{x_new}.
#'
#' @rdname fun
#' @keywords internal
#' @export

classifier_rpart <- function(x_train, y_train, weights, ...) {
  model <- rpart::rpart(formula = y_train~.,
                        data = data.frame(x_train, y_train),
                        weights = weights,
                        control = rpart::rpart.control(minsplit = -1,
                                                       maxcompete = 0,
                                                       maxsurrogate = 0,
                                                       usesurrogate = 0),
                        ...)
  return(model)
}


#'
#' @rdname fun
#' @keywords internal
#' @export

predicter_rpart <- function(model, x_new, ...) {
  x_new <- as.data.frame(x_new)
  preds <- predict(object = model, newdata = x_new, type = "class", ...)
  return(preds)
}


#' @rdname fun
#' @keywords internal
#' @export

classifier_glm <- function(x_train, y_train, weights, ...){
  model <- suppressWarnings(stats::glm(y_train~., family = "binomial",
                                       data = data.frame(x_train, y_train),
                                       weights = weights, ...))
  return(model)
}

#' @rdname fun
#' @keywords internal
#' @export

predicter_glm <- function(model, x_new, ...) {
  dat <- model$data
  y <- dat[,ncol(dat)]
  class_names <- levels(y)
  class_pos <- names(which.min(table(y)))
  class_neg <- as.character(class_names[class_names != class_pos])

  x_new <- as.data.frame(x_new)
  preds <- stats::predict.glm(object = model, newdata = x_new, type = "response", ...)
  preds <- factor(ifelse(preds > 0.5, class_pos, class_neg),
                  levels = class_names, labels = class_names)
  return(preds)
}

#' @rdname fun
#' @keywords internal
#' @export

classifier_nb <- function(x_train, y_train, weights, ...){
  model <- w_naive_bayes(x_train = x_train, y_train = y_train, w = weights)
  return(model)
}

#' @rdname fun
#' @keywords internal
#' @export

predicter_nb <- function(model, x_new, ...) {
  preds <- predict.w_naive_bayes(object = model, newdata = x_new,
                                 type = "pred", ...)
  return(preds)
}

#' @rdname fun
#' @keywords internal
#' @export

classifier_earth <- function(x_train, y_train, weights, ...){
  model <- earth::earth(x = x_train, y = y_train, weights = weights, nk = 4,
                        ...)
  return(model)
}

#' @rdname fun
#' @keywords internal
#' @export

predicter_earth <- function(model, x_new, ...) {
  preds <- predict(object = model, newdata = x_new,
                   type = "class", ...)
  return(preds)
}
