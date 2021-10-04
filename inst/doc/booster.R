## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library("rbooster")
data(iris)
set.seed(1)

cv_sampler <- function(y, train_proportion) {
  unlist(lapply(unique(y), function(m) sample(which(y==m), round(sum(y==m))*train_proportion)))
}

x <- iris[,1:4]
y <- iris[,5]
### multiclass classification
unique(y)

train_i <- cv_sampler(y, 0.8)

x_train <- x[train_i,]
y_train <- y[train_i]

x_test <- x[-train_i,]
y_test <- y[-train_i]

## ----fig.height = 4, fig.width = 8, fig.align = "center"----------------------
par(mfrow = c(1,2))
# boosting using decision tree
m <- booster(x_train = x_train, y_train = y_train,
             x_test = x_test, y_test = y_test,
             classifier = "rpart", bag_frac = 1, lambda = 1,
            print_detail = FALSE, print_plot = TRUE, max_iter = 50)
preds <- predict(object = m, newdata = x_test, print_detail = FALSE)
probs <- predict(object = m, newdata = x_test, print_detail = FALSE, type = "prob")

# boosting using decision tree example 2
m2 <- booster(x_train = x_train, y_train = y_train,
              x_test = x_test, y_test = y_test,
              classifier = "rpart", bag_frac = 0.5, lambda = 1,
              print_detail = FALSE, print_plot = TRUE, max_iter = 50)

## -----------------------------------------------------------------------------
preds2 <- predict(object = m, newdata = x_test, print_detail = FALSE)
probs2 <- predict(object = m, newdata = x_test, print_detail = FALSE, type = "prob")
head(probs)
head(probs2)

sum(preds != y_test)/length(y_test)
sum(preds2 != y_test)/length(y_test)

## ----fig.height = 4, fig.width = 8, fig.align = "center"----------------------
par(mfrow = c(1,2))
# boosting using naive bayes
m <- booster(x_train = x_train, y_train = y_train,
             x_test = x_test, y_test = y_test,
             classifier = "nb", bag_frac = 0.5, lambda = 1,
             print_detail = FALSE, print_plot = TRUE, max_iter = 175)

# boosting using naive bayes with bootstrap
m2 <- booster(x_train = x_train, y_train = y_train,
              x_test = x_test, y_test = y_test, weighted_bootstrap = TRUE,
              classifier = "nb", bag_frac = 0.5, lambda = 1,
              print_detail = FALSE, print_plot = TRUE, max_iter = 175)

## ----fig.height = 4, fig.width = 8, fig.align = "center"----------------------
preds2 <- predict(object = m, newdata = x_test, print_detail = FALSE)
probs2 <- predict(object = m, newdata = x_test, print_detail = FALSE, type = "prob")
head(probs)
head(probs2)

sum(preds != y_test)/length(y_test)
sum(preds2 != y_test)/length(y_test)

# custom classifier
library(nnet)
classifier <- function(x_train, y_train, weights) {
  x_train <- as.data.frame(x_train)
  m <- nnet(y_train~., data = data.frame(x_train, y_train), size = 1,
            weights = weights, trace = FALSE)
  return(m)
}

predicter <- function(model, x_new) {
  x_new <- as.data.frame(x_new)
  preds <- predict(object = model, newdata = x_new, type = "class")
  return(preds)
}

## ----fig.height = 4, fig.width = 8, fig.align = "center"----------------------
par(mfrow = c(1,2))
# boosting using naive bayes
m <- booster(x_train = x_train, y_train = y_train,
             x_test = x_test, y_test = y_test,
             classifier = classifier,
             predicter = predicter,
             bag_frac = 0.5, lambda = 1,
             print_detail = FALSE, print_plot = TRUE, max_iter = 100)

# boosting using naive bayes with bootstrap
m2 <- booster(x_train = x_train, y_train = y_train,
              x_test = x_test, y_test = y_test,
              classifier = classifier,
              predicter = predicter,
              weighted_bootstrap = TRUE,
              bag_frac = 0.5, lambda = 1,
              print_detail = FALSE, print_plot = TRUE, max_iter = 100)

## -----------------------------------------------------------------------------
preds2 <- predict(object = m, newdata = x_test, print_detail = FALSE)
probs2 <- predict(object = m, newdata = x_test, print_detail = FALSE, type = "prob")
head(probs)
head(probs2)

sum(preds != y_test)/length(y_test)
sum(preds2 != y_test)/length(y_test)

# Dataset 2, binary class, a non-linear separated data example
library("imbalance")
data(banana)
x <- banana[,1:2]
y <- banana[,3]

## ----fig.height = 6, fig.width = 6, fig.align = "center"----------------------
plot(x, col = y)

## -----------------------------------------------------------------------------
train_i <- cv_sampler(y, 0.9)
x_train <- x[train_i,]
y_train <- y[train_i]

x_test <- x[-train_i,]
y_test <- y[-train_i]

## ----fig.height = 5, fig.width = 6, fig.align = "center"----------------------
# rpart
m_rpart <- booster(x_train = x_train, y_train = y_train,
                   classifier = "rpart", x_test = x_test, y_test = y_test,
                   max_iter = 100, weighted_bootstrap = FALSE,
                   lambda = 1, print_detail = FALSE, print_plot = TRUE,
                   bag_frac = 0.5)
preds_rpart <- predict(object = m_rpart, newdata = x_test, type = "pred")
sum(y_test != preds_rpart)/length(y_test) # error

## ----fig.height = 5, fig.width = 6, fig.align = "center"----------------------
# glm
m_glm <- booster(x_train = x_train, y_train = y_train,
                 classifier = "glm", x_test = x_test, y_test = y_test,
                 max_iter = 500, weighted_bootstrap = FALSE,
                 lambda = 2, print_detail = FALSE, print_plot = TRUE,
                 bag_frac = 0.5)
preds_glm <- predict(object = m_glm, newdata = x_test, type = "pred", print_detail = FALSE)
sum(y_test != preds_glm)/length(y_test) # error

## ----fig.height = 5, fig.width = 6, fig.align = "center"----------------------
# custom classifier
m_nnet <- booster(x_train = x_train, y_train = y_train,
                  classifier = classifier,
                  predicter = predicter,
                  x_test = x_test, y_test = y_test,
                  max_iter = 300, weighted_bootstrap = FALSE,
                  lambda = 2, print_detail = FALSE, print_plot = TRUE,
                  bag_frac = 0.5)
preds_nnet <- predict(object = m_nnet, newdata = x_test, type = "pred", print_detail = FALSE)
sum(y_test != preds_nnet)/length(y_test) # error

# for seeing decision boundaries
x1_grid <- seq(min(x[,1]), max(x[,1]), length = 150)
x2_grid <- seq(min(x[,2]), max(x[,2]), length = 150)
grid <- expand.grid(x1_grid, x2_grid)
colnames(grid) <- colnames(x)

prob_rpart <- predict(object = m_rpart, newdata = grid, type = "prob")[,1]
prob_glm <- predict(object = m_glm, newdata = grid, type = "prob", print_detail = FALSE)[,1]
prob_nnet <- predict(object = m_nnet, newdata = grid, type = "prob", print_detail = FALSE)[,1]



## ----fig.height = 4, fig.width = 8, fig.align = "center"----------------------

par(mfrow = c(1,3))

plot(x = x_train[,1], y = x_train[,2], col = y_train, cex = 0.5, pch = ".")
points(x = x_test[,1], y = x_test[,2], col = y_test, cex = 1, pch = 16)
contour(x = x1_grid, y = x2_grid, z = matrix(prob_rpart, nrow = 150),
        levels = c(0.5), add = TRUE,
        method = "edge")

plot(x = x_train[,1], y = x_train[,2], col = y_train, cex = 0.5, pch = ".")
points(x = x_test[,1], y = x_test[,2], col = y_test, cex = 1, pch = 16)
contour(x = x1_grid, y = x2_grid, z = matrix(prob_glm, nrow = 150),
        levels = c(0.5), add = TRUE,
        method = "edge")

plot(x = x_train[,1], y = x_train[,2], col = y_train, cex = 0.5, pch = ".")
points(x = x_test[,1], y = x_test[,2], col = y_test, cex = 1, pch = 16)
contour(x = x1_grid, y = x2_grid, z = matrix(prob_nnet, nrow = 150),
        levels = c(0.5), add = TRUE,
        method = "edge")


