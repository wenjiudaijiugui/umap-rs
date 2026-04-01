#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(uwot)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 10) {
  stop(
    "Usage: run_r_uwot_algo.R <input.csv> <output.csv> <n_neighbors> <n_components> <n_epochs> <seed> <init> <metric> <warmup> <repeats> [knn_idx.csv] [knn_dist.csv]"
  )
}

input_path <- args[[1]]
output_path <- args[[2]]
n_neighbors <- as.integer(args[[3]])
n_components <- as.integer(args[[4]])
n_epochs <- as.integer(args[[5]])
seed <- as.integer(args[[6]])
init <- args[[7]]
metric <- args[[8]]
warmup <- as.integer(args[[9]])
repeats <- as.integer(args[[10]])

x <- as.matrix(read.csv(input_path, header = FALSE, check.names = FALSE))
storage.mode(x) <- "double"

nn_method <- NULL
if (length(args) >= 12) {
  idx_path <- args[[11]]
  dist_path <- args[[12]]

  idx <- as.matrix(read.csv(idx_path, header = FALSE, check.names = FALSE))
  dist <- as.matrix(read.csv(dist_path, header = FALSE, check.names = FALSE))

  if (!all(dim(idx) == dim(dist))) {
    stop("precomputed knn idx and dist dimensions must match")
  }

  # uwot expects 1-based row indices in precomputed neighbor data.
  idx <- matrix(as.integer(round(idx)) + 1L, nrow = nrow(idx), ncol = ncol(idx))
  storage.mode(dist) <- "double"

  nn_method <- list(idx = idx, dist = dist)
}

fit_times <- c()
emb <- NULL
total <- warmup + repeats
for (i in seq_len(total)) {
  set.seed(seed)
  t0 <- proc.time()[["elapsed"]]
  emb <- uwot::umap(
    x,
    n_neighbors = n_neighbors,
    n_components = n_components,
    metric = metric,
    n_epochs = n_epochs,
    learning_rate = 1.0,
    init = init,
    min_dist = 0.1,
    spread = 1.0,
    set_op_mix_ratio = 1.0,
    local_connectivity = 1.0,
    repulsion_strength = 1.0,
    negative_sample_rate = 5,
    n_threads = 1,
    n_sgd_threads = 0,
    fast_sgd = FALSE,
    ret_model = FALSE,
    seed = seed,
    nn_method = nn_method,
    verbose = FALSE
  )
  dt <- proc.time()[["elapsed"]] - t0
  if (i > warmup) {
    fit_times <- c(fit_times, as.numeric(dt))
  }
}

if (is.null(emb)) {
  stop("embedding generation failed")
}

write.table(
  emb,
  file = output_path,
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

fit_mean <- if (length(fit_times) > 0) mean(fit_times) else 0.0
fit_std <- if (length(fit_times) > 0) sd(fit_times) else 0.0
if (is.na(fit_std)) {
  fit_std <- 0.0
}

result <- list(
  mode = "fit",
  metric = metric,
  precomputed_knn = !is.null(nn_method),
  fit_times_sec = fit_times,
  fit_mean_sec = fit_mean,
  fit_std_sec = fit_std
)
cat(toJSON(result, auto_unbox = TRUE), "\n")
