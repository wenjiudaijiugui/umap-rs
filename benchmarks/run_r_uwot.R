#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(uwot)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("Usage: run_r_uwot.R <input.csv> <output.csv> <n_neighbors> <n_components> <n_epochs> <seed> <init>")
}

input_path <- args[[1]]
output_path <- args[[2]]
n_neighbors <- as.integer(args[[3]])
n_components <- as.integer(args[[4]])
n_epochs <- as.integer(args[[5]])
seed <- as.integer(args[[6]])
init <- args[[7]]

x <- as.matrix(read.csv(input_path, header = FALSE, check.names = FALSE))
storage.mode(x) <- "double"

set.seed(seed)
emb <- uwot::umap(
  x,
  n_neighbors = n_neighbors,
  n_components = n_components,
  metric = "euclidean",
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
  verbose = FALSE
)

write.table(
  emb,
  file = output_path,
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
