#!/usr/bin/env Rscript
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

install_or_pin <- function(pkg, version) {
  has_pkg <- requireNamespace(pkg, quietly = TRUE)
  if (has_pkg) {
    installed <- as.character(utils::packageVersion(pkg))
    if (identical(installed, version)) {
      message(sprintf("%s already pinned at %s", pkg, version))
      return(invisible(NULL))
    }
  }

  remotes::install_version(
    package = pkg,
    version = version,
    repos = getOption("repos"),
    upgrade = "never",
    dependencies = TRUE
  )
}

install_or_pin("jsonlite", "1.8.8")
install_or_pin("uwot", "0.2.3")

message(sprintf("jsonlite=%s", as.character(utils::packageVersion("jsonlite"))))
message(sprintf("uwot=%s", as.character(utils::packageVersion("uwot"))))
