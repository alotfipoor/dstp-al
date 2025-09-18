options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!require("pak")) install.packages("pak")
pkgs = c(
  "sf",
  "tidyverse",
  "osmextract",
  "tmap",
  "maptiles",
  "stats19",
  "pct"
)
pak::pkg_install(pkgs)
sapply(pkgs, require, character.only = TRUE)