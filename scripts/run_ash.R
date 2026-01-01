#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(ashr)
})

args <- commandArgs(trailingOnly=TRUE)
in_csv  <- args[1]
out_csv <- args[2]

dt <- fread(in_csv)

fit <- ash(
  betahat   = dt$betahat,
  sebetahat = dt$sebetahat,
  method    = "shrinkage",
  mixcompdist = "halfuniform",
  pointmass = FALSE,
  outputlevel = c("PosteriorMean","PosteriorSD","lfsr","qvalue")
)

dt[, postmean := get_pm(fit)]
dt[, postsd   := get_psd(fit)]
dt[, lfsr     := get_lfsr(fit)]
dt[, qvalue   := get_qvalue(fit)]

fwrite(dt, out_csv)
