#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(ashr)
})

fit_ash <- function(l2fc, l2fc_se, min_l2fc, max_l2fc, n_sample) {
  # Suppress warning about biased lfsr estimates due to not using null biased prior.
  if (min_l2fc == max_l2fc) {
    pad <- ifelse(max_l2fc == 0, 1, abs(max_l2fc) * 0.1)
    min_l2fc <- min_l2fc - pad
    max_l2fc <- max_l2fc + pad
  }
  withCallingHandlers({
    ash_model <- ash(
      betahat = l2fc,
      sebetahat = l2fc_se,
      mixcompdist = "halfuniform",
      pointmass = FALSE,
      prior = "uniform",
      grange = c(min_l2fc, max_l2fc),
      outputlevel = c("PosteriorMean","PosteriorSD","lfsr","qvalue")
    )
    ash_model
  }, warning = function(w) {
    if (grepl("nullbiased", conditionMessage(w))) {
      invokeRestart("muffleWarning")
    }
  })
}

args <- commandArgs(trailingOnly=TRUE)
in_csv  <- args[1]
out_csv <- args[2]

dt <- fread(in_csv)

l2fc <- dt$betahat
l2fc_se <- dt$sebetahat
l2fc_range <- range(l2fc, na.rm = TRUE)
fit <- fit_ash(l2fc, l2fc_se, l2fc_range[1], l2fc_range[2], nrow(dt))

dt[, postmean := get_pm(fit)]
dt[, postsd   := get_psd(fit)]
dt[, lfsr     := get_lfsr(fit)]
dt[, qvalue   := get_qvalue(fit)]

fwrite(dt, out_csv)
