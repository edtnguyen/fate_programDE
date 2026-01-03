#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: run_mashr.R <gene_summary_for_mash.csv> <gene_summary_mash_out.csv>", call. = FALSE)
}

suppressPackageStartupMessages({
  library(data.table)
  library(mashr)
})

input_csv <- args[1]
output_csv <- args[2]

dt <- fread(input_csv)

betahat_cols <- grep("^betahat_d", names(dt), value = TRUE)
se_cols <- grep("^se_d", names(dt), value = TRUE)

if (length(betahat_cols) == 0 || length(se_cols) == 0) {
  stop("Input must include betahat_d* and se_d* columns.", call. = FALSE)
}

parse_day <- function(x) {
  as.integer(sub(".*d", "", x))
}

betahat_days <- vapply(betahat_cols, parse_day, integer(1))
se_days <- vapply(se_cols, parse_day, integer(1))

betahat_cols <- betahat_cols[order(betahat_days)]
betahat_days <- betahat_days[order(betahat_days)]
se_cols <- se_cols[order(se_days)]
se_days <- se_days[order(se_days)]

if (!identical(unname(betahat_days), unname(se_days))) {
  stop("betahat_d* and se_d* columns must have matching day indices.", call. = FALSE)
}

Bhat <- as.matrix(dt[, ..betahat_cols])
Shat <- as.matrix(dt[, ..se_cols])

data <- mash_set_data(Bhat, Shat)
Vhat <- tryCatch(
  estimate_null_correlation_simple(data),
  error = function(e) {
    message("Warning: estimate_null_correlation_simple failed; using identity covariance.")
    diag(ncol(Bhat))
  }
)
data <- mash_set_data(Bhat, Shat, V = Vhat)
U.c <- cov_canonical(data)
m <- mash(data, U.c)

postmean <- get_pm(m)
lfsr <- get_lfsr(m)

for (j in seq_along(betahat_days)) {
  day <- betahat_days[j]
  dt[[paste0("postmean_d", day)]] <- postmean[, j]
  dt[[paste0("lfsr_d", day)]] <- lfsr[, j]
}

lfsr_cols <- grep("^lfsr_d", names(dt), value = TRUE)
dt[, lfsr_min := do.call(pmin, c(.SD, na.rm = TRUE)), .SDcols = lfsr_cols]
dt[, best_day := betahat_days[apply(lfsr, 1, which.min)]]
dt[, hit_anyday := lfsr_min < 0.05]

fwrite(dt, output_csv)
