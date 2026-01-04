#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: run_mashr_two_mode.R <in_csv> <out_csv> <mode> <config_yaml>", call. = FALSE)
}

suppressPackageStartupMessages({
  library(data.table)
  library(mashr)
  library(Matrix)
  library(yaml)
})

input_csv <- args[1]
output_csv <- args[2]
mode <- args[3]
config_yaml <- args[4]

if (!(mode %in% c("conservative", "enriched"))) {
  stop("mode must be one of: conservative, enriched", call. = FALSE)
}

cfg <- yaml::read_yaml(config_yaml)
if (is.null(cfg)) {
  cfg <- list()
}

regularize_V <- function(V, eig_floor = 1e-6, rho = 0.05) {
  # nearest PSD correlation, eigen floor, then shrink-to-identity
  Vnpd <- as.matrix(nearPD(V, corr = TRUE, keepDiag = TRUE)$mat)
  eig <- eigen(Vnpd, symmetric = TRUE)
  vals <- pmax(eig$values, eig_floor)
  Vreg <- eig$vectors %*% diag(vals) %*% t(eig$vectors)
  Dinv <- diag(1 / sqrt(diag(Vreg)))
  Vreg <- Dinv %*% Vreg %*% Dinv
  K <- ncol(Vreg)
  Vout <- (1 - rho) * Vreg + rho * diag(K)
  return(Vout)
}

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

se_inflate <- if (!is.null(cfg$mash_se_inflate)) cfg$mash_se_inflate else 1.0
se_floor <- if (!is.null(cfg$mash_se_floor)) cfg$mash_se_floor else 0.0
Shat <- Shat * se_inflate
if (se_floor > 0) {
  Shat <- pmax(Shat, se_floor)
}

data <- mash_set_data(Bhat, Shat)

estimate_V <- isTRUE(cfg$mash_estimate_V)
Vhat <- if (estimate_V) {
  tryCatch(
    estimate_null_correlation_simple(data),
    error = function(e) {
      message("Warning: estimate_null_correlation_simple failed; using identity covariance before regularization.")
      diag(ncol(Bhat))
    }
  )
} else {
  diag(ncol(Bhat))
}
eig_floor <- if (!is.null(cfg$mash_v_eig_floor)) cfg$mash_v_eig_floor else 1e-6
rho <- if (!is.null(cfg$mash_v_shrink_rho)) cfg$mash_v_shrink_rho else 0.05
Vhat <- regularize_V(Vhat, eig_floor = eig_floor, rho = rho)
data <- mash_set_data(Bhat, Shat, V = Vhat)

U.c <- cov_canonical(data)
Ulist <- U.c

if (mode == "conservative") {
  prior <- "nullbiased"
  nullweight <- 10
  usepointmass <- TRUE
} else {
  prior <- if (!is.null(cfg$mash_enriched_prior)) cfg$mash_enriched_prior else "uniform"
  nullweight <- if (!is.null(cfg$mash_enriched_nullweight)) cfg$mash_enriched_nullweight else 1
  usepointmass <- if (!is.null(cfg$mash_enriched_usepointmass)) cfg$mash_enriched_usepointmass else TRUE

  if (isTRUE(cfg$mash_add_datadriven_cov)) {
    strong_thresh <- if (!is.null(cfg$mash_strong_lfsr_for_cov)) cfg$mash_strong_lfsr_for_cov else 0.1
    npc <- if (!is.null(cfg$mash_cov_pca_npc)) cfg$mash_cov_pca_npc else 2
    min_strong <- if (!is.null(cfg$mash_min_strong_for_cov)) cfg$mash_min_strong_for_cov else 50
    try({
      m.1by1 <- mash_1by1(data)
      lfsr1 <- get_lfsr(m.1by1)
      strong_idx <- which(apply(lfsr1, 1, function(x) any(x < strong_thresh)))
      if (length(strong_idx) >= min_strong) {
        data.strong <- mash_set_data(Bhat[strong_idx, , drop = FALSE], Shat[strong_idx, , drop = FALSE], V = Vhat)
        U.pca <- cov_pca(data.strong, npc = npc)
        U.ed <- cov_ed(data.strong, U.pca)
        Ulist <- c(U.c, U.ed)
      } else {
        message(sprintf(
          "Warning: strong set too small for data-driven covariances (n=%d < %d); using canonical only.",
          length(strong_idx),
          min_strong
        ))
      }
    }, silent = TRUE)
  }
}

m <- tryCatch(
  mash(data, Ulist, prior = prior, nullweight = nullweight, usepointmass = usepointmass),
  error = function(e) {
    message("Warning: mash failed with provided covariances; retrying with canonical covariances and identity V.")
    data_fallback <- mash_set_data(Bhat, Shat, V = diag(ncol(Bhat)))
    U.c.fallback <- cov_canonical(data_fallback)
    mash(data_fallback, U.c.fallback, prior = prior, nullweight = nullweight, usepointmass = usepointmass)
  }
)

postmean <- get_pm(m)
postsd <- get_psd(m)
lfsr <- get_lfsr(m)

for (j in seq_along(betahat_days)) {
  day <- betahat_days[j]
  dt[[paste0("postmean_d", day)]] <- postmean[, j]
  dt[[paste0("postsd_d", day)]] <- postsd[, j]
  dt[[paste0("lfsr_d", day)]] <- lfsr[, j]
}

lfsr_cols_out <- grep("^lfsr_d", names(dt), value = TRUE)
dt[, lfsr_min := do.call(pmin, c(.SD, na.rm = TRUE)), .SDcols = lfsr_cols_out]
dt[, best_day := betahat_days[apply(lfsr, 1, which.min)]]

fwrite(dt, output_csv)
