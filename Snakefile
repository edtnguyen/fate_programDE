import time

configfile: "config.yaml"

OUT = config["out_dir"]

SIM_OUT = config.get("sim_out_dir", "out_fate_pipeline_sim")
SIM_ADATA = config.get("sim_adata_path", "data/sim_adata.h5ad")
SIM_GUIDE_MAP = config.get("sim_guide_map_csv", "data/sim_guide_map.csv")
SIM_CONFIG = config.get("sim_config_path", f"{SIM_OUT}/sim_config.yaml")
SIM_RECOVERY = config.get("sim_recovery_csv", f"{SIM_OUT}/sim_recovery.csv")
SIM_EXPORT = config.get("sim_export_csv", f"{SIM_OUT}/gene_summary_for_ash.csv")
SIM_ASH = config.get("sim_ash_csv", f"{SIM_OUT}/gene_summary_ash_out.csv")
SIM_HITS = config.get("sim_hits_csv", f"{SIM_OUT}/hits_ranked.csv")

SIM_CELLS = config.get("sim_cells", 200)
SIM_GENES = config.get("sim_genes", 30)
SIM_GUIDES = config.get("sim_guides", 90)
SIM_DAYS = config.get("sim_days", 4)
SIM_REPS = config.get("sim_reps", 2)
SIM_KMAX = config.get("sim_kmax", 4)
SIM_NTC_GUIDES = config.get("sim_ntc_guides", 1)
SIM_NTC_FRAC = config.get("sim_ntc_frac", 0.2)
SIM_CONC = config.get("sim_concentration", 50.0)
SIM_NUM_STEPS = config.get("sim_num_steps", 500)
SIM_BATCH = config.get("sim_batch_size", 128)
SIM_LR = config.get("sim_lr", 1e-3)
SIM_CLIP = config.get("sim_clip_norm", 5.0)
SIM_DRAWS = config.get("sim_num_draws", 200)
SIM_ALWAYS_RUN = config.get("sim_always_run", True)
SIM_FORCE = None
if SIM_ALWAYS_RUN:
    SIM_FORCE_TOKEN = config.get("sim_force_token", None)
    if SIM_FORCE_TOKEN is None:
        SIM_FORCE_TOKEN = int(time.time())
    SIM_FORCE = f"{SIM_OUT}/.sim_force_{SIM_FORCE_TOKEN}"

STRESS_SUMMARY = config.get("sim_stress_summary", f"{SIM_OUT}/stress_summary.csv")
STRESS_DETAIL = config.get("sim_stress_detail", f"{SIM_OUT}/stress_detail.csv")

PRIOR_SWEEP_SUMMARY = config.get("sim_prior_sweep_summary", f"{SIM_OUT}/prior_sweep_summary.csv")
PRIOR_SWEEP_DETAIL = config.get("sim_prior_sweep_detail", f"{SIM_OUT}/prior_sweep_detail.csv")
SWEEP_SEEDS = config.get("sim_sweep_seeds", "0,1")
SWEEP_S_TIME = config.get("sim_sweep_s_time", "1.0,0.5")
SWEEP_S_GUIDE = config.get("sim_sweep_s_guide", "1.0,0.5")
SWEEP_CONC = config.get("sim_sweep_concentration", 5.0)
SWEEP_CELLS = config.get("sim_sweep_cells", 200)
SWEEP_STEPS = config.get("sim_sweep_num_steps", 300)
SWEEP_DRAWS = config.get("sim_sweep_num_draws", 200)
SWEEP_BATCH = config.get("sim_sweep_batch_size", 128)
SWEEP_LR = config.get("sim_sweep_lr", 1e-3)
SWEEP_CLIP = config.get("sim_sweep_clip_norm", 5.0)
SWEEP_GENES = config.get("sim_sweep_genes", SIM_GENES)
SWEEP_GUIDES = config.get("sim_sweep_guides", SIM_GUIDES)
SWEEP_DAYS = config.get("sim_sweep_days", SIM_DAYS)
SWEEP_REPS = config.get("sim_sweep_reps", SIM_REPS)
SWEEP_KMAX = config.get("sim_sweep_kmax", SIM_KMAX)
SWEEP_NTC_GUIDES = config.get("sim_sweep_ntc_guides", SIM_NTC_GUIDES)
SWEEP_NTC_FRAC = config.get("sim_sweep_ntc_frac", SIM_NTC_FRAC)

TAU_SWEEP_SUMMARY = config.get("sim_tau_sweep_summary", f"{SIM_OUT}/tau_sweep_summary.csv")
TAU_SWEEP_DETAIL = config.get("sim_tau_sweep_detail", f"{SIM_OUT}/tau_sweep_detail.csv")
TAU_SWEEP_VALUES = config.get("sim_tau_sweep_values", "1.0,0.7,0.5,0.3,0.1")
TAU_SWEEP_S_TIME = config.get("sim_tau_sweep_s_time", 0.3)
TAU_SWEEP_S_GUIDE = config.get("sim_tau_sweep_s_guide", 0.5)
TAU_SWEEP_SEEDS = config.get("sim_tau_sweep_seeds", SWEEP_SEEDS)
TAU_SWEEP_CONC = config.get("sim_tau_sweep_concentration", 2.0)
TAU_SWEEP_CELLS = config.get("sim_tau_sweep_cells", 100)
TAU_SWEEP_STEPS = config.get("sim_tau_sweep_num_steps", 300)
TAU_SWEEP_DRAWS = config.get("sim_tau_sweep_num_draws", 200)
TAU_SWEEP_BATCH = config.get("sim_tau_sweep_batch_size", 128)
TAU_SWEEP_LR = config.get("sim_tau_sweep_lr", 1e-3)
TAU_SWEEP_CLIP = config.get("sim_tau_sweep_clip_norm", 5.0)
TAU_SWEEP_GENES = config.get("sim_tau_sweep_genes", SIM_GENES)
TAU_SWEEP_GUIDES = config.get("sim_tau_sweep_guides", SIM_GUIDES)
TAU_SWEEP_DAYS = config.get("sim_tau_sweep_days", SIM_DAYS)
TAU_SWEEP_REPS = config.get("sim_tau_sweep_reps", SIM_REPS)
TAU_SWEEP_KMAX = config.get("sim_tau_sweep_kmax", SIM_KMAX)
TAU_SWEEP_NTC_GUIDES = config.get("sim_tau_sweep_ntc_guides", SIM_NTC_GUIDES)
TAU_SWEEP_NTC_FRAC = config.get("sim_tau_sweep_ntc_frac", SIM_NTC_FRAC)

rule all:
    input:
        f"{OUT}/hits_ranked.csv"

rule fit_pyro_export:
    input:
        adata=config["adata_path"],
        guide_map=config["guide_map_csv"]
    output:
        summary=f"{OUT}/gene_summary_for_ash.csv"
    params:
        cfg="config.yaml"
    conda:
        "envs/pyro.yaml"
    threads: 4
    resources:
        gpu=1,
        mem_mb=64000
    shell:
        r"""
        mkdir -p {OUT}
        python scripts/fit_pyro_export.py \
            --config {params.cfg} \
            --adata {input.adata} \
            --guide-map {input.guide_map} \
            --out {output.summary}
        """

rule run_ash:
    input:
        summary=f"{OUT}/gene_summary_for_ash.csv"
    output:
        ash=f"{OUT}/gene_summary_ash_out.csv"
    conda:
        "envs/ash.yaml"
    threads: 1
    resources:
        mem_mb=16000
    shell:
        r"""
        Rscript scripts/run_ash.R {input.summary} {output.ash}
        """

rule rank_hits:
    input:
        ash=f"{OUT}/gene_summary_ash_out.csv"
    output:
        hits=f"{OUT}/hits_ranked.csv"
    params:
        cfg="config.yaml"
    conda:
        "envs/pyro.yaml"
    threads: 1
    resources:
        mem_mb=8000
    shell:
        r"""
        python scripts/rank_hits.py \
            --config {params.cfg} \
            --ash {input.ash} \
            --out {output.hits}
        """

rule diagnostics:
    input:
        adata=config["adata_path"],
        guide_map=config["guide_map_csv"]
    output:
        diag=f"{OUT}/diagnostics.json"
    params:
        cfg="config.yaml"
    conda:
        "envs/pyro.yaml"
    threads: 4
    resources:
        gpu=1,
        mem_mb=64000
    shell:
        r"""
        mkdir -p {OUT}
        python scripts/run_diagnostics.py \
            --config {params.cfg} \
            --adata {input.adata} \
            --guide-map {input.guide_map} \
            --out {output.diag}
        """

rule simulate_recovery:
    input:
        SIM_FORCE if SIM_ALWAYS_RUN else []
    output:
        adata=SIM_ADATA,
        guide_map=SIM_GUIDE_MAP,
        config=SIM_CONFIG,
        summary=SIM_EXPORT,
        recovery=SIM_RECOVERY
    params:
        out_dir=SIM_OUT
    conda:
        "envs/pyro.yaml"
    threads: 4
    resources:
        gpu=1,
        mem_mb=64000
    shell:
        r"""
        python scripts/simulate_recovery.py \
            --cells {SIM_CELLS} \
            --genes {SIM_GENES} \
            --guides {SIM_GUIDES} \
            --days {SIM_DAYS} \
            --reps {SIM_REPS} \
            --kmax {SIM_KMAX} \
            --ntc-guides {SIM_NTC_GUIDES} \
            --ntc-frac {SIM_NTC_FRAC} \
            --concentration {SIM_CONC} \
            --num-steps {SIM_NUM_STEPS} \
            --batch-size {SIM_BATCH} \
            --lr {SIM_LR} \
            --clip-norm {SIM_CLIP} \
            --num-draws {SIM_DRAWS} \
            --out-csv {output.recovery} \
            --write-anndata \
            --adata-out {output.adata} \
            --guide-map-out {output.guide_map} \
            --config-out {output.config} \
            --out-dir {params.out_dir} \
            --run-export \
            --export-out {output.summary} \
            --force
        """

if SIM_ALWAYS_RUN:
    rule sim_force:
        output:
            temp(SIM_FORCE)
        shell:
            r"""
            date +%s > {output}
            """

rule sim_run_ash:
    input:
        summary=SIM_EXPORT
    output:
        ash=SIM_ASH
    conda:
        "envs/ash.yaml"
    threads: 1
    resources:
        mem_mb=16000
    shell:
        r"""
        Rscript scripts/run_ash.R {input.summary} {output.ash}
        """

rule sim_rank_hits:
    input:
        ash=SIM_ASH,
        cfg=SIM_CONFIG
    output:
        hits=SIM_HITS
    conda:
        "envs/pyro.yaml"
    threads: 1
    resources:
        mem_mb=8000
    shell:
        r"""
        python scripts/rank_hits.py \
            --config {input.cfg} \
            --ash {input.ash} \
            --out {output.hits}
        """

rule sim_all:
    input:
        rules.simulate_recovery.output,
        SIM_HITS

rule sim_stress:
    output:
        summary=STRESS_SUMMARY,
        detail=STRESS_DETAIL
    params:
        cfg="config.yaml"
    conda:
        "envs/pyro.yaml"
    threads: 1
    resources:
        mem_mb=16000
    shell:
        r"""
        python scripts/simulate_stress.py \
            --config {params.cfg} \
            --out-summary {output.summary} \
            --out-detail {output.detail} \
            --force
        """

rule sim_prior_sweep:
    output:
        summary=PRIOR_SWEEP_SUMMARY,
        detail=PRIOR_SWEEP_DETAIL
    conda:
        "envs/pyro.yaml"
    threads: 1
    resources:
        mem_mb=16000
    shell:
        r"""
        python scripts/simulate_prior_sweep.py \
            --seed-data 0 \
            --seeds "{SWEEP_SEEDS}" \
            --s-time-values "{SWEEP_S_TIME}" \
            --s-guide-values "{SWEEP_S_GUIDE}" \
            --cells {SWEEP_CELLS} \
            --genes {SWEEP_GENES} \
            --guides {SWEEP_GUIDES} \
            --days {SWEEP_DAYS} \
            --reps {SWEEP_REPS} \
            --kmax {SWEEP_KMAX} \
            --ntc-guides {SWEEP_NTC_GUIDES} \
            --ntc-frac {SWEEP_NTC_FRAC} \
            --concentration {SWEEP_CONC} \
            --num-steps {SWEEP_STEPS} \
            --batch-size {SWEEP_BATCH} \
            --lr {SWEEP_LR} \
            --clip-norm {SWEEP_CLIP} \
            --num-draws {SWEEP_DRAWS} \
            --out-summary {output.summary} \
            --out-detail {output.detail} \
            --force
        """

rule sim_tau_sweep:
    output:
        summary=TAU_SWEEP_SUMMARY,
        detail=TAU_SWEEP_DETAIL
    conda:
        "envs/pyro.yaml"
    threads: 1
    resources:
        mem_mb=16000
    shell:
        r"""
        python scripts/simulate_tau_sweep.py \
            --seed-data 0 \
            --seeds "{TAU_SWEEP_SEEDS}" \
            --s-tau-values "{TAU_SWEEP_VALUES}" \
            --s-time {TAU_SWEEP_S_TIME} \
            --s-guide {TAU_SWEEP_S_GUIDE} \
            --cells {TAU_SWEEP_CELLS} \
            --genes {TAU_SWEEP_GENES} \
            --guides {TAU_SWEEP_GUIDES} \
            --days {TAU_SWEEP_DAYS} \
            --reps {TAU_SWEEP_REPS} \
            --kmax {TAU_SWEEP_KMAX} \
            --ntc-guides {TAU_SWEEP_NTC_GUIDES} \
            --ntc-frac {TAU_SWEEP_NTC_FRAC} \
            --concentration {TAU_SWEEP_CONC} \
            --num-steps {TAU_SWEEP_STEPS} \
            --batch-size {TAU_SWEEP_BATCH} \
            --lr {TAU_SWEEP_LR} \
            --clip-norm {TAU_SWEEP_CLIP} \
            --num-draws {TAU_SWEEP_DRAWS} \
            --out-summary {output.summary} \
            --out-detail {output.detail} \
            --force
        """
