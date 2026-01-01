configfile: "config.yaml"

OUT = config["out_dir"]

SIM_OUT = config.get("sim_out_dir", "out_fate_pipeline_sim")
SIM_ADATA = config.get("sim_adata_path", "data/sim_adata.h5ad")
SIM_GUIDE_MAP = config.get("sim_guide_map_csv", "data/sim_guide_map.csv")
SIM_CONFIG = config.get("sim_config_path", f"{SIM_OUT}/sim_config.yaml")
SIM_RECOVERY = config.get("sim_recovery_csv", f"{SIM_OUT}/sim_recovery.csv")
SIM_EXPORT = config.get("sim_export_csv", f"{SIM_OUT}/gene_summary_for_ash.csv")

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
