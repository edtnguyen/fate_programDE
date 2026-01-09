import time

CONFIG_PATH = "config_iPSC.yaml"
configfile: CONFIG_PATH

OUT = config["out_dir"]

# Use pre-installed conda envs on Sherlock (Pyro + R/mash/ash).
PYRO_ENV = "/oak/stanford/groups/engreitz/Users/tri/envs/sc-dl-gpu"
R_ENV = "/oak/stanford/groups/engreitz/Users/tri/envs/scrnaR"
CPU_PARTITION = "engreitz"
GPU_PARTITION = "gpu"
CPU_TIME = "02:00:00"
GPU_TIME = "08:00:00"
# GPU requests are injected via profiles/slurm/config.yaml (slurm_extra=--gpus=1).
GPU_GPUS = 1
MEM_SMALL_MB = 8000
MEM_MED_MB = 16000
MEM_GPU_MB = 128000

MASH_MODES = config.get("mash_modes", ["conservative", "enriched"])
MASH_DEFAULT_MODE = config.get("mash_default_mode", "enriched")
if MASH_DEFAULT_MODE not in MASH_MODES:
    raise ValueError(f"mash_default_mode '{MASH_DEFAULT_MODE}' not in mash_modes {MASH_MODES}")

wildcard_constraints:
    mode="|".join(MASH_MODES)

GENE_MASH_IN = f"{OUT}/gene_daywise_for_mash.csv"
GUIDE_MASH_IN = f"{OUT}/guide_daywise_for_mash.csv"
MASH_GENE = f"{OUT}/mash_gene_{{mode}}.csv"
MASH_GUIDE = f"{OUT}/mash_guide_{{mode}}.csv"
AGG_GENE = f"{OUT}/gene_from_guide_mash_{{mode}}.csv"
AGG_GENE_DEFAULT = AGG_GENE.format(mode=MASH_DEFAULT_MODE)
MASH_GENE_DEFAULT = MASH_GENE.format(mode=MASH_DEFAULT_MODE)
MASH_COMPARE = f"{OUT}/mash_mode_comparison.csv"

SIM_OUT = config.get("sim_out_dir", "out_fate_pipeline_sim")
SIM_ADATA = config.get("sim_adata_path", "data/sim_adata.h5ad")
SIM_GUIDE_MAP = config.get("sim_guide_map_csv", "data/sim_guide_map.csv")
SIM_USE_EXISTING = config.get("sim_use_existing", False)
SIM_CONFIG = config.get("sim_config_path")
if SIM_CONFIG is None:
    SIM_CONFIG = CONFIG_PATH if SIM_USE_EXISTING else f"{SIM_OUT}/sim_config.yaml"

if not SIM_USE_EXISTING:
    ruleorder: fit_pyro_export > simulate_recovery
SIM_RECOVERY = config.get("sim_recovery_csv", f"{SIM_OUT}/sim_recovery.csv")
SIM_GENE_EXPORT = config.get("sim_gene_export_csv", f"{SIM_OUT}/gene_daywise_for_mash.csv")
SIM_GUIDE_EXPORT = config.get("sim_guide_export_csv", f"{SIM_OUT}/guide_daywise_for_mash.csv")
SIM_EXPORT_ASH = config.get("sim_export_ash_csv", f"{SIM_OUT}/gene_summary_for_ash.csv")
SIM_MASH_GENE = f"{SIM_OUT}/mash_gene_{{mode}}.csv"
SIM_MASH_GUIDE = f"{SIM_OUT}/mash_guide_{{mode}}.csv"
SIM_AGG_GENE = f"{SIM_OUT}/gene_from_guide_mash_{{mode}}.csv"
SIM_AGG_GENE_DEFAULT = SIM_AGG_GENE.format(mode=MASH_DEFAULT_MODE)
SIM_MASH_COMPARE = config.get("sim_mash_compare_csv", f"{SIM_OUT}/mash_mode_comparison.csv")
SIM_ASH = config.get("sim_ash_csv", f"{SIM_OUT}/gene_summary_ash_out.csv")
SIM_HITS = config.get("sim_hits_csv", f"{SIM_OUT}/hits_ranked.csv")
SIM_META = config.get("sim_metadata_path", f"{SIM_OUT}/sim_metadata.yaml")

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
SIM_ALWAYS_RUN = config.get("sim_always_run", True) and not SIM_USE_EXISTING
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

PERM_OUT = config.get("perm_out_dir", f"{OUT}_perm")
PERM_INPUT_ADATA = config.get("perm_adata_path") or config.get("adata_path")
PERM_ADATA = f"{PERM_OUT}/adata_perm.h5ad"
PERM_GUIDE_MAP = config.get("perm_guide_map_csv", config["guide_map_csv"])
PERM_GENE_EXPORT = f"{PERM_OUT}/gene_daywise_for_mash.csv"
PERM_GUIDE_EXPORT = f"{PERM_OUT}/guide_daywise_for_mash.csv"
PERM_MODE = "conservative"
PERM_MASH_GENE = f"{PERM_OUT}/mash_gene_{PERM_MODE}.csv"
PERM_MASH_GUIDE = f"{PERM_OUT}/mash_guide_{PERM_MODE}.csv"
PERM_AGG_GENE = f"{PERM_OUT}/gene_from_guide_mash_{PERM_MODE}.csv"
PERM_HITS = f"{PERM_OUT}/hits_ranked.csv"
PERM_SUMMARY = f"{PERM_OUT}/perm_summary.json"

rule all:
    input:
        f"{OUT}/hits_ranked.csv",
        MASH_COMPARE,
        expand(MASH_GENE, mode=MASH_MODES)

rule fit_pyro_export:
    input:
        adata=config["adata_path"],
        guide_map=config["guide_map_csv"]
    output:
        gene=GENE_MASH_IN,
        guide=GUIDE_MASH_IN
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 8
    resources:
        slurm_partition=GPU_PARTITION,
        partition=GPU_PARTITION,
        time=GPU_TIME,
        gpus=GPU_GPUS,
        mem_mb=MEM_GPU_MB
    shell:
        r"""
        mkdir -p {OUT}
        python scripts/fit_pyro_export.py \
            --config {params.cfg} \
            --adata {input.adata} \
            --guide-map {input.guide_map} \
            --out-gene {output.gene} \
            --out-guide {output.guide}
        """

rule run_mash_gene:
    input:
        gene=GENE_MASH_IN
    output:
        mash=MASH_GENE
    params:
        cfg=CONFIG_PATH
    conda:
        R_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        Rscript scripts/run_mashr_two_mode.R {input.gene} {output.mash} {wildcards.mode} {params.cfg}
        """

rule run_mash_guide:
    input:
        guide=GUIDE_MASH_IN
    output:
        mash=MASH_GUIDE
    params:
        cfg=CONFIG_PATH
    conda:
        R_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        Rscript scripts/run_mashr_two_mode.R {input.guide} {output.mash} {wildcards.mode} {params.cfg}
        """

rule aggregate_gene:
    input:
        mash=MASH_GUIDE
    output:
        agg=AGG_GENE
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/aggregate_guides_to_genes.py \
            --in-mash-guide {input.mash} \
            --out-gene {output.agg} \
            --config {params.cfg}
        """

rule compare_modes:
    input:
        expand(AGG_GENE, mode=MASH_MODES)
    output:
        compare=MASH_COMPARE
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/compare_mash_modes.py \
            --inputs {input} \
            --out {output.compare}
        """

rule run_ash:
    input:
        summary=f"{OUT}/gene_summary_for_ash.csv"
    output:
        ash=f"{OUT}/gene_summary_ash_out.csv"
    conda:
        R_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        Rscript scripts/run_ash.R {input.summary} {output.ash}
        """

rule rank_hits:
    input:
        mash=MASH_GENE_DEFAULT
    output:
        hits=f"{OUT}/hits_ranked.csv"
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/rank_hits.py \
            --config {params.cfg} \
            --mash {input.mash} \
            --out {output.hits}
        """

rule diagnostics:
    input:
        adata=config["adata_path"],
        guide_map=config["guide_map_csv"]
    output:
        diag=f"{OUT}/diagnostics.json"
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 4
    resources:
        slurm_partition=GPU_PARTITION,
        partition=GPU_PARTITION,
        time=GPU_TIME,
        gpus=GPU_GPUS,
        mem_mb=MEM_GPU_MB
    shell:
        r"""
        mkdir -p {OUT}
        python scripts/run_diagnostics.py \
            --config {params.cfg} \
            --adata {input.adata} \
            --guide-map {input.guide_map} \
            --out {output.diag}
        """

rule permute_guides:
    input:
        adata=PERM_INPUT_ADATA
    output:
        adata=PERM_ADATA
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        python scripts/make_permuted_guides.py \
            --adata-in {input.adata} \
            --adata-out {output.adata} \
            --config {params.cfg}
        """

rule perm_fit_export:
    input:
        adata=PERM_ADATA,
        guide_map=PERM_GUIDE_MAP
    output:
        gene=PERM_GENE_EXPORT,
        guide=PERM_GUIDE_EXPORT
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 4
    resources:
        slurm_partition=GPU_PARTITION,
        partition=GPU_PARTITION,
        time=GPU_TIME,
        gpus=GPU_GPUS,
        mem_mb=MEM_GPU_MB
    shell:
        r"""
        mkdir -p {PERM_OUT}
        python scripts/fit_pyro_export.py \
            --config {params.cfg} \
            --adata {input.adata} \
            --guide-map {input.guide_map} \
            --out-gene {output.gene} \
            --out-guide {output.guide}
        """

rule perm_run_mash_gene:
    input:
        gene=PERM_GENE_EXPORT
    output:
        mash=PERM_MASH_GENE
    params:
        cfg=CONFIG_PATH
    conda:
        R_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        Rscript scripts/run_mashr_two_mode.R {input.gene} {output.mash} {PERM_MODE} {params.cfg}
        """

rule perm_run_mash_guide:
    input:
        guide=PERM_GUIDE_EXPORT
    output:
        mash=PERM_MASH_GUIDE
    params:
        cfg=CONFIG_PATH
    conda:
        R_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        Rscript scripts/run_mashr_two_mode.R {input.guide} {output.mash} {PERM_MODE} {params.cfg}
        """

rule perm_aggregate_gene:
    input:
        mash=PERM_MASH_GUIDE
    output:
        agg=PERM_AGG_GENE
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/aggregate_guides_to_genes.py \
            --in-mash-guide {input.mash} \
            --out-gene {output.agg} \
            --config {params.cfg}
        """

rule perm_rank_hits:
    input:
        mash=PERM_MASH_GENE
    output:
        hits=PERM_HITS
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/rank_hits.py \
            --config {params.cfg} \
            --mash {input.mash} \
            --out {output.hits}
        """

rule perm_summary:
    input:
        mash=PERM_MASH_GENE
    output:
        summary=PERM_SUMMARY
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/perm_summary.py \
            --mash {input.mash} \
            --config {params.cfg} \
            --out {output.summary}
        """

rule perm_all:
    input:
        PERM_HITS,
        PERM_SUMMARY

if not SIM_USE_EXISTING:
    rule simulate_recovery:
        input:
            SIM_FORCE if SIM_ALWAYS_RUN else []
        output:
            adata=SIM_ADATA,
            guide_map=SIM_GUIDE_MAP,
            config=SIM_CONFIG,
            gene_summary=SIM_GENE_EXPORT,
            guide_summary=SIM_GUIDE_EXPORT,
            recovery=SIM_RECOVERY,
            metadata=SIM_META
        params:
            out_dir=SIM_OUT
        conda:
            PYRO_ENV
        threads: 4
        resources:
            slurm_partition=GPU_PARTITION,
            partition=GPU_PARTITION,
            time=GPU_TIME,
            gpus=GPU_GPUS,
            mem_mb=MEM_GPU_MB
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
                --metadata-out {output.metadata} \
                --out-dir {params.out_dir} \
                --run-export \
                --export-out {output.gene_summary} \
                --export-guide-out {output.guide_summary} \
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
else:
    rule sim_export_existing:
        input:
            adata=SIM_ADATA,
            guide_map=SIM_GUIDE_MAP,
            cfg=SIM_CONFIG
        output:
            gene=SIM_GENE_EXPORT,
            guide=SIM_GUIDE_EXPORT
        conda:
            PYRO_ENV
        threads: 1
        resources:
            slurm_partition=CPU_PARTITION,
            partition=CPU_PARTITION,
            time=CPU_TIME,
            mem_mb=MEM_MED_MB
        shell:
            r"""
            python scripts/fit_pyro_export.py \
                --config {input.cfg} \
                --adata {input.adata} \
                --guide-map {input.guide_map} \
                --out-gene {output.gene} \
                --out-guide {output.guide}
            """

rule sim_run_ash:
    input:
        summary=SIM_EXPORT_ASH
    output:
        ash=SIM_ASH
    conda:
        R_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        Rscript scripts/run_ash.R {input.summary} {output.ash}
        """

rule sim_run_mash_gene:
    input:
        gene=SIM_GENE_EXPORT
    output:
        mash=SIM_MASH_GENE
    params:
        cfg=SIM_CONFIG
    conda:
        R_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        Rscript scripts/run_mashr_two_mode.R {input.gene} {output.mash} {wildcards.mode} {params.cfg}
        """

rule sim_run_mash_guide:
    input:
        guide=SIM_GUIDE_EXPORT
    output:
        mash=SIM_MASH_GUIDE
    params:
        cfg=SIM_CONFIG
    conda:
        R_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
    shell:
        r"""
        Rscript scripts/run_mashr_two_mode.R {input.guide} {output.mash} {wildcards.mode} {params.cfg}
        """

rule sim_aggregate_gene:
    input:
        mash=SIM_MASH_GUIDE
    output:
        agg=SIM_AGG_GENE
    params:
        cfg=SIM_CONFIG
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/aggregate_guides_to_genes.py \
            --in-mash-guide {input.mash} \
            --out-gene {output.agg} \
            --config {params.cfg}
        """

rule sim_compare_modes:
    input:
        expand(SIM_AGG_GENE, mode=MASH_MODES)
    output:
        compare=SIM_MASH_COMPARE
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/compare_mash_modes.py \
            --inputs {input} \
            --out {output.compare}
        """

rule sim_rank_hits:
    input:
        mash=SIM_MASH_GENE.format(mode=MASH_DEFAULT_MODE),
        cfg=SIM_CONFIG
    output:
        hits=SIM_HITS
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_SMALL_MB
    shell:
        r"""
        python scripts/rank_hits.py \
            --config {input.cfg} \
            --mash {input.mash} \
            --out {output.hits}
        """

if SIM_USE_EXISTING:
    SIM_BASE_INPUTS = [SIM_ADATA, SIM_GUIDE_MAP, SIM_CONFIG, SIM_GENE_EXPORT, SIM_GUIDE_EXPORT]
else:
    SIM_BASE_INPUTS = rules.simulate_recovery.output

rule sim_all:
    input:
        SIM_BASE_INPUTS,
        SIM_HITS,
        SIM_MASH_COMPARE,
        expand(SIM_MASH_GENE, mode=MASH_MODES)

rule sim_stress:
    output:
        summary=STRESS_SUMMARY,
        detail=STRESS_DETAIL
    params:
        cfg=CONFIG_PATH
    conda:
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
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
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
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
        PYRO_ENV
    threads: 1
    resources:
        slurm_partition=CPU_PARTITION,
        partition=CPU_PARTITION,
        time=CPU_TIME,
        mem_mb=MEM_MED_MB
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
