configfile: "config.yaml"

OUT = config["out_dir"]

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
