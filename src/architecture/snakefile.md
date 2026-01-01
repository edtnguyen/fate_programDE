## 1) `Snakefile` (with per-rule conda envs)

```python
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
    shell:
        r"""
        python scripts/rank_hits.py \
            --config {params.cfg} \
            --ash {input.ash} \
            --out {output.hits}
        """
```

Run with:

```bash
snakemake -j 1 --use-conda
```

(Use `-j` larger if your cluster setup allows it.)

---

## 2) `envs/pyro.yaml`

```yaml
name: fate-pyro
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scipy
  - pyyaml
  - anndata
  - scanpy
  - pytorch>=2.2
  - pytorch-cuda=12.1  # if you have CUDA; remove if CPU-only
  - pip:
      - pyro-ppl>=1.9
```

Notes:

* If your machine is CPU-only, remove `pytorch-cuda=12.1`.
* On some systems, you may prefer `conda-forge::pytorch` instead of `pytorch` channel; adjust if solver complains.

---

## 3) `envs/ash.yaml`

```yaml
name: fate-ash
channels:
  - conda-forge
dependencies:
  - r-base>=4.3
  - r-data.table
  - r-ashr
```

This keeps R clean and avoids cross-language package conflicts.

---

## 4) Directory layout (suggested)

```
project/
  Snakefile
  config.yaml
  envs/
    pyro.yaml
    ash.yaml
  scripts/
    fit_pyro_export.py
    run_ash.R
    rank_hits.py
  data/
    adata.h5ad
    guide_map.csv
  out_fate_pipeline/
    (generated)
```

---

## 5) One small Snakemake tip for clusters/GPU

If you’re on a scheduler, you can keep the env split and add resource tags, e.g.:

```python
resources:
    gpu=1
```


