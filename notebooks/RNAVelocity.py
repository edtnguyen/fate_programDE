#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")


# In[3]:


import os

cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
ntasks = int(os.environ.get("SLURM_NTASKS", 1))
total_cpus = cpus_per_task * ntasks

print("cpus_per_task:", cpus_per_task)
print("ntasks:", ntasks)
print("total_cpus_allocated:", total_cpus)


# In[4]:


import gc
import glob
import os
import sys

import anndata as ad

# In[5]:
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse
import scvelo as scv
from dask.diagnostics import ProgressBar

ProgressBar().register()


# In[6]:


working_dir = "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/240307_eTN5_iPSC-EC_DA_hits/WT/"
tscp_paths = glob.glob(f"{working_dir}sub*/process/tscp_assignment.csv.gz")
subs = [*range(1, 4)]


# In[7]:


sc.settings.verbosity = 1  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(
    dpi=100, fontsize=10, dpi_save=300, figsize=(5, 4), format="png"
)
sc.settings.figdir = "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/240307_eTN5_iPSC-EC_DA_hits/WT_combined/"


# ### Open tscp files

# In[52]:


import os
import subprocess

tscp_unzipped = []
# Unzip tscp file, if not already done

for path in tscp_paths:
    out_path = path.replace(".gz", "")
    if not os.path.exists(out_path):
        subprocess.run(["pigz", "-k", "-d", path], check=True)
    tscp_unzipped.append(out_path)

print(tscp_unzipped)


# ### Define the function for generating splice matrices

# In[53]:


import pandas as pd
from pandas.api.types import is_string_dtype


def _sanitize_string_columns(df):
    for col in df.columns:
        # catches string[python], string[pyarrow], etc.
        if is_string_dtype(df[col].dtype) or isinstance(df[col].dtype, pd.StringDtype):
            # convert to plain Python-string / object array
            df[col] = df[col].astype("object")
            # if you prefer categories (usually smaller on disk):
            # df[col] = df[col].astype("category")
    return df


# In[54]:


def generate_splice_matrices(tscp_path, cutoff, adata_path):
    print(f"Reading in {tscp_path}")
    tscp_assign_df = dd.read_csv(tscp_path, blocksize="800MB")

    tscp_assign_df = tscp_assign_df.compute()
    cell_tscp_cnts = tscp_assign_df.groupby("bc_wells").size()
    cell_tscp_cnts = cell_tscp_cnts[cell_tscp_cnts >= cutoff]
    filtered_cell_dict = dict(zip(cell_tscp_cnts.index, np.zeros(len(cell_tscp_cnts))))

    def check_filtered_cell(cell_ind):
        try:
            filtered_cell_dict[cell_ind]
        except:
            return False
        else:
            return True

    genes = tscp_assign_df.gene_name.unique()
    bcs = cell_tscp_cnts.index
    gene_dict = dict(zip(genes, range(len(genes))))
    barcode_dict = dict(zip(bcs, range(len(bcs))))
    reads_to_keep = tscp_assign_df.bc_wells.apply(check_filtered_cell)

    print("\nFiltering tscp file..")
    tscp_assign_df_filt = tscp_assign_df[reads_to_keep]
    tscp_assign_df_filt["cell_index"] = tscp_assign_df_filt.bc_wells.apply(
        lambda s: barcode_dict[s]
    )
    tscp_assign_df_filt["gene_index"] = tscp_assign_df_filt.gene_name.apply(
        lambda s: gene_dict[s]
    )
    print("Done:", tscp_assign_df_filt.shape)

    rcv = (
        tscp_assign_df_filt.query("exonic")
        .groupby(["cell_index", "gene_index"])
        .size()
        .reset_index()
        .values
    )
    rows = list(rcv[:, 0]) + [len(barcode_dict) - 1]
    cols = list(rcv[:, 1]) + [len(genes) - 1]
    vals = list(rcv[:, 2]) + [0]
    X_exonic = scipy.sparse.csr_matrix((vals, (rows, cols)))

    rcv = (
        tscp_assign_df_filt.query("~exonic")
        .groupby(["cell_index", "gene_index"])
        .size()
        .reset_index()
        .values
    )
    rows = list(rcv[:, 0]) + [len(barcode_dict) - 1]
    cols = list(rcv[:, 1]) + [len(genes) - 1]
    vals = list(rcv[:, 2]) + [0]
    X_intronic = scipy.sparse.csr_matrix((vals, (rows, cols)))

    X = X_exonic + X_intronic
    adata = scv.AnnData(
        X=X,
    )

    x_row, x_col = adata.shape
    adata.obs = pd.DataFrame({"barcodes": bcs}, index=bcs)
    adata.var = pd.DataFrame({"gene": genes, "gene_name": genes})
    adata.var.index = genes

    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.layers["spliced"] = X_exonic
    adata.layers["unspliced"] = X_intronic
    scv.utils.show_proportions(adata)

    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)

    adata.obs = _sanitize_string_columns(adata.obs)
    adata.var = _sanitize_string_columns(adata.var)

    adata.write(f"{adata_path}sub{subs[i]}_adata.h5ad")
    print(f"Saved anndata to {adata_path}{subs[i]}_adata.h5ad.\n")
    return adata


# ### Generate splice matrices and concatenate anndata objects

# In[55]:


ad_list_sp = []
tscp_cutoffs = [195.9763, 152.7348, 217.3041]
for i in range(len(tscp_paths)):
    ad_list_sp.append(
        generate_splice_matrices(tscp_unzipped[i], tscp_cutoffs[i], f"{working_dir}")
    )


# In[56]:


# Concatenate objects
ad_splice = ad.concat(ad_list_sp, keys=subs, index_unique="__s")


# ### Add sample information (cell metadata) to anndata object

# In[57]:


meta_path = "/oak/stanford/groups/engreitz/Users/tri/Perturb-Seq/analysis/240307_eTN5_iPSC-EC_DA_hits/WT_combined/all-sample/DGE_filtered/cell_metadata.csv"
comb_meta = pd.read_csv(meta_path, index_col=0)

meta_common = ad_splice.obs.join(other=comb_meta, on=ad_splice.obs.index, how="left")
ad_splice.obs = meta_common


# ### Save/Read ad_splice object to/from h5ad

# In[8]:


# ad_splice.write(f"{working_dir}spliced_adata.h5ad")

ad_splice = ad.read_h5ad(f"{working_dir}spliced_adata.h5ad")


# In[9]:


sc.pp.filter_cells(ad_splice, min_genes=600)
sc.pp.filter_genes(ad_splice, min_cells=5)


# In[10]:


# 1. Create a boolean mask where True means the value is NOT NaN
mask = ad_splice.obs["sample"].notna()

# 2. Subset the AnnData object using the mask
ad_splice = ad_splice[mask].copy()
ad_splice.shape


# ### Run through cell preprocessing and clustering

# In[11]:


scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)

scv.pp.filter_and_normalize(
    ad_splice,
    #    min_shared_counts=5,
    n_top_genes=3000,
    subset_highly_variable=False,
)

sc.tl.pca(ad_splice)
sc.pp.neighbors(ad_splice, n_pcs=30, n_neighbors=30, random_state=0)
scv.pp.moments(ad_splice, n_pcs=None, n_neighbors=None)


# In[12]:


ad_splice.shape


# In[13]:


# --- velocity + velocity graph with scvelo ---
scv.tl.recover_dynamics(ad_splice, n_jobs=cpus_per_task)


# In[ ]:


scv.tl.velocity(ad_splice, mode="dynamical")


# In[ ]:


scv.tl.velocity_graph(ad_splice)


# ### Add a "day" column for moscot

# In[ ]:


# define mapping
sample_to_day = {
    "D0": 0,
    "sample_D1": 1,
    "sample_D2": 2,
    "sample_D3": 3,
}

ad_splice.obs["day"] = ad_splice.obs["sample"].map(sample_to_day)
ad_splice.obs["day"] = ad_splice.obs["day"].astype(float).astype("category")


# ### Normalize and PCA

# In[ ]:


# 1) Normalization / log
sc.pp.normalize_total(ad_splice, target_sum=1e4)
sc.pp.log1p(ad_splice)

# 2) Mark HVGs, but DO NOT subset
sc.pp.highly_variable_genes(ad_splice, n_top_genes=3000, flavor="seurat")

# 3) Scale + PCA using only HVGs for the embedding
sc.pp.scale(ad_splice, max_value=10)
sc.tl.pca(ad_splice, n_comps=50, use_highly_variable=True)

sc.pp.neighbors(ad_splice, use_rep="X_pca", random_state=0)


# ### Save anndata object to h5ad file

# In[ ]:


# Save anndata object to h5ad file
# ad_splice.write(f"{working_dir}scvelo_adata.h5ad")

# ad_splice = ad.read_h5ad(f"{working_dir}scvelo_adata.h5ad")

ad_splice.write(f"{working_dir}premoscot_adata.h5ad")


# ### Plot results

# In[ ]:


scv.pl.proportions(ad_splice)


# In[ ]:


ad_splice


# In[ ]:


# --- embedding + clustering: use scanpy, not scvelo ---
sc.tl.umap(ad_splice)  # uses existing neighbors from moments
sc.tl.leiden(ad_splice, resolution=1.0)  # or sc.tl.louvain(ad_splice, resolution=1.0)


# In[ ]:


palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

sc.pl.umap(ad_splice, color=["sample"], color_map="viridis")


# In[ ]:


sc.tl.leiden(ad_splice, resolution=0.75)


# In[ ]:


# Set figure parameters
scv.settings.presenter_view = True  # set max width size for presenter view
scv.set_figure_params("scvelo")  # for beautified visualization
scv.set_figure_params(
    figsize=(6, 4),
    dpi=150,
    format="png",
    dpi_save=300,
    transparent=False,
    facecolor="white",
    fontsize=8,
)

import seaborn as sns

cluster_colors = sns.color_palette("hls", 28)
scv.pl.velocity_embedding_stream(
    ad_splice,
    basis="umap",
    color="leiden",
    palette=cluster_colors,
    size=30,
    alpha=0.8,
    fontsize=10,
    save="stream_embedding",
)


# In[77]:


# get how many sample categories you have
cats = ad_splice.obs["sample"].astype("category").cat.categories

# handcrafted palette: blue, orange, purple, brown, cyan, grey (no red/green)
palette = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # cyan
    "#7f7f7f",  # grey
][: len(cats)]

# attach palette to 'sample'
ad_splice.uns["sample_colors"] = palette

sc.pl.umap(
    ad_splice,
    color="sample",
    # no need for palette=... if you set uns["sample_colors"]
)


# In[78]:


sc.set_figure_params(
    figsize=(6, 4),
    dpi=150,
    format="png",
    dpi_save=300,
    transparent=False,
    facecolor="white",
    fontsize=14,
)
sc.pl.umap(
    ad_splice,
    color=["HAND1", "FLI1", "KDR", "ETV2", "CDH5", "CNN1"],
    ncols=2,
    color_map="Reds",
    legend_fontsize=8,
)


# ### Set up the VelocityKernel

# In[79]:


import cellrank as cr

vk = cr.kernels.VelocityKernel(ad_splice)


# In[80]:


vk.compute_transition_matrix()


# ### Combine with gene expression similarity

# In[81]:


ck = cr.kernels.ConnectivityKernel(ad_splice)
ck.compute_transition_matrix()

combined_kernel = 0.8 * vk + 0.2 * ck


# In[82]:


print(combined_kernel)


# In[83]:


vk.plot_projection(basis="umap", color="leiden")


# In[84]:


sc.pl.umap(ad_splice, color="leiden", legend_loc="on data")


# In[86]:


vk.plot_random_walks(start_ixs={"leiden": "1"}, max_iter=200, seed=0)


# In[ ]:
