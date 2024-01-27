### Imports ###

### Streamlit ###
import streamlit as st

# Housekeeping
import os

# Math
import pandas as pd
import numpy as np

#Visualization
import seaborn as sns

### RNA-Seq ###
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import gseapy as gp
from gseapy.plot import gseaplot

### Page Config ###
st.set_page_config(page_title=None,
                   page_icon="🌱",
                   layout="centered",
                   initial_sidebar_state="expanded",
                   menu_items=None)

### Functions ###

### Main ###

if __name__ == "__main__":

    # Load Data
    data_dir = os.path.join('data', 'E-GEOD-60052-raw-counts.tsv')
    meta_dir = os.path.join('data', 'E-GEOD-60052-experiment-design.tsv')
    df = pd.read_csv(data_dir, sep='\t')
    meta_df = pd.read_csv(meta_dir, sep='\t')

    # Data Preprocessing
    df['Gene Name'] = df['Gene Name'].fillna(df['Gene ID'])
    df.drop(columns=['Gene ID'], inplace=True)
    df = df.T
    df.columns = df.iloc[0]
    df = df[1:]
    df.rename_axis(None, axis=1,inplace=True)
    df = df.loc[:,~df.columns.duplicated()].copy()
    df = df[df.columns[df.sum(axis=0) >= 10]]

    meta_df = meta_df[['Run', 'Sample Characteristic[disease]']]
    meta_df.rename(columns={'Sample Characteristic[disease]':'condition'}, inplace=True)
    meta_df.set_index('Run', inplace=True)
    meta_df = meta_df.rename_axis(None)

    # Differential Analysis

    inference = DefaultInference(n_cpus=8)

    dds = DeseqDataSet(
        counts=df,
        metadata=meta_df,
        design_factors="condition",
        refit_cooks=True,
        inference=inference,
    )

    dds.deseq2()

    stat_res = DeseqStats(dds, inference=inference)
    stat_res.summary()
    res = stat_res.results_df

    st.write(res)

    col1, col2, col3 = st.columns(3)

    with col1:
        padj = st.number_input("padj", value=0.05)
    with col2:
        log2FoldChange = st.number_input("log2FoldChange", value = 0.05)
    with col3:
        baseMean = st.number_input("base mean", value=20)

    numGenes = st.number_input("Number of most and least expressed genes to keep:", value=50)

    if st.button("Generate Heatmap"):

        res = res[(res['padj'] < padj)]
        res = res[(abs(res['log2FoldChange']) > log2FoldChange)]
        res = res[(res['baseMean'] > baseMean)]
        res.sort_values(by=['log2FoldChange'], ascending=False, inplace=True)


        top_res = pd.concat([res.head(numGenes), res.tail(numGenes)])

        dds.layers['log1p'] = np.log1p(dds.layers['normed_counts'])
        dds_sigs = dds[:, top_res.index]

        diffexpr_df = pd.DataFrame(dds_sigs.layers['log1p'].T,
                        index=dds_sigs.var_names,
                        columns=dds_sigs.obs_names)

        st.pyplot(sns.clustermap(diffexpr_df,
                                z_score = 0,
                                cmap='RdBu_r'
                                )
                )
