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
                   initial_sidebar_state="auto",
                   menu_items=None)

### Functions ###

@st.cache_data(show_spinner=False)
def preprocess_datasets(countData, colData):

    # Preprocess countData to pyDESEQ2 format
    if 'Gene Name' in countData.columns:
        gene_names = countData['Gene Name']
        countData.drop(columns=['Gene Name'], inplace=True)

    countData = countData.T
    countData.columns = countData.iloc[0]
    countData = countData[1:]
    countData.columns.name = None

    # Preprocess colData to pyDESEQ2 format
    colData = colData[['Run', 'Sample Characteristic[disease]']]
    colData = colData.rename(columns={'Sample Characteristic[disease]':'condition'})
    colData = colData.set_index('Run')
    colData.index.name = None

    return countData, colData

@st.cache_data(show_spinner=False)
def perform_diff_analysis(countData, colData):

    inference = DefaultInference(n_cpus=8)

    dds = DeseqDataSet(
        counts=countData,
        metadata=colData,
        design_factors="condition",
        refit_cooks=True,
        inference=inference,
    )

    dds.deseq2()
    stat_res = DeseqStats(dds, inference=inference)
    stat_res.summary()

    return stat_res.results_df, dds

### Main ###

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if __name__ == "__main__":

    if not st.session_state['data_loaded']:

        with st.spinner("Loading data..."):

            dir = os.path.join('data', 'E-GEOD-60052-raw-counts.tsv') # countData
            df = pd.read_csv(dir, sep='\t')
            st.session_state.df = df

            meta_dir = os.path.join('data', 'E-GEOD-60052-experiment-design.tsv') #colData
            meta_df = pd.read_csv(meta_dir, sep='\t')
            st.session_state.meta_df = meta_df

        with st.spinner("Preprocessing data..."):

            countData, colData = preprocess_datasets(df, meta_df)
            st.session_state.countData = countData
            st.session_state.colData = colData
            st.session_state.data_loaded = True

    # Differential Analysis
    if not st.session_state['analysis_done']:
        with st.spinner("Performing differential analysis..."):

            res, dds = perform_diff_analysis(st.session_state['countData'], st.session_state['colData'])
            st.session_state.res = res
            st.session_state.dds = dds
            st.session_state.analysis_done = True

    # Input parameters for generating the heatmap
    col1, col2, col3 = st.columns(3)
    with col1:
        padj = st.number_input("padj", value=0.05)
    with col2:
        log2FoldChange = st.number_input("log2FoldChange", value = 0.05)
    with col3:
        baseMean = st.number_input("base mean", value=20)
    numGenes = st.number_input("Number of overexpressed and underexpressed genes to keep:", value=50)

    if st.button("Generate Heatmap"):

        with st.spinner("Generating heatmap..."):

            filtered_res = st.session_state.res[(st.session_state.res['padj'] < padj) &
                                            (abs(st.session_state.res['log2FoldChange']) > log2FoldChange) &
                                            (st.session_state.res['baseMean'] > baseMean)]

            filtered_res.sort_values(by=['log2FoldChange'], ascending=False, inplace=True)

            top_res = pd.concat([filtered_res.head(numGenes), filtered_res.tail(numGenes)])

            st.session_state.dds.layers['log1p'] = np.log1p(st.session_state.dds.layers['normed_counts'])

            dds_sigs = st.session_state.dds[:, top_res.index]

            diffexpr_df = pd.DataFrame(dds_sigs.layers['log1p'].T,
                                    index=dds_sigs.var_names,
                                    columns=dds_sigs.obs_names)

            st.pyplot(sns.clustermap(diffexpr_df, z_score=0, cmap='RdBu_r'))
