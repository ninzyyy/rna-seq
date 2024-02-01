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




### Main ###

if __name__ == "__main__":

    file = uploaded_file = st.file_uploader("Upload a .csv file",
                                            type=['csv'],
                                            accept_multiple_files=False,
                                           )
    st.session_state['file'] = file


    # Load Data

    with st.spinner("Loading data..."):

        dir = os.path.join('data', 'E-GEOD-60052.csv')
        df = pd.read_csv(dir, index_col=0)

    # Data Preprocessing

    with st.spinner("Preprocessing data..."):

        df = df.loc[:,~df.columns.duplicated()].copy()
        df = df[df.columns[df.sum(axis=0) >= 10]]
        st.session_state['df'] = df

        meta_df = df['condition'].reset_index().rename(columns={'index':'sample'})
        meta_df = meta_df.set_index('sample').rename_axis(None)
        meta_df
        st.session_state['meta_df'] = meta_df

    # Differential Analysis

    with st.spinner("Perfoming differential analysis..."):
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
        st.session_state['res'] = res

        st.write(res)

    col1, col2, col3 = st.columns(3)

    with col1:
        padj = st.number_input("padj", value=0.05)
        st.session_state['padj'] = padj
    with col2:
        log2FoldChange = st.number_input("log2FoldChange", value = 0.05)
        st.session_state['log2FoldChange'] = log2FoldChange

    with col3:
        baseMean = st.number_input("base mean", value=20)
        st.session_state['baseMean'] = baseMean


    numGenes = st.number_input("Number of overexpressed and underexpressed genes to keep:", value=50)
    st.session_state['numGenes'] = numGenes

    if st.button("Generate Heatmap"):

        with st.spinner("Generating heatmap..."):

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
