### Imports ###

### Streamlit ###
import streamlit as st

# Housekeeping
import os, re, requests


# Math
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns

### RNA-Seq ###
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import gseapy as gp
from gseapy.plot import gseaplot

### Page Config ###
st.set_page_config(
    page_title=None,
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)

### Functions ###


@st.cache_data(show_spinner=False)
def preprocess_datasets(countData, colData):

    # Preprocess countData to pyDESEQ2 format
    if "Gene Name" in countData.columns:
        gene_names = countData["Gene Name"]
        countData.drop(columns=["Gene Name"], inplace=True)

    countData = countData.T
    countData.columns = countData.iloc[0]
    countData = countData[1:]
    countData.columns.name = None

    # Preprocess colData to pyDESEQ2 format
    colData = colData[["Run", "Sample Characteristic[disease]"]]
    colData = colData.rename(columns={"Sample Characteristic[disease]": "condition"})
    colData = colData.set_index("Run")
    colData.index.name = None

    return countData, colData


@st.cache_data(show_spinner=False)
def process_files(uploaded_files):
    df = meta_df = None

    for file in uploaded_files:

        if "countData" in file.name:
            df = pd.read_csv(file)

        elif "colData" in file.name:
            meta_df = pd.read_csv(file)

    if df is not None and meta_df is not None:
        return preprocess_datasets(df, meta_df)

    else:
        st.error("Please upload both countData and colData files.")
        return None, None


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


@st.cache_data(show_spinner=False)
def get_gene_name(ensembl_id):
    try:
        url = f"https://mygene.info/v3/gene/{ensembl_id}"
        response = requests.get(url)
        data = response.json()

        # Check if 'symbol' is in the data and not None
        if "symbol" in data and data["symbol"] is not None:
            return data["symbol"]
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


### Main ###

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

uploaded_files = st.file_uploader(
    "Upload a countData and a colData CSV file:", accept_multiple_files=True
)

if st.button("Upload Files") and len(uploaded_files) == 2:

    countData, colData = process_files(uploaded_files)

    if countData is not None and colData is not None:
        res, dds = perform_diff_analysis(countData, colData)

        st.session_state["countData"] = countData
        st.session_state["colData"] = colData
        st.session_state["res"] = res
        st.session_state["dds"] = dds
        st.session_state.analysis_done = True

st.write("or click the button below to view a preview:")
if st.button("Demo"):

    with st.spinner("Loading and preprocessing data..."):

        dir = os.path.join("data", "E-GEOD-60052-raw-counts.tsv")
        df = pd.read_csv(dir, sep="\t")

        meta_dir = os.path.join("data", "E-GEOD-60052-experiment-design.tsv")
        meta_df = pd.read_csv(meta_dir, sep="\t")

        countData, colData = preprocess_datasets(df, meta_df)

    with st.spinner("Performing differential analysis..."):
        res, dds = perform_diff_analysis(countData, colData)

    # Update session state immediately after processing
    st.session_state["countData"] = countData
    st.session_state["colData"] = colData
    st.session_state["res"] = res
    st.session_state["dds"] = dds
    # Setting this state variable here ensures the rest of your script knows the analysis is complete.
    st.session_state.analysis_done = True

st.divider()

if st.session_state.analysis_done:

    # Input parameters for generating the heatmap
    st.markdown(
        '<span style="text-decoration: underline;">Threshold Criteria for Gene Expression Analysis</span>',
        unsafe_allow_html=True,
    )
    # st.write("Threshold Criteria for Gene Expression Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        padj = st.number_input("padj", value=0.05)
    with col2:
        log2FoldChange = st.number_input("log2FoldChange", value=0.05)
    with col3:
        baseMean = st.number_input("base mean", value=20)
    numGenes = st.number_input(
        "Number of overexpressed and underexpressed genes to keep:", value=50
    )

    if st.button("Generate Heatmap"):
        st.divider()
        with st.spinner("Generating heatmap..."):

            filtered_res = st.session_state.res[
                (st.session_state.res["padj"] < padj)
                & (abs(st.session_state.res["log2FoldChange"]) > log2FoldChange)
                & (st.session_state.res["baseMean"] > baseMean)
            ]

            filtered_res.sort_values(
                by=["log2FoldChange"],
                ascending=False,
                inplace=True,
            )

            top_res = pd.concat(
                [filtered_res.head(numGenes), filtered_res.tail(numGenes)]
            )

            st.session_state.dds.layers["log1p"] = np.log1p(
                st.session_state.dds.layers["normed_counts"]
            )

            dds_sigs = st.session_state.dds[:, top_res.index]

            diffexpr_df = pd.DataFrame(
                dds_sigs.layers["log1p"].T,
                index=dds_sigs.var_names,
                columns=dds_sigs.obs_names,
            )

            st.markdown(
                '<span style="text-decoration: underline;">Heatmap of Gene Expression Levels Across Samples</span>',
                unsafe_allow_html=True,
            )
            st.pyplot(sns.clustermap(diffexpr_df, z_score=0, cmap="RdBu_r"))
            st.divider()

        with st.spinner("Getting Gene Names for Gene IDs..."):
            top_res["Gene Name"] = top_res.index.map(get_gene_name)
            top_res["padj"] = top_res["padj"].apply(lambda x: format(float(x), ".5f"))
            top_res["log2FoldChange"] = top_res["log2FoldChange"].apply(
                lambda x: format(float(x), ".5f")
            )
            top_res.index.name = "Gene ID"

        st.markdown(
            '<span style="text-decoration: underline;">Table of Differential Gene Expression Analysis Results</span>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            top_res[["Gene Name", "padj", "log2FoldChange"]],
            use_container_width=False,
            hide_index=False,
            height=((5 + 1) * 35 + 3),  # Where 5 is the number of rows to be shown
        )
