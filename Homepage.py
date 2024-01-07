### Imports ###

import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

### Page Config ###
st.set_page_config(page_title=None,
                   page_icon="🌱",
                   layout="centered",
                   initial_sidebar_state="expanded",
                   menu_items=None)

### Functions ###

### Main ###

if __name__ == "__main__":

    dir = os.path.join('sf5n64hydt-1', 'cancer types.mat')
    mat = scipy.io.loadmat(dir)

    ### Data collection ###
    cancerTypes = [type[0][0] for type in mat['cancerTypes']]
    data = mat['data'][:,:971] # removed column indicating cancer type
    genes = [id[0]for id in mat['geneIds'][0]] # 971 genes

    ### Data preprocessing ###
    df = pd.DataFrame(data=data, columns=genes)
    X = df.values
    scaled_X = StandardScaler().fit_transform(X)

    ### Table ###
    st.write("RPKM RNA-Seq Values of 971 Genes in Five Different Cancer Types Across 2086 Samples")

    st.dataframe(pd.concat([pd.DataFrame(cancerTypes), df], axis=1).rename(columns={0:'Cancer Type'}),
                 width=1000,
                 height=int(35.2*(5+1))) # Where 5 is the number of rows displayed

    with st.expander("Cancer Types", expanded=False):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["BRCA", "KIRC", "LUAD", "LUSC", "UCEC"])
        with tab1:
            st.write("**Breast Invasive Carcinoma**")
            st.write("A cancer that develops from breast tissue.")
        with tab2:
            st.write("**Kidney Renal Clear Cell Carcinoma**")
            st.write("A kidney cancer that originates in the lining of the proximal convoluted tubule, a part of the tubes in the kidney that transport urine. ")
        with tab3:
            st.write("**Lung Adenocarcinoma**")
            st.write("The most common type of lung cancer, most often found in patients with a history of smoking cigarettes.")
        with tab4:
            st.write("**Lung Squamous Cell Carcinoma**")
            st.write("A non-small-cell lung cancer that originates in the bronchi. Most prevalent type of lung cancer after lung adenocarcinoma.")
        with tab5:
            st.write("**Uterine Corpus Endometrial Carcinoma**")
            st.write("Also known as endometrial cancer, it is a cancer that arises from the endometrium (the lining of the uterus or womb).")

    ### PCA ###
    with st.spinner("Loading PCA..."):
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(scaled_X)
        principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
        final_df = pd.concat([principal_df, pd.DataFrame(cancerTypes)], axis=1)
        final_df.rename(columns={0:'cancer_type'}, inplace=True)

        fig = px.scatter_3d(final_df,
                            x='PC1', y='PC2', z='PC3',
                            color='cancer_type',
                            width=600, height=500,
                            opacity=1.0
                            )

        fig.update_traces(marker = dict(size=3))

        fig.update_layout(title={'text': "3D PCA Distinguishing Cancer Types <br> Through RPKM RNA-Seq Expression Values",
                                'y':0.9,
                                'x':0.50,
                                'xanchor': 'center',
                                'yanchor': 'top'},

                        scene = dict(
                            xaxis_title='PC1',
                            yaxis_title='PC2',
                            zaxis_title='PC3'
                            ),

                        legend=dict(
                            x=0.9,
                            y=0.8,
                        )

                        )

    st.write(fig)
    with st.expander("What is PCA?", expanded=False):
        st.write('''Principal component analysis (PCA) is a linear
                dimensionality reduction technique that preserves
                the variability within the dataset. This technique
                identifies principal components (variables that capture
                the variance in the data) and reorients the data's axes
                to the direction of maximum variance. PCA simplifies the
                complexity of high-dimensional data allowing for easier
                visualization, noise reduction, and feature selection.''')
        st.write(''' **Analysis of a complex of statistical variables into principal components**  \n*Hotelling, H.*, 1933''')

    ### t-SNE ###
    with st.spinner("Loading t-SNE..."):
        tsne = TSNE(n_components=3).fit_transform(scaled_X)
        principal_df = pd.DataFrame(data=tsne, columns=['t-SNE_1', 't-SNE_2', 't-SNE_3'])
        final_df = pd.concat([principal_df, pd.DataFrame(cancerTypes)], axis=1)
        final_df.rename(columns={0:'cancer_type'}, inplace=True)

        fig = px.scatter_3d(final_df,
                            x='t-SNE_1', y='t-SNE_2', z='t-SNE_3',
                            color='cancer_type',
                            width=600, height=500,
                            opacity=1.0
                            )

        fig.update_traces(marker = dict(size=3))

        fig.update_layout(title={'text': "3D t-SNE Distinguishing Cancer Types <br> Through RPKM RNA-Seq Expression Values",
                                'y':0.9,
                                'x':0.50,
                                'xanchor': 'center',
                                'yanchor': 'top'},

                        scene = dict(
                            xaxis_title='t-SNE_1',
                            yaxis_title='t-SNE_2',
                            zaxis_title='t-SNE_3'
                            ),

                        legend=dict(
                            x=0.9,
                            y=0.8,
                            )
                        )

    st.write(fig)
    with st.expander("What is t-SNE?", expanded=False):
        st.write('''t-Distributed Stochastic Neighbor Embedding (t-SNE)
                is a non-linear dimensionality reduction technique.
                t-SNE converts high-dimensional distances between
                datapoints into probabilites and minimizes the
                divergence between the probabilities. t-SNE can reveal
                clusters or groupings within the dataset. Unlike PCA,
                it maintains local patterns and relationships.''')
        st.write(''' **Visualizing Data using t-SNE**  \n*Laurens van der Maaten, Geoffrey Hinton*, 2008''')
