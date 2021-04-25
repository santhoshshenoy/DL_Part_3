import streamlit as st
from st_aggrid import AgGrid
from streamlit_quill import st_quill
import streamlit_theme as stt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
from scipy import stats
from sklearn.preprocessing import StandardScaler
import itertools
import os
from collections import defaultdict 
import json
import time
from apps.SessionStats import SessionState

# try:
#     # Before Streamlit 0.65
#     from streamlit.ReportThread import get_report_ctx
#     from streamlit.server.Server import Server
# except ModuleNotFoundError:
#     # After Streamlit 0.65
#     from streamlit.report_thread import get_report_ctx
#     from streamlit.server.server import Server


sncp_c1 = sns.color_palette("muted", 255)
# stt.set_theme({'primary': '#1b3388'})
plt.style.use('fivethirtyeight')
# notes_dict = defaultdict(str)


def app(state):
    
    st.markdown("# Exploratory Data Analysis :mag: :mag_right:",unsafe_allow_html=True)
    chart_title = st.empty()
    main_canvas = st.empty()
    main_options = st.empty()
    notes_area = st.empty()
    status_text = st.empty()
    
    
    analysis_types = ['Tabular View','Univariate','Multivariate','Pairplot','Heatmap','PCA','Outliers']

    if state.df is not None:

        if (state.file_loc != None) | (state.data_file_name_no_ext != None) :
            note_file_name = state.file_loc+'/'+state.data_file_name_no_ext+'_notes.json'
            if  os.path.exists(note_file_name) :
                notes_dict = load_notes_from_file(note_file_name)
            else:
                notes_dict = defaultdict(str)

        
        with main_options.beta_expander("Expand for Analysis Type",expanded=True):
            col1, col2 = st.beta_columns(2)

        
        with col1:
            ch_analysis = st.selectbox("Select an Analysis",analysis_types,key=5)
        
        with notes_area.beta_expander("Expand for Notes - +/- for Refresh",expanded=True):
            try:
                note_text = notes_dict[ch_analysis]
            except:
                note_text = " "
            note_content = st_quill(placeholder=note_text,value=note_text, html=True,readonly=False,key=10)
            if st.button('Save') == True:
                notes_dict[ch_analysis] = note_content
                status_text.info(f"Notes Saved for {ch_analysis} Analysis")
                if save_notes(note_file_name,notes_dict) == False:
                    status_text.warning(f"Notes could not be created")

        if ch_analysis == 'Tabular View':
            table_opts = ['Raw Data','Describe','Correlation','Variance Threshold','VIF','PCA']
            with col2:
                sel_tab = st.selectbox("Select a table type",table_opts,key=4)
            if sel_tab == 'Raw Data':
                show_table(main_canvas,chart_title,state.df,type='raw')

            elif sel_tab == 'Describe':
                des_df = pd.DataFrame(state.df.describe().T)
                des_df['dtypes'] = state.df.dtypes
                des_df['N-Uniques'] = state.df.nunique()
                des_df['NaNs'] = state.df.isna().sum()
                show_table(main_canvas,chart_title,des_df,type='des')

            elif sel_tab == 'Correlation':
                show_table(main_canvas,chart_title,get_correlation_table(state.df),type='corr')

            elif sel_tab == 'Variance Threshold':
                if state.target_choice is not None:
                    show_table(main_canvas,chart_title,show_variance_threshold(state.df,state.target_choice),type='vart')
                else:
                    chart_title.empty()
                    main_canvas.empty()
                    with main_canvas.beta_container():
                        st.markdown("### Please Select the Target Variable")
                        pass

            elif sel_tab == 'VIF':
                if state.target_choice is not None:
                    show_table(main_canvas,
                    chart_title,
                    get_VIF_Table(state.df,state.target_choice),
                    type='vif')
                else:
                    chart_title.empty()
                    main_canvas.empty()
                    with main_canvas.beta_container():
                        st.markdown("### Please Select the Target Variable")
                        pass

            elif sel_tab == 'PCA':
                if state.target_choice is not None:
                    show_table(main_canvas,
                    chart_title,
                    get_PCA(state.df,state.target_choice,pca_opt='table'),
                    type='pca')
                else:
                    chart_title.empty()
                    main_canvas.empty()
                    with main_canvas.beta_container():
                        st.markdown("### Please Select the Target Variable")
                        pass
            else:
                pass

##############################################################################################################
        if ch_analysis == 'Univariate':
            with col2:
                sel_uni = st.selectbox("Univariate",state.df.columns,key=1)
            plot_univariate(main_canvas,chart_title,state.df[sel_uni])

##############################################################################################################

        elif ch_analysis == 'Multivariate':
            with col2:
                sel_mul_1 = st.selectbox("Multivariate-X",state.df.columns,key=2)
                sel_mul_2 = st.selectbox("Multivariate-Y",state.df.columns,key=3)
            plot_multivariate(main_canvas,chart_title,state.df[sel_mul_1],state.df[sel_mul_2])
##############################################################################################################

        elif ch_analysis == 'Pairplot':
            plot_pair(main_canvas,chart_title,state.df)
        
##############################################################################################################
        elif ch_analysis == 'Heatmap':
            plot_heatmap(main_canvas,chart_title,state.df)

##############################################################################################################        
        elif ch_analysis == 'PCA':
            if state.target_choice is not None:
                get_PCA(state.df,state.target_choice,pca_opt='graph',canvas=main_canvas,title=chart_title)
            else:
                chart_title.empty()
                main_canvas.empty()
                with main_canvas.beta_container():
                    st.markdown("### Please Select the Target Variable")
                    pass
##############################################################################################################
        elif ch_analysis == 'Outliers':
            if state.target_choice is not None:
                get_Outliers(state.df,state.target_choice,out_opt='graph',canvas=main_canvas,title=chart_title)
            else:
                chart_title.empty()
                main_canvas.empty()
                with main_canvas.beta_container():
                    st.markdown("### Please Select the Target Variable")
                    pass

##############################################################################################################        
        else:
            pass

############################################ End of Function App #################################################

def save_notes(fname,notes):
    try:
        with open(fname,'w') as outfile:
            json.dump(notes,outfile)
        return True
    except:
        return False

##############################################################################################################        
def load_notes_from_file(fname):
    try:
        with open(fname,'r') as infile:
            notes = json.load(infile)
        return notes
    except:
        return None

##############################################################################################################        

def plot_univariate(canvas,title,feature):
    canvas.empty()
    title.empty()

    title_str = "### <center> Univariate Analysis - "+feature.name+"</center>"
    with title.beta_container():
        st.markdown(title_str,unsafe_allow_html=True)

    with canvas.beta_container(): 
        fig,ax = plt.subplots(1,2)
        color = sncp_c1[random.randint(0,100)]
        sns.histplot(feature,kde=True,color=color,ax=ax[0])
        sns.boxplot(x=feature,orient='h',color=color,ax=ax[1])
        st.pyplot(fig)
    return
##############################################################################################################


def show_table(canvas,title,df,type='raw'):
    canvas.empty()
    title.empty()

    if type == 'raw':
        title_str = f"### <center> Raw Data </center> \n ##### **Shape: {df.shape}** "
    elif type == 'des':
        title_str = f"### <center> Data Description </center> "
    elif type == 'corr':
        title_str = f"### <center> Correlation Table </center> "
    elif type == 'vart':
        title_str = f"### Variance Threshold "
    elif type == 'vif':
        title_str = f"### Variance Inflation Factor "
    elif type == 'pca':
        title_str = f"### <center> Principal Component Analysis </center>"
    else:
        title_str = " "

    with title.beta_container():
        st.markdown(title_str,unsafe_allow_html=True)

    with canvas.beta_container():
        # st.write(df)
        AgGrid(df.reset_index())
    return
##############################################################################################################
    

def plot_multivariate(canvas,title,x,y):
    canvas.empty()
    title.empty()
    
    title_str = "### <center> Multivariate Analysis - " + x.name + " / " + y.name +"</center>"
    with title.beta_container():
        st.markdown(title_str,unsafe_allow_html=True)

    legend_text = f"ρ = {round(x.corr(y),2)}"
    with canvas.beta_container():
        fig = plt.figure()
        sns.regplot(x=x,y=y,  line_kws={'color': 'black'},color= sncp_c1[random.randint(0,100)],label=legend_text).legend(loc="best")
        st.pyplot(fig)
    return
##############################################################################################################

def plot_pair(canvas,title,df):
    canvas.empty()
    title.empty()

    title_str = "### <center> Pairplot Analysis </center>"
    with title.beta_container():
        st.markdown(title_str,unsafe_allow_html=True)

    with canvas.beta_container():
        fig = sns.pairplot(df)
        st.pyplot(fig)
    return
##############################################################################################################


def plot_heatmap(canvas,title,df):
    canvas.empty()
    title.empty()

    title_str = "### <center> Heatmap Analysis </center> "
    with title.beta_container():
        st.markdown(title_str, unsafe_allow_html=True)

    with canvas.beta_container():
        fig=plt.figure()
        sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f",annot_kws={'fontsize':8})
        st.pyplot(fig)
    return
##############################################################################################################


def get_correlation_table(df,alpha=0.05):
    xy_pairs = [_ for _ in itertools.combinations(df.columns,2)]
    corr_df = pd.DataFrame(columns=['Variable 1','Variable 2','Corr_Coeff','P_Value'])
    for pairs in xy_pairs:
        pearson_coef, p_value = stats.pearsonr(df[pairs[0]], df[pairs[1]])

        if (p_value < alpha):
            significance = f"Yes (p_value <= {alpha})"
        else:
            significance = "No"

        corr_df = corr_df.append({'Variable 1':pairs[0],
        'Variable 2':pairs[1],
        'Corr_Coeff':pearson_coef,
        'P_Value':p_value,
        'Significance':significance},
        ignore_index=True)

    return corr_df
##############################################################################################################


def show_variance_threshold(df,target,threshold=0.0):
    from sklearn.feature_selection import VarianceThreshold

    columns_to_show = df.drop(target,axis=1).columns
    vthreshold = VarianceThreshold(threshold=threshold)
    vthreshold.fit(df[columns_to_show].values)
    vt_df = pd.DataFrame({'Feature':columns_to_show,
                        'Variance Threshold':vthreshold.variances_}
                        ).sort_values(by='Variance Threshold',ascending=False)
    return vt_df
##############################################################################################################


def get_VIF_Table(df,target):
    from statsmodels.stats.outliers_influence import variance_inflation_factor 
    df = df.drop(target,axis=1).copy()
    vif_data = pd.DataFrame() 
    vif_data["Feature"] = df.columns 
    vif_data["VIF"] =[variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data.sort_values(by='VIF',ascending=False)
##############################################################################################################


def get_PCA(df,target,pca_opt='table',canvas=None,title=None):

    from sklearn.decomposition import PCA
    
    X_columns = df.drop(target,axis=1).columns
    scaler = StandardScaler().fit_transform(df[X_columns])
    pca = PCA()
    pca.fit(scaler)

    pca_len = len(pca.explained_variance_ratio_)

    if pca_opt == 'table':
        pca_comp = pd.DataFrame(data=np.abs(pca.components_),columns=["PCA-"+str(i) for i in range(0,pca_len) ])
        return pca_comp

    pca_data = {"Components": ["PCA-"+str(i) for i in range(0,pca_len) ],
                "Cumulative Variance":np.cumsum(pca.explained_variance_ratio_) }
    pca_df = pd.DataFrame(pca_data)

    if pca_opt == 'graph':
        canvas.empty()
        title.empty()

        title_str = "### <center> Principal Component Analysis </center> "
        with title .beta_container():
            st.markdown(title_str,unsafe_allow_html=True)

        with canvas.beta_container():
            fig,ax = plt.subplots()
            sns.lineplot(x=range(pca_len),y=np.cumsum(pca.explained_variance_ratio_), drawstyle='steps-post')
            plt.ylabel('Explained Variance')
            plt.xlabel('Principal Components')
            plt.axhline(y=0.95,color='red',linestyle=':',linewidth=3)
            ax.text(0,0.96,'95% Threshold',color='k')
            [ax.text(rows[0]-0.5,rows[1][1]+0.01,rows[1][0],color='k',fontdict={'fontsize':9}) for rows in pca_df.iterrows()]
            st.pyplot(fig)
        return
 ##############################################################################################################
    
def get_Outliers(df,target,out_opt='table',canvas=None,title=None,threshold=1.0e-09):
        from scipy.stats import multivariate_normal

        columns_to_use = df.drop(target,axis=1).columns

        covariance_matrix = df[columns_to_use].cov()
        mean_values = [df[cols].mean() for cols in columns_to_use]

        model = multivariate_normal(cov=covariance_matrix,mean=mean_values)

        # threshold = 1.0e-9
        outlier = model.pdf(df[columns_to_use]).reshape(-1)

        outlier_label = [ True if l < threshold else False for l in outlier]

        if out_opt == 'table':
            df['Outlier'] = outlier_label
            return df

        elif out_opt == 'graph':
            canvas.empty()
            title.empty()

            title_str = "### <center> Multivariate Outlier Analysis (Gaussian Method) </center> "
            with title .beta_container():
                st.markdown(title_str,unsafe_allow_html=True)

            with canvas.beta_container():
                fig,ax = plt.subplots()

                unique_labels = set(np.unique(outlier_label))
                colors = ['blue', 'red']

                for color,label in zip(colors, unique_labels):
                    sample_mask = [True if l == label else False for l in outlier_label]
                    sns.scatterplot(x=df[columns_to_use[0]][sample_mask], y=df[columns_to_use[1]][sample_mask],color=color,)
                plt.xlabel(columns_to_use[0])
                plt.ylabel(columns_to_use[1])
                plt.legend(['Inliers','Outliers'],loc='best')
                st.pyplot(fig)
            return
        
# def _get_session():
#     session_id = get_report_ctx().session_id
#     session_info = Server.get_current()._get_session_info(session_id)

#     if session_info is None:
#         raise RuntimeError("Couldn't get your Streamlit Session object.")
    
#     return session_info.session

# ### Function to retrieve the current session state
# def _get_state():
#     session = _get_session()

#     if not hasattr(session, "_custom_session_state"):
#         session._custom_session_state = SessionState(session)

#     return session._custom_session_state

