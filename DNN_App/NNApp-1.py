import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from apps import NNClassification , NNRegression, NNEDA
from apps.SessionStats import SessionState
from collections import defaultdict 


def main():

    state = SessionState.get(data_file_name_ext="",data_file_name_no_ext="",file_loc="",df=pd.DataFrame(),target_choice="")

    st.set_page_config(page_title="My EDA",layout='wide')
    
    author = st.sidebar.empty()
    with author.beta_expander("Developer Information", expanded=True):
        st.markdown("1. Om Prakash Suthar ")
        st.markdown("2. Santhosh Shenoy Panambur")

    st.sidebar.markdown("# :classical_building: Home")
    
    st.sidebar.markdown("###### :floppy_disk: Upload a CSV File")
    uploaded_file = st.sidebar.file_uploader('',type=['.csv'])

    if uploaded_file is not None:
        
        state.data_file_name_ext,state.data_file_name_no_ext,state.file_loc = get_storage_loc(os.path.abspath(uploaded_file.name))
        state.df = pd.read_csv(uploaded_file)
        state.target_choice = st.sidebar.selectbox('Select the target variable', [None]+list(state.df.columns))
        


    pages = {
        "EDA": NNEDA.app,
        "Regression": NNRegression.app,
        "Classification": NNClassification.app
    }
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)
    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    # state.sync() ### Commented with new code of SessionStats


def get_storage_loc(data_file_loc):
    fname_ext = data_file_loc.split("/")[-1]
    fname_no_ext = fname_ext.split(".")[:-1][0]
    file_loc = "/".join(data_file_loc.split("/")[:-1])
    return fname_ext, fname_no_ext, file_loc


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


if __name__ == "__main__":
    main()


