import streamlit as st
from apps.DNN import DNN_Model
from apps.LivePlot import LivePlot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
from datetime import datetime
from apps.SessionStats import SessionState
import base64
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

### global variables
score_df = pd.DataFrame()
train_complete , train_start,first_run = False, False, True
hist_df = {}

def app(state):
    global train_complete, train_start, first_run, hist_df, score_df
     
    state_cls = SessionState.get(train_start=False,
                                save_btn=False,
                                cls_model=None,
                                graph=None,
                                prep_data=True)

    state_cls.kwargs = {}

    ml_type = 'classification'
    st.title(" Neural Networks - :cat2: Classification :dog2:")
    main_canvas = st.empty()
    train_pct = st.sidebar.slider("Adjust Train percentage",0,100,80,format='%d %%')/100  
    prep_data = st.sidebar.checkbox("Preprocess data",value=True,key=111)
    if prep_data :
        state_cls.prep_data = True
    else :
        state_cls.prep_data = False

    train_start = st.sidebar.button("Train")  
    if train_start :
        state_cls.train_start = True 
    else:
        state_cls.train_start = False
    
    if (state.df is not None) or (state_cls.train_start == True):
        
        with main_canvas.beta_container():
            st.markdown("### Loss/Metric Vs Epoch")
            state_cls.graph = st.line_chart(pd.DataFrame())

        score_col, h_param_col = st.beta_columns(2)
         
        with score_col:
            st.markdown("### Performance Score")
            score_cont = st.empty()
               
        with h_param_col:
            st.markdown("##### Enter Hyper Parameters here")
            hparams = st.text_area("  ",height=100)
        
        save_model_col, dummy_col = st.beta_columns(2)

        if state_cls.train_start:
            if state.target_choice is not None:
                
                with score_cont.beta_container():
                    score_df = pd.DataFrame()
                    score = st.table(score_df)
                
                ###### Preprocessing data #######
                if state_cls.prep_data :
                    print("Pre-processing")
                    select_df = feature_selection(state.df,state.target_choice,ml_type=ml_type)
                    X_train,X_test,y_train,y_test = split_data(select_df,state.target_choice,split_ratio=1-train_pct)
                else:
                    print("No pre-processing")
                    X_train,X_test,y_train,y_test = split_data(state.df,state.target_choice,split_ratio=1-train_pct)

                ### custom preprocessing Y-axis ####
                y_train = y_train - 3
                y_test = y_test - 3
                ##################################
                X_train,X_test = scale_data([X_train,X_test])
                callbacks = LivePlot(2,train_loss='val_loss',train_metric='val_accuracy',graph=state_cls.graph)
                nn = DNN_Model(type=ml_type)
                state_cls.kwargs = validate_hparams(hparams)
                print(f" OutPut neurons = {state.df[state.target_choice].nunique()}")
                if state_cls.kwargs != None:
                    m1 = nn.create_model(X_train.shape[1],state.df[state.target_choice].nunique(),**state_cls.kwargs)
                else:
                    m1 = nn.create_model(X_train.shape[1],state.df[state.target_choice].nunique())
                state_cls.cls_model = m1
                train_results = nn.train_model(m1,X_train,y_train,['sparse_categorical_crossentropy'],['accuracy'],callbacks=[callbacks],ret_result=True)   
                test_results = nn.test_model(X_test,y_test)
                score_df = score_df.append(pd.DataFrame({'Train Score':train_results[1], 'Test Score':test_results[1]},index=['accuracy']))
                with score_cont.beta_container():
                    score.add_rows(score_df)
                if len(score_df) > 0:
                    train_complete = True
                    hist_df = pd.DataFrame(nn.history.history)
                    plot_all_charts(state_cls.graph,hist_df)
                else:
                    train_complete = False
            else:
                with main_canvas.beta_container():
                    st.warning(f"Please select a target variable from the sidebar")

        with save_model_col:
            save_btn = st.button("Save Model")
            download_link = st.empty()
            save_info = st.empty()
            if save_btn :
                state_cls.save_btn = True
                if train_complete:
                    with save_info.beta_container():
                        st.info(f"Saving Model {state_cls.cls_model}")

                    plot_all_charts(state_cls.graph,hist_df)
                    with score_cont.beta_container():
                        score = st.table(score_df)
                    model_name = save_model(state_cls.cls_model,state.data_file_name_no_ext,ext=True,type=ml_type)

                    with download_link.beta_container():
                        href = create_download_link(model_name,state.file_loc )
                        st.markdown(href,unsafe_allow_html=True)
                    with save_info.beta_container():
                        st.info(f"Model Saved as  {model_name}")
                elif state_cls.train_start == False:
                    with save_info.beta_container():
                        st.info("Please Press 'Train' first to train and then Save the model")
                else:
                    with save_info.beta_container():
                        st.info("Cannot save the model now. Either Training in Progress or yet to start. ")
            else:
                state_cls.save_btn = False
    else:
        with main_canvas.beta_container():
            st.warning(f"Please upload the data to run the {ml_type} and select a target from the sidebar")
    

def feature_selection(df,target,ml_type='regression'):
    # print(f"{df.columns}")
    x_columns = df.drop([target],axis=1).columns
    X, y = df[x_columns], df[target]
    if ml_type == 'regression':
        clf = ExtraTreesRegressor(n_estimators=10)
        clf = clf.fit(X, y)
    elif ml_type == 'classification':
        clf = ExtraTreesClassifier(n_estimators=10)
        clf = clf.fit(X, y)
    else:
        return None
    model = SelectFromModel(clf, prefit=True,threshold='mean')
    X = model.transform(X)
    feat_df = pd.DataFrame({'Features':x_columns, 'Importance':clf.feature_importances_}).sort_values(by='Importance',ascending=False)
    sel_df = df[feat_df.loc[:X.shape[1]]['Features']].copy()
    sel_df[target] = y.values
    return sel_df


def create_download_link(fname,path):
    with open(fname, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{fname}">Click here to Download the model </a>'
    return href



def plot_all_charts(graph,df):
    # graph.empty()
    # graph.add_rows(df)
    graph.line_chart(df)
    return


def save_model(model,fname,ext=True,type=None):
    if ext == True:
        file_name = fname+"_"+datetime.now().strftime("%Y%m%d-%H%M%S")+"_"+type+".h5"
    else:
        file_name = fname+"_"+type+".h5"
    model.save(file_name,save_format='.h5')
    return file_name

    
def split_data(df,target,split_ratio=0.2,random_state=0):
    X = df.drop(target,axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=0)

def scale_data(vector_list,use_fit=0):
    ret_list =[]
    scaled = StandardScaler().fit(vector_list[use_fit])
    for ele in vector_list:
        ret_list.append(scaled.transform(ele))
    return ret_list


def validate_hparams(hparams):
    try:
        json_obj = json.loads(json.dumps(hparams))
        return eval(json_obj)
    except:
        return None

    
    




    


    
    