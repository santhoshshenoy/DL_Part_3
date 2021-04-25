import tensorflow as tf
import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import time



class LivePlot(tf.keras.callbacks.Callback):
    
    
    def __init__(self,refresh_rate=5,train_loss=None,train_metric=None,graph=None):
        self.validation_prefix = "val_"
        self.refresh_rate = refresh_rate
        self.train_loss = train_loss
        self.val_loss = self.validation_prefix + train_loss
        self.train_metric = train_metric
        self.val_metric = self.validation_prefix + train_metric
        self.graph = graph

        
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and metrics
        self.train_losses = []
        self.train_metrics = []
        self.val_losses = []
        self.val_metrics = []
        
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        """
        Calculates and plots loss and metrics
        """
        # Extract from the log
        log_train_loss = logs.get(self.train_loss)
        log_train_metric = logs.get(self.train_metric)
        log_val_loss = logs.get(self.val_loss)
        log_val_metric = logs.get(self.val_metric)
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.train_losses.append(log_train_loss)
        self.train_metrics.append(log_train_metric)
        self.val_losses.append(log_val_loss)
        self.val_metrics.append(log_val_metric)

        temp_df = pd.DataFrame({
                                # self.train_loss:log_train_loss,
                                self.train_metric:log_train_metric,
                                # self.val_loss : log_val_loss,
                                # self.val_metric : log_val_metric
                                },index=[epoch])
        self.graph.add_rows(temp_df)
            
        
        
            
            
