from php_analysis import PulseHeatPipe
## PHP Data Analysis and Plotting Class
import numpy as np
import pandas as pd
from os import listdir
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set()

## Data Visualization
class DataVisualization(PulseHeatPipe):
    """ Data Visualization class to plot PHP experimental and processed data.

        ## use: 
        
        importing module
        from analysis import DataVisualization
        
        creating the reference variable
        visual = DataVisualization('dir_path', 'sample')
        
        data Visualization; eg. plotting all data
        visual.plot_all_data()
    """
    #0
    def __init__(self, dir_path: str, 
                 sample: str):
        super().__init__(dir_path, 
                         sample)
    #1    
    def plot_all_data(self, data:pd.DataFrame):
        """ Data Visualization of all data
            
            use: 
                visual.plot_all_data(data)
        """
        plt.figure(figsize=(10,5))
        sns.lineplot(data)
        plt.xlabel('Data')
        plt.ylabel('Properties')
        plt.title(f"All Data - {self.sample}")
        plt.legend()

    def plot_Te_Tc(self, data:pd.DataFrame):
        """ Data Visualization of Te vs Tc
            
            use:
                visual.plot_Te_Tc(data)
        """
        plt.figure(figsize=(10,5))
        plt.plot(data['Te[K]'], label = 'Te[K]')
        plt.plot(data['Tc[K]'], label = 'Tc[K]')
        plt.xlabel('Te[K]')
        plt.ylabel('Tc[K]')
        plt.title(f"Te[K] vs Tc[K] - {self.sample}")
        plt.legend()

    def plot_eu(self, df_mean:pd.DataFrame, df_std:pd.DataFrame, property:str, point='.k', eu='r'):
        """ Data Visualization with expanded uncertainty
            
            usage: visual.plot_eu(df_mean, df_std, property='Tc[K]', point='.k', eu='r')
                    here, choose value from property list: ['Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]', 'GFE_Te[KJ/mol]', 'GFE_Tc[KJ/mol]', 'dG[KJ/mol]']
        """
        self.property = property
        self.xproperty = 'Te[K]'
        self.point = point
        self.eu = eu
        properties = ['Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]', 'GFE_Te[KJ/mol]', 'GFE_Tc[KJ/mol]', 'dG[KJ/mol]']
        if self.property in properties:    
            plt.figure(figsize=(10,5))
            plt.plot(df_mean[self.xproperty], df_mean[self.property], self.point, label=self.property)
            idx = df_std.index
            df_mean_idx = df_mean.loc[idx]
            plt.fill_between(df_std[self.xproperty], df_mean_idx[self.property] - 2* df_std[self.property], df_mean_idx[self.property] + 2* df_std[self.property],color=self.eu, alpha=0.2, label='Expanded Uncertainty')
            plt.xlabel(self.xproperty)
            plt.ylabel(self.property)
            plt.title(f"Expanded Uncertainty - {self.sample}")
            plt.legend()
        else:
            print(f"Entered invalid value [{self.property}] of thermal property!\n")
            print(f"Select any correct value from: {properties}")
        return