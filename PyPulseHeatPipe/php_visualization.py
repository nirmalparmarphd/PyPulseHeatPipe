## PHP Data Analysis and Plotting Class
import numpy as np
import pandas as pd
from os import listdir
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pygwalker as pyg
from datetime import datetime
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit as st
from streamlit.components.v1 import components
import subprocess
from typing import Union
sns.set()
# from PyPulseHeatPipe.php_analysis import PulseHeatPipe
# from php_analysis import PulseHeatPipe

## Data Visualization
class DataVisualization(PulseHeatPipe):
    """ DataVisualization class can be used for PHP experimental data plotting and interactive visualization with the help of PyGWalker and Streamlit python libraries.

        ## use: 
        
        importing module
        from analysis import DataVisualization
        
        creating the reference variable
        visual = DataVisualization('dir_path', 'sample')
        
        data Visualization; eg. plotting all data
        visual.plot_all_data()
    """
    #0
    def __init__(self, 
                 dir_path: str, 
                 sample: str):
        self.dir_path = dir_path
        self.sample: sample        
    #1    
    def get_dashboard(self, 
                      data: pd.DataFrame,
                      spec: str = 'php_chart.json'):
        
        """ To get a data visualization dashboard for interactive plotting.
            This dashboard is created with 'PyGWalker' library. 
            Please find more details on operating the dashboard here: https://github.com/Kanaries/pygwalker?tab=readme-ov-file
            
            args:
                data: pd.DataFrame
                spec: str   # to save json chart

            use: 
                visual.get_dashboard(data)
        """

        path = os.path.join(self.dir_path, spec)
        dashboard = pyg.walk(dataset=data,
                            spec=path,
                            kernel_computation=True,
                            )
        return dashboard
    
    #2
    def get_plots(self,
                  data: pd.DataFrame,
                  x_col: str,
                  y_col: str,
                  data_chop: bool = True,
                  plot_method: str = Union['combined', 'separate'],
                  figsize: tuple = (18, 9),
                  ):
        
        """
        To plot thermal properties for given experimental dataset.

        args:
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            data_chop: bool,
            plot_method: str = Union['combined', 'separate']

        returns:
            plot # matplotlib.pyplot.plt
        """
        frs = data['FR[%]'].unique()
        qs = data['Q[W]'].unique()
        alphas = data['alpha'].unique()
        betas = data['beta'].unique()

        # combined plot
        if plot_method.lower() == 'combined':
            # Assuming frs, qs, alphas, betas are defined and data is your DataFrame
            plt.figure(figsize=figsize)
            for fr in frs:
                print(f'FR {fr}')
                for q in qs:
                    print(f'Q {q}')
                    for a in alphas:
                        for b in betas:
                            print(f'alpha {a}, beta {b}')
                            
                            # Filter the dataframe
                            data_ = data[(data['FR[%]'] == fr) & (data['Q[W]'] == q) & (data['alpha'] == a) & (data['beta'] == b)]

                            if data_chop:
                                data_chop = self.data_chop(data=data_,
                                                                Tmin=300,
                                                                Tmax=360,
                                                                T_col='Te_mean[K]',
                                                                chop_suggested=True)
                                            
                            
                        if not data_chop.empty:
                            # Plotting
                            plt.scatter(x=data_chop[x_col], y=data_chop[y_col], label=f'FR{fr}[%]_Q{q}[W]_A[{a}]_B[{b}]_{x_col}_vs_{y_col}')
                        else:
                            print('DataFrame is empty!')
                # combined
                plt.legend()
                plt.title(f'FR {fr}% - Q {q}W - alpha {a} - beta {b}')
                plt.show()
        
        # separate plot
        if plot_method.lower() == 'combined':
            # Assuming frs, qs, alphas, betas are defined and data is your DataFrame
            plt.figure(figsize=figsize)
            for fr in frs:
                print(f'FR {fr}')
                for q in qs:
                    print(f'Q {q}')
                    for a in alphas:
                        for b in betas:
                            print(f'alpha {a}, beta {b}')
                            
                            # Filter the dataframe
                            data_ = data[(data['FR[%]'] == fr) & (data['Q[W]'] == q) & (data['alpha'] == a) & (data['beta'] == b)]

                            if data_chop:
                                data_chop = self.data_chop(data=data_,
                                                                Tmin=300,
                                                                Tmax=360,
                                                                T_col='Te_mean[K]',
                                                                chop_suggested=True)
                                            
                            
                        if not data_chop.empty:
                            # Plotting
                            plt.figure(figsize=figsize)
                            plt.scatter(x=data_chop[x_col], y=data_chop[y_col], label=f'FR{fr}[%]_Q{q}[W]_A[{a}]_B[{b}]_{x_col}_vs_{y_col}')
                        else:
                            print('DataFrame is empty!')
                        # separate
                        plt.legend()
                        plt.title(f'FR {fr}% - Q {q}W - alpha {a} - beta {b}')
                        plt.show()
            

        

