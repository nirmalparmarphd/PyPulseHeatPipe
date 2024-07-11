
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
import pygwalker as pyg
from datetime import datetime

sns.set()

## Data Visualization
class DataVisualization(PulseHeatPipe):
    """ DataVisualization class can be used for PHP experimental data plotting and interactive visualization

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
    def get_dashboard(self, 
                      data: pd.DataFrame,
                      spec: str = 'php_chart.json'):
        """ To get a data visualization dashboard for interactive plotting.
            This dashboard is created with 'PyGWalker' library. Please find more details on operating the dashboard here: https://github.com/Kanaries/pygwalker?tab=readme-ov-file
            
            use: 
                visual.get_dashboard(data)
        """
        path = os.path.join(self.dir_path, spec)
        dashboard = pyg.walk(dataset=data,
                             spec=path,
                             kernel_computation=True,
                             use_preview=True,
                             )
        
        return dashboard
        

