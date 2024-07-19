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
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit as st
from streamlit.components.v1 import components
sns.set()
from typing import Union
import matplotlib.cm as cm
from itertools import cycle

## Data Analysis
class PulseHeatPipe:
    """
    PulseHeatPipe is a Python library designed to perform thermodynamic data analysis on experimental data from pulsating heat pipes (PHP). 
    This library helps estimate optimal working conditions based on the change in Gibbs free energy within the PHP system.

    Key Features:
    - Generate a blank template file for experimental data entry.
    - Perform data extraction, transformation, and loading (ETL) from multiple experimental files.
    - Convert units to the SI system.
    - Calculate thermal resistance.
    - Calculate Gibbs free energy at given temperatures and pressures.
    - Select data within a specific temperature range.
    - Compute and save statistical properties (mean, standard deviation) of the data.
    - Calculate average values of thermal properties.
    - Identify optimal thermal properties based on the minimum Gibbs free energy change.

    Attributes:
        T_k (float): Conversion constant to Kelvin (default is 273.15).
        P_const (float): Conversion constant to absolute bar (default is 1.013).
        R_const (float): Real gas constant (default is 8.314 J/Kmol).
        dG_standard (float): Standard Gibbs free energy change of water (default is 30.9 KJ/mol).
        P_standard (float): Standard atmospheric pressure (default is 1.013 bar).
        verbose (bool): Verbosity flag for printing detailed messages (default is True).
        dir_path (str): Directory path for the working directory (default is '.').
        sample (str): Sample name for the data file (default is 'default').

    Methods:
        blank_file(blank='blank.xlsx'): Generate a blank sample Excel file for experimental data entry.
        data_etl(name='*', ext='.xlsx'): Load, filter, and combine experimental data from multiple Excel files.
        convert_to_si(df: pd.DataFrame): Convert data columns to SI units based on their units in the column names.
        compute_thermal_resistance(data: pd.DataFrame, T_evaporator_col: str, T_condenser_col: str, Q_heat_col: str,TR_output_col: str)
        compute_gibbs_free_energy(data: pd.DataFrame, T_evaporator_col: str, T_condenser_col: str, P_bar: str, to_csv: bool): Calculate the change in Gibbs free energy at given temperatures and pressures.
        data_chop(data: pd.DataFrame, Tmin: int, Tmax: int, T_col: str, chop_suggested: bool): Select data within a specified temperature range.
        compute_data_stat(data: pd.DataFrame, property: str, to_csv: bool, method: str, decimals: int): Calculate and save statistical properties (mean, standard deviation) of the data grouped by a specified property.
        thermal_property_avg(data: pd.DataFrame, decimals: int): Calculate average values of measured thermal properties and return them as a list of strings.
        best_TP(data: pd.DataFrame, decimals: int, gfe_col: str): Identify optimal thermal properties based on the minimum change in Gibbs free energy and return them as a list of strings.

    Usage:
        from PyPulseHeatPipe import PulseHeatPipe

        # Setting up the analysis
        analysis = PulseHeatPipe("dir_path", "sample")

        # Generating a blank sample file
        analysis.blank_file()

        # Performing data ETL
        df, df_conv = analysis.data_etl()

        # Converting data to SI units
        df_si = analysis.convert_to_si(df)

        # Calculating thermal resistance
        df_tr = analysis.compute_thermal_resistance(data: pd.DataFrame, T_evaporator_col: str, T_condenser_col: str,Q_heat_col: str,TR_output_col: str)

        # Calculating Gibbs free energy
        gfe_data = analysis.compute_gibbs_free_energy(df)

        # Selecting data within a temperature range
        selected_data = analysis.data_chop(df, Tmin=300, Tmax=400)

        # Computing statistical properties of the data
        df_mean, df_std = analysis.compute_data_stat(df)

        # Calculating average values of thermal properties
        avg_properties = analysis.thermal_property_avg(df)

        # Identifying optimal thermal properties
        optimal_properties = analysis.best_TP(df)

    Please find more detail at https://github.com/nirmalparmarphd/PyPulseHeatPipe 

    """
    
    #0
    def __init__(self, 
                 dir_path:str = '.', 
                 sample:str = 'default',
                 T_K:float = 273.15,
                 P_const:float = 1.013,
                 R_const:float = 8.314,
                 dG_standard:float = 30.9,
                 P_standard:float = 1.013,
                 verbose:bool = True):
        
        self.T_k = T_K # To convert in kelvin
        self.P_const = P_const # To convert in absolute bar
        self.R_const = R_const # Real Gas constant
        self.dG_standard = dG_standard # dG of water
        self.P_standard = P_standard # atmosphere pressure
        self.verbose = verbose
        self.dir_path = dir_path
        self.sample = sample

        if self.verbose:
            print(f"""\t --- set default params ---
            Temperature constant (Kelvin conversion):       {self.T_k}\t[K]
            Pressure constant (bar conversion):             {self.P_const}\t[bar]
            Real gas constant:                              {self.R_const}\t[J/Kmol]
            Change in Gibbs Free Energy of water:           {self.dG_standard}\t[KJ/mol]
            Standard pressure:                              {self.P_standard}\t[bar]
            verbose:                                        {self.verbose}
            \n
                  """)
            print(f"Directory path loaded for working directory: '{self.dir_path}'\n")
            print(f"Sample name saved as: '{self.sample}'")

    #1
    # sample xlsx blank file
    def blank_file(self, 
                    blank='blank.xlsx'):
        """
        blank_file is a method to generate a blank sample (.xlsx) file to prepare data file that can further used in thermodynamic analysis. 
        
        'time'= timestamp,
        'Te[C]'= Evaporator Temperature,
        'Tc[C]'= Condenser Temperature,
        'P[bar]'= Pressure (gauge) of PHP,
        'Q[W]'= Power Supply,
        'alpha'= Horizontal Angle of PHP,
        'beta'= Vertical Angle of PHP, 
        'pulse'= Visible pulse generation (y=1/n=0)

        args:
            blank = 'blank.xlsx'

        returns:
            pd.DataFrame # blank pd.DataFrame and writes blank.xlsx file

        use: 
            analysis = PulseHeatPipe("data/experiment/")
            analysis.blank_file()

        NOTE: This method is specially wrote for in-house experiment. This method can be use with prescribed file format and structure
        """
        self.blank = blank
        df_blank = pd.DataFrame({'time':[1] ,'Te[C]':[1], 'Tc[C]':[1],'P[bar]':[1], 'Q[W]':[1], 'alpha':[1], 'beta':[1], 'pulse':[1]})
        # creating blank file
        df_blank_out = df_blank.to_excel(self.dir_path + self.blank)
        
        if self.verbose:
            msg = (f"### {self.blank} file is created. Please enter the experimental data in this file. Do not alter or change of the column's head. ###")
            print(msg)
        return df_blank
    
    #2
    # data ETL    
    def data_etl(self, name='*', ext='.xlsx'):
        """
        data_etl loads experimental data from all experimental data files (xlsx).
        Filters data and keeps only important columns.
        Combine selected data and save to csv file.
        Convert units to SI [K, bar] system and save to csv file. 

        usage: analysis = PulseHeatPipe("path", "sample")
                df, df_conv = analysis.data_etl()

        NOTE: This method is specially wrote for in-house experiment. This method can be use with prescribed file format and structure
        """
        self.name = name
        self.ext = ext
        data_filenames_list = glob.glob((self.dir_path + self.name + self.ext))
        df_frames = []

        if self.verbose:
            print('list of files considered for ETL: \n',data_filenames_list)

        for i in range(0, len(data_filenames_list)) :
            # loading data in loop
            df_raw = pd.read_excel(data_filenames_list[i])
            selected_columns = ['time' ,'Te[C]', 'Tc[C]','P[bar]', 'Q[W]', 'alpha', 'beta', 'pulse']
            df_selected_columns = df_raw[selected_columns]
            df_frames.append(df_selected_columns)
        df = pd.concat(df_frames, axis=0, ignore_index=True).dropna()
      
        df_conv_fram = [df['time'], 
                        df['Te[C]']+self.T_k, 
                        df['Tc[C]']+self.T_k, 
                        df['Te[C]']-df['Tc[C]'] , 
                        df['P[bar]'] + self.P_const, 
                        df['Q[W]'], 
                        (df['Te[C]']-df['Tc[C]'])/df['Q[W]'] , 
                        df['alpha'], 
                        df['beta'], 
                        df['pulse']]
        df_conv = pd.concat(df_conv_fram, axis=1, ignore_index=True).dropna()
        df_conv_columns = ['time' ,'Te[K]', 'Tc[K]', 'dT[K]', 'P[bar]', 'Q[W]', 'TR[K/W]','alpha', 'beta', 'pulse']
        df_conv.columns = df_conv_columns
        # saving
        df_out = df.to_csv(self.dir_path + self.sample + "_combined_data.csv")
        df_conv_out = df_conv.to_csv(self.dir_path + self.sample + "_combined_converted_data.csv")

        if self.verbose:
            print(f"### Compiled and converted data is saved at: {self.dir_path}'{self.sample}_combined_converted_data.csv' ###")
        return df, df_conv

    #3
    def convert_to_si(self, 
                       df: pd.DataFrame):
        """
        to convert given Pressure and Temperature data to SI units

        NOTE: please write units in square brackets '[]'

        args:
            df: pd.DataFrame

        returns:
            df: pd.DataFrame
        """
        # pressure conversion function
        def pressure_to_bar(pressure, from_unit):
            # Conversion factors to bar with standard abbreviations
            conversion_factors = {
                'pa': 1e-5,         # Pascal
                'kpa': 1e-2,        # Kilopascal
                'mpa': 10,          # Megapascal
                'atm': 1.01325,     # Atmosphere
                'bar': 1,           # Bar
                'mbar': 1e-3,       # Millibar
                'torr': 1.33322e-3, # Torr
                'psi': 6.89476e-2,   # Pounds per square inch
                'mmhg': 1.33322e-3  # Millimeters of mercury
            }
            
            # Normalize unit to lower case
            from_unit = from_unit.lower()
            
            if from_unit not in conversion_factors:
                raise ValueError(f"Unsupported unit: {from_unit}")
            
            # Convert to bar
            bar_value = pressure * conversion_factors[from_unit]
            
            return bar_value
        
        # temperature conversion function
        def temperature_to_kelvin(temperature, from_unit):
            # Conversion formulas to Kelvin
            conversion_formulas = {
                'c': lambda x: x + 273.15,           # Celsius to Kelvin
                'f': lambda x: (x - 32) * 5.0/9.0 + 273.15, # Fahrenheit to Kelvin
                'k': lambda x: x,                    # Kelvin to Kelvin (no change)
                'r': lambda x: x * 5.0/9.0           # Rankine to Kelvin
            }
            
            # Normalize unit to lower case
            from_unit = from_unit.lower()
            
            if from_unit not in conversion_formulas:
                raise ValueError(f"Unsupported unit: {from_unit}")
            
            # Convert to Kelvin
            kelvin_value = conversion_formulas[from_unit](temperature)
            
            return kelvin_value

        for col in df.columns:
            match = re.search(r'\[(.*?)\]', col)
            if match:
                unit = match.group(1).lower()
                col_name = col.split('[')[0].strip()
                # pressure conversion
                if unit in ['pa', 'kpa', 'mpa', 'atm', 'bar', 'mbar', 'torr', 'psi']:
                    df[col_name+'[bar]'] = df[col].apply(lambda x: pressure_to_bar(x, unit))
                # temperature conversion
                elif unit in ['c', 'f', 'k', 'r']:
                    df[col_name+'[K]'] = df[col].apply(lambda x: temperature_to_kelvin(x, unit))

        return df      

    #4
    def compute_thermal_resistance(self,
                                   data: pd.DataFrame,
                                   T_evaporator_col: str,
                                   T_condenser_col: str,
                                   Q_heat_col: str,
                                   TR_output_col: str,
                                   )->pd.DataFrame:
        """
        To calculate thermal resistance from given experimental data

        args:
            data: pd.DataFrame,
            T_evaporator_col: str,
            T_condenser_col: str,
            Q_heat_col: str
            TR_output_col: str      # user set a column name

        returns:
            pd.DataFrame            # col will be added for thermal resistance

        """
        # checking df is not empty
        if not data.empty:
            data[TR_output_col] = (data[T_evaporator_col] - data[T_condenser_col]) / data[Q_heat_col]
        else:
            if self.verbose:
                print('given pandas DataFrame is empty!')
        return data

    #5
    # to calculate gibbs free energy at given (T[K],P[bar])
    def compute_gibbs_free_energy(self, 
                                    data: pd.DataFrame,
                                    T_evaporator_col: str = 'Te[K]',
                                    T_condenser_col: str = 'Tc[K]',
                                    P_bar: str = 'P[bar]',
                                    to_csv: bool = False
                                    ):
        """
        gibbs_free_energy (GFE) method calculates the change in the gibbs free energy at a given vacuum pressure and temperature of the PHP.

        NOTE: please convert to absolute pressure before using 'compute_gibbs_free_energy' function

        default considered thermal parameters for calculation.
        dG = dG' + RTln(P/P')
        here, R = 8.314 [J/molK]
        P and P' = Pressure [bar]
        T = Temperature [K]
        
        args:
            data: pd.DataFrame      # pandas pd.DataFrame
            T_evaporator_col:str    # column name for evaporator data
            T_condenser_col:str     # column name for condenser data
            to_csv:bool             # default False, to save output DataFrame as a csv file

        returns:
            pd.DataFrame
        """
        Te = (data[T_evaporator_col]) 
        Tc = (data[T_condenser_col])  
        P_vacuum = (data[P_bar]) # converting to bar

        # calculating gfe
        dG_vacuum_Te = self.R_const * Te * np.log(P_vacuum/self.P_standard)
        dG_vacuum_Tc = self.R_const * Tc * np.log(P_vacuum/self.P_standard)
        dG = dG_vacuum_Te - dG_vacuum_Tc

        df_dG_vacuum_Te = pd.DataFrame({'GFE_Te[KJ/mol]': dG_vacuum_Te})
        df_dG_vacuum_Tc = pd.DataFrame({'GFE_Tc[KJ/mol]': dG_vacuum_Tc})
        df_dG = pd.DataFrame({'dG[KJ/mol]': dG})

        # making df
        data = pd.concat([data, df_dG_vacuum_Te, df_dG_vacuum_Tc, df_dG], axis=1, ignore_index=False)
        data.fillna(0, inplace=True)
        
        # to csv
        if to_csv:
            data_out = data.to_csv(self.dir_path + "gfe_combined.csv", index=False)
            if self.verbose:
                msg = print(f"Gibbs Free Energy calculated data saved at: {self.dir_path}'gfe_combined.csv")
        return data
    
    #6
    # To select data from specific Te range
    def data_chop(self, 
                  data: pd.DataFrame, 
                  Tmin: int = 300, 
                  Tmax: int = 400,
                  T_col: str = 'Te[K]',
                  chop_suggested: bool = False
                  ):
        """ 
        data_chop method is used to chop the data for the selected temperature value from the Te[K] column.
        if selected temperature range is not correct then method will suggest suitable temperature range.

        args:
            data: pd.DataFrame
            Tmin: int = 300,                    # choice of T min 
            Tmax: int = 400,                    # choice of T max
            T_col: str = 'Te[K]                 # selected column name
            chop_suggested: bool                # to chop DataFrame on suggested bounds

        returns:
            pd.DataFrame
        
        use: 
            data = analysis.data_chop(df, Tmin, Tmax)
        
        here, Tmin/Tmax is a suitable value (int) from the data.
        default values: Tmin=300, Tmax=400
        """
        if not data.empty:
            Tmina = data[T_col].min()
            Tmaxa = data[T_col].max()

            assert Tmin < Tmax, f"Entered wrong values: Correct range [Tmin:{round(Tmina,4)}, Tmax:{round(Tmaxa,4)}]"
            
            print(f"Optimal range of temperature(Te) for data selection: [Tmin:{round(Tmina,4)}, Tmax:{round(Tmaxa)}]")
            data_T = data[(data[T_col] <= Tmax) & (data[T_col] >= Tmin)]

            if chop_suggested:
                data_T = data[(data[T_col] <= Tmaxa) & (data[T_col] >= Tmina)]
        else:
            print('DataFrame is empty!')

        return data_T
    
    #7
    # data mixing and re-arranging
    def compute_data_stat(self, 
                  data: pd.DataFrame, 
                  property = 'Te[K]',
                  to_csv: bool = False,
                  method: str = 'mean',
                  decimals: int = 2 ):
        """
        compute_data_stat sorts and arrange value by a group from the experimental data, calculates mean and standard deviation of the grouped data.
        Calculated result will be stored at the location of data files.

        args:
            data: pd.DataFrame, 
            property = 'Te[K]',
            to_csv: bool = False
            method: mean = 'mean' or 'std' or 'min' or 'max' etc.
            decimals: int = 2 # to help in grouping with a choice of precision

        returns
            pd.DataFrame

        use:
            df_mean, df_std = analysis.compute_data_stat(data)
        """
        # obj cols
        obj_cols = data.select_dtypes(include='object').columns

        # stat compute
        df_num = data.drop(columns=obj_cols)\
                        .sort_values(by=property)\
                        .round(decimals)\
                        .groupby(property, as_index=False)\
                        .agg(method)\
   
        # Select first object column value per group
        df_obj = data.sort_values(by=property)\
                        .round(decimals)\
                        .groupby(by=property, as_index=False)[obj_cols]\
                        .first()
        
        # Concatenate numerical and object dataframes
        df_out = pd.concat([df_num, df_obj], axis=1)

        if to_csv:
            df_out.to_csv(self.dir_path + f'grouped_{method}.csv', index=False)
            if self.verbose:
                print(f"Calculated property saved at {self.dir_path}'grouped_{method}.csv'")

        return df_out
    
    #8
    # prepare average values for all thermal properties
    def thermal_property_avg(self, 
                             data: pd.DataFrame,
                             decimals: int = 2
                             ) -> list[str]:
        """
        Calculates average values of measured thermal properties for the given experiment data.

        args:
            data (pd.DataFrame): DataFrame containing the experimental data.
            decimals (int): Precision for rounding the data. Default is 2.

        returns:
            list[str]: List of strings containing the average and standard deviation for each thermal property.
        """
        df_mean = data.round(decimals)
        data.fillna(0, inplace=True)

        if self.verbose:
            print('--- Thermal Properties Average ---')

        msgs = []
        for col in df_mean.select_dtypes(exclude=['object', 'datetime']).columns:
            # Extracting the unit from the column name if it exists
            unit_match = re.search(r'\[(.*?)\]', col)
            if unit_match:
                unit = unit_match.group(0)
                property_name = col.split('[')[0].strip()
            else:
                continue
                # unit = ''
                # property_name = col
            try:
                # Calculate average and standard deviation
                property_avg = np.round(df_mean[col].mean(), 2)
                property_std = np.round(df_mean[col].std(), 2)

                if np.isnan(property_std):
                    property_std = 0
                
                msg = f"{property_name}\t\t average:\t\t {property_avg} ± {property_std}\t\t {unit}"
                msgs.append(msg)

                if self.verbose:
                    print(msg)
            except Exception as e:
                if self.verbose:
                    print(f"Could not process column '{col}': {e}")
        return "\n".join(msgs)
        
    
    #9
    # find optimal G(T,P) of PHP
    def best_TP(self, 
                data: pd.DataFrame,
                decimals: int = 2,
                gfe_col: str = 'dG[KJ/mol]'):
        """ 
        best_TP finds best G(T,P) with lowest dG (Change in Gibbs Free Energy for Te->Tc values at constant Pressure) from the experimental dataset.
        
        NOTE: please write units in square brackets '[]'

        args:
            data: pd.DataFrame
            decimals: int = 2               # to help in grouping with a choice of precision
            gfe_col: str = 'GFE[KJ/mol]'    # column name that contain the change in Gibbs Free Energy values

        returns:
            string

        use: 
            analysis.best_TP(data)
        """

        data = data.round(decimals)
        df_opt = data[data[gfe_col] == data[gfe_col].min()]
        df_opt_idx = df_opt.index

        if self.verbose:
            print(f'--- optimal thermal property at min(dG) ---')
        msgs = []
        for col in data.select_dtypes(exclude=['object', 'datetime']).columns:
            if '[' in col and ']' in col:
                unit_match = re.search(r'\[(.*?)\]', col)
                try:
                    if unit_match:
                        unit = unit_match.group(0)
                        property_avg = data[col].loc[df_opt_idx].mean()
                        property_std = data[col].loc[df_opt_idx].std()
                        if np.isnan(property_std):
                            property_std = 0

                        msg = f"{col.split('[')[0]}\t\t Optimal:\t\t  {property_avg} ± {property_std}\t\t {unit}"
                        msgs.append(msg)
                        if self.verbose:
                            print(msg)
                    else:
                        continue
                except Exception as e:
                    if self.verbose:
                        print(f"Could not process column '{col}': {e}")
        return "\n".join(msgs)
    
## Data Visualization
class DataVisualization(PulseHeatPipe):
    """ DataVisualization class can be used for PHP experimental data plotting and interactive visualization with the help of PyGWalker and Streamlit python libraries.

        ## use: 
        
        importing module
        from analysis import DataVisualization
        
        creating the reference variable
        visual = DataVisualization('dir_path', 'sample')
        
        data Visualization; eg. plotting all data
        visual.get_dashboard()

        calling help
        help(visual)
    """
    #0
    def __init__(self, dir_path: str = '.', 
                 sample: str = 'default', T_K: 
                 float = 273.15, 
                 P_const: float = 1.013, 
                 R_const: float = 8.314, 
                 dG_standard: float = 30.9, 
                 P_standard: float = 1.013, 
                 verbose: bool = True):
        super().__init__(dir_path, sample, T_K, P_const, R_const, dG_standard, P_standard, verbose)

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
                auto_data_chop: bool = True,
                plot_method: Union['combined', 'separate'] = 'combined',
                figsize: tuple = (18, 9),
                save_figure: bool = False,
                show: bool = False,
                plot_background: str = 'seaborn-v0_8-whitegrid',
                T_pulse_col: str = None,
                colormap: str = 'tab10'):  # Added colormap argument
        
        """
        To plot thermal properties for given experimental dataset.

        args:
            data: pd.DataFrame,         # data
            x_col: str,                 # x axis data
            y_col: str,                 # y axis data
            auto_data_chop: bool,       # if need to chop/select good data
            plot_method: str = Union['combined', 'separate']
            save_figure: bool           # save as pdf
            show: bool = False          # show figure after run
            plot_background: str = 'seaborn-v0_8-whitegrid',
            T_pulse_col: str = 'T_pulse[K]', # col of Pulsation start temperature, if available
            colormap: str = 'tab10'     # colormap to use for different colors, such as viridis, plasma, inferno, magma, cividis, tab10, tab20, etc.

        returns:
            plot # matplotlib.pyplot.plt
        """
        # Set a style
        plt.style.use(plot_background)
        
        frs = data['FR[%]'].unique()
        qs = data['Q[W]'].unique()
        alphas = data['alpha'].unique()
        betas = data['beta'].unique()

        # Create a colormap
        cmap = cm.get_cmap(colormap, len(frs) * len(qs) * len(alphas) * len(betas))
        colors = cmap(np.linspace(0, 1, len(frs) * len(qs) * len(alphas) * len(betas)))
        color_cycle = cycle(colors)

        # combined plot
        if plot_method.lower() == 'combined':
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
                            color = next(color_cycle)

                            if auto_data_chop:
                                data_chop = self.data_chop(data=data_,
                                                        Tmin=200,
                                                        Tmax=400,
                                                        T_col='Te_mean[K]',
                                                        chop_suggested=True) 
                            else:
                                data_chop = data_

                            if not data_chop.empty:
                                # Plotting
                                plt.scatter(x=data_chop[x_col], y=data_chop[y_col], label=f'FR{fr}[%]_Q{q}[W]_A[{a}]_B[{b}]_{self.sample}_{x_col}_vs_{y_col}', color=color)
                                if T_pulse_col:
                                    # pulsation start temperature
                                    T_pulse = data_[T_pulse_col].min()
                                    if T_pulse > 273:
                                        plt.axvline(x=T_pulse, color=color)
                                    else:
                                        continue
                                plt.xlabel(f'{x_col}')
                                plt.ylabel(f'{y_col}')
                            else:
                                print('DataFrame is empty!')
                                continue
            # combined
            plt.legend()
            plt.title(f'FR {frs}% - Q {qs}W - alpha {alphas} - beta {betas} - {self.sample}')
            if save_figure:
                x_col_ = x_col.split('[')[0]
                y_col_ = y_col.split('[')[0]
                file_name = f'FR{fr}_Q{q}_A{a}_B{b}_{self.sample}_{x_col_}_vs_{y_col_}.pdf'
                file_path = f'{self.dir_path}/{file_name}'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                if self.verbose:
                    print(f'plot saved as "{file_path}"')
            if show:
                plt.show()

        # separate plot
        elif plot_method.lower() == 'separate':
            for fr in frs:
                print(f'FR {fr}')
                for q in qs:
                    print(f'Q {q}')
                    for a in alphas:
                        for b in betas:
                            print(f'alpha {a}, beta {b}')
                            
                            # Filter the dataframe
                            data_ = data[(data['FR[%]'] == fr) & (data['Q[W]'] == q) & (data['alpha'] == a) & (data['beta'] == b)]
                            color = next(color_cycle)

                            if auto_data_chop:
                                data_chop = self.data_chop(data=data_,
                                                        Tmin=200,
                                                        Tmax=400,
                                                        T_col='Te_mean[K]',
                                                        chop_suggested=True)
                            else:
                                data_chop = data_

                            if not data_chop.empty:
                                plt.figure(figsize=figsize)
                                plt.scatter(x=data_chop[x_col], y=data_chop[y_col], label=f'FR{fr}[%]_Q{q}[W]_A[{a}]_B[{b}]_{self.sample}_{x_col}_vs_{y_col}', color=color)
                                if T_pulse_col:
                                    # pulsation start temperature
                                    T_pulse = data_[T_pulse_col].min()
                                    if T_pulse > 273:
                                        plt.axvline(x=T_pulse, color=color)
                                    else:
                                        continue
                                plt.xlabel(f'{x_col}')
                                plt.ylabel(f'{y_col}')
                            else:
                                print('DataFrame is empty!')
                                continue
                            # separate
                            plt.legend()
                            plt.title(f'FR {fr}% - Q {q}W - alpha {a} - beta {b} - {self.sample}')
                            if save_figure:
                                x_col_ = x_col.split('[')[0]
                                y_col_ = y_col.split('[')[0]
                                file_name = f'FR{fr}_Q{q}_A{a}_B{b}_{self.sample}_{x_col_}_vs_{y_col_}.pdf'
                                file_path = f'{self.dir_path}/{file_name}'
                                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                                if self.verbose:
                                    print(f'plot saved as "{file_path}"')
                            if show:
                                plt.show()

        else:
            print("give appropriate argument ['combined', 'separate']")