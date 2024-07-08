## PHP Data Analysis and Plotting Class
import numpy as np
import pandas as pd
from os import listdir
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

## Data Analysis
class PulseHeatPipe:
    """
    ## PulseHeatPipe is a Python library to perform thermodynamic data analysis on experimental data of pulsating heat pipe. This library is developed to estimate optimal working condition based on the change in Gibbs free energy in the pulsating heat pipe.

    Please find more detail at https://github.com/nirmalparmarphd/PyPulseHeatPipe 

    ## use:
    #  
    # importing the module
    # from PyPulseHeatPipe import PulsHeatPipe, DataVisualization
    #
    # setting working directory
    # analysis = PulseHeatPipe("datapath", "sample)
    #
    # for a class help 
    # help(analysis)
    #
    # for a function help
    # help(analysis.data_etl)
    """
    #0
    def __init__(self, 
                 datapath:str = '.', 
                 sample:str = 'default',
                 T_K:float = 273.15,
                 P_const:float = 1.013,
                 R_const:float = 8.314,
                 dG_standard:float = 30.9,
                 P_standard:float = 1.013):
        
        self.T_k = T_K # To convert in kelvin
        self.P_const = P_const # To convert in absolute bar
        self.R_const = R_const # Real Gas constant
        self.dG_standard = dG_standard # dG of water
        self.P_standard = P_standard # atmosphere pressure
        self.datapath = datapath
        self.sample = sample
        print(f"Datapath loaded for working directory: {self.datapath}\n")
        print(f"Sample name saved as: {self.sample}")

    #1
    # sample xlsx blank file
    def blank_filße(self, blank='blank.xlsx'):
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

        usage: analysis = PulseHeatPipe("path")
                analysis.blank_file()

        NOTE: This method is specially wrote for in-house experiment. This method can be use with prescribed file format and structure
        """
        self.blank = blank
        df_blank = pd.DataFrame({'time':[1] ,'Te[C]':[1], 'Tc[C]':[1],'P[bar]':[1], 'Q[W]':[1], 'alpha':[1], 'beta':[1], 'pulse':[1]})
        # creating blank file
        df_blank_out = df_blank.to_excel(self.datapath + self.blank)
        msg = (f"### {self.blank} file is created. Please enter the experimental data in this file. Do not alter or change of the column's head. ###")
        return msg
    
    #2
    # data ETL    
    def data_etl(self, name='*', ext='.xlsx'):
        """
        data_etl loads experimental data from all experimental data files (xlsx).
        Filters data and keeps only important columns.
        Combine selected data and save to csv file.
        Convert units to MKS [K, bar] system and save to csv file. 

        usage: analysis = PulseHeatPipe("path", "sample")
                df, df_conv = analysis.data_etl()

        NOTE: This method is specially wrote for in-house experiment. This method can be use with prescribed file format and structure
        """
        self.name = name
        self.ext = ext
        data_filenames_list = glob.glob((self.datapath + self.name + self.ext))
        df_frames = []
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
        df_out = df.to_csv(self.datapath + self.sample + "_combined_data.csv")
        df_conv_out = df_conv.to_csv(self.datapath + self.sample + "_combined_converted_data.csv")
        print(f"### Compiled and converted data is saved at: {self.datapath}'{self.sample}_combined_converted_data.csv' ###")
        return df, df_conv
    
    #3
    def convert_to_si(self, 
                       df: pd.DataFrame):
        """
        to convert given data to SI units

        args:
            df: pd.DataFrame

        returns:
            df: pd.DataFrame
        """
        pass

    #4
    # to calculate gibbs free energy at given (T[K],P[bar])
    def compute_gibbs_fe(self, 
                 data: pd.DataFrame,
                 T_evaporator_col: str = 'Te[K]',
                 T_condenser_col: str = 'Tc[K]',
                 to_csv: bool = False):
        """
        gibbs_fe calculates the change in the gibbs free energy at a given vacuum pressure and temperature.

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
        P_vacuum = (data[T_condenser_col]) # converting to bar

        # calculating gfe
        dG_vacuume_Te = self.R_const * Te * np.log(P_vacuum/self.P_standard)
        dG_vacuume_Tc = self.R_const * Tc * np.log(P_vacuum/self.P_standard)
        dG = dG_vacuume_Te - dG_vacuume_Tc

        # making df
        data = pd.concat([data, {dG_vacuume_Te: 'GFE_Te[KJ/mol]'}, {dG_vacuume_Tc: 'GFE_Tc[KJ/mol]'}, {dG: 'dG[KJ/mol]'}], axis=1, ignore_index=True)
        
        # to csv
        if to_csv:
            data_out = data.to_csv(self.datapath + "gfe_combined.csv")
            msg = print(f"Gibbs Free Energy calculated data saved at: {self.datapath}'gfe_combined.csv")
        return data
    
    #5
    # To select data from specific Te range
    def data_chop(self, 
                  data: pd.DataFrame, 
                  Tmin: int = 300, 
                  Tmax: int = 400,
                  T_evaporator_col: str = 'Te[K]',
                  T_condenser_col: str = 'Tc[K]'):
        """ 
        data_chop method is used to chop the data for the selected temperature value from the Te[K] column.

        args:
            data: pd.DataFrame
            Tmin: int = 300,                    # choice of T min 
            Tmax: int = 400,                    # choice of T max
            T_evaporator_col: str = 'Te[K]',    # evaporator column name
            T_condenser_col: str = 'Tc[K]'      # condenser column name
        
        returns:
            pd.DataFrame
        
        use: 
            data = analysis.data_chop(df, Tmin, Tmax)
        
        here, Tmin/Tmax is a suitable value (int) from the data.
        default values: Tmin=300, Tmax=400
        """
        Tmina = data[T_evaporator_col].min()
        Tmaxa = data[T_condenser_col].max()

        assert Tmin < Tmax, f"Entered wrong values: Correct range [Tmin:{round(Tmina,4)}, Tmax:{round(Tmaxa,4)}]"
        
        print(f"Optimal range of temperature(Te) for data selection: [Tmin:{round(Tmina,4)}, Tmax:{round(Tmaxa)}]")
        data_T = data[(data['Te[K]'] <= Tmax) & (data['Te[K]'] >= Tmin)]

        return data_T
    
    #6
    # data mixing and re-arranging
    def compute_data_stat(self, 
                  data: pd.DataFrame, 
                  property = 'Te[K]',
                  to_csv: bool = False,
                  method: str = 'mean'):
        """
        data_stat sorts and arrange value by a group from the experimental data, calculates mean and standard deviation of the grouped data.
        Calculated result will be stored at the location of data files.

        args:
            data: pd.DataFrame, 
            property = 'Te[K]',
            to_csv: bool = False
            method: mean = 'mean' or 'std' or 'min' or 'max' etc.

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
                        .groupby(property, as_index=False)\
                        .agg(method)\
                        .dropna()
        
        # obj selection per group
        df_obj = data.groupby(by=property)[obj_cols].first()

        # concat
        df_out = pd.concat([df_num, df_obj], axis=1).reset_index()

        if to_csv:
            df_o = df_out.to_csv(self.datapath + f'grouped_{method}.csv')
            
            print(f"Calculated property saved at {self.datapath}'combined_{method}.csv'")
        return df_out
    
    #7
    # prepare average values for all thermal properties
    def data_property_avg(self, df_mean:pd.DataFrame, df_std:pd.DataFrame):
        """
        data_property_avg calculates average values of measured thermal properties for the given experiment data.

        usage: analysis.data_property_avg(df_mean, df_std)
        """
        # avg values 
        Tc_avg = df_mean['Tc[K]'].mean()
        P_avg = df_mean['P[bar]'].mean()
        dT_avg = df_mean['dT[K]'].mean()
        TR_avg = df_mean['TR[K/W]'].mean()
        GFE_avg = df_mean['GFE_Tc[KJ/mol]'].mean()
        # std values
        Tc_std = df_std['Tc[K]'].mean()
        P_std = df_std['P[bar]'].mean()
        dT_std = df_std['dT[K]'].mean()
        TR_std = df_std['TR[K/W]'].mean()
        GFE_std = df_std['GFE_Tc[KJ/mol]'].mean()
        # calculated results
        msg = (f"Tc  average:     {round(Tc_avg,4)} +- {round(Tc_std,4)} [K]\n"
        f"P   average:     {round(P_avg,4)} +- {round(P_std,4)} [bar]\n"
        f"dT  average:     {round(dT_avg,4)} +- {round(dT_std,4)} [K]\n"
        f"TR  average:     {round(TR_avg,4)} +- {round(TR_std,4)} [K/W]\n"
        f"GFE average:     {round(GFE_avg,4)} +- {round(GFE_std,4)} [KJ/mol]\n")
        return print(msg)
    
    #8
    # find optimal G(T,P) of PHP
    def best_TP(self, data:pd.DataFrame):
        """ 
        best_TP finds best G(T,P) with lowest dG (Change in Gibbs Free Energy for Te->Tc values at constant Pressure) from the experimental dataset.

        usage: analysis.best_TP(data)
        """
        df_opt = data[data['dG[KJ/mol]'] == data['dG[KJ/mol]'].min()]
        df_opt_idx = df_opt.index
        Tc_opt = data['Tc[K]'].loc[df_opt_idx]
        Te_opt = data['Te[K]'].loc[df_opt_idx]
        dT_opt = data['dT[K]'].loc[df_opt_idx]
        P_opt = data['P[bar]'].loc[df_opt_idx]
        dG_opt = data['dG[KJ/mol]'].loc[df_opt_idx]
        GFE_opt = data['GFE_Tc[KJ/mol]'].loc[df_opt_idx]
        TR_opt = data['TR[K/W]'].loc[df_opt_idx]
        msg = (f'Optimal G(T,P) condition at lowest (optimal) dG[{round(dG_opt.iloc[0],4)}]\n'
               f'Tc optimal:        {round(Tc_opt.iloc[0],4)}[K] \n'
               f'Te optimal:        {round(Te_opt.iloc[0],4)}[K] \n'
               f'P  optimal:        {round(P_opt.iloc[0],4)}[bar] \n'
               f'dT optimal:        {round(dT_opt.iloc[0],4)}[K] \n'
               f'TR optimal:        {round(TR_opt.iloc[0],4)}[K/W] \n'
               f'GFE optimal:       dG({round(Te_opt.iloc[0],4)}, {round(P_opt.iloc[0],4)}) = {round(GFE_opt.iloc[0],4)} [KJ/mol]\n')
        return print(msg)
    
## Data Visualisation
class DataVisualisation(PulseHeatPipe):
    """ ## Data Visualisation class to plot PHP data.

        ## usage: 
        ### importing module
        from analysis import DataVisualisation
        ### creating the reference variable
        visual = DataVisualisation('sample')
        ### data visualisation; eg. plotting all data
        visual.plot_all_data()
    """
    def __init__(self, datapath: str, sample: str):
        super().__init__(datapath, sample)
        
    def plot_all_data(self, data:pd.DataFrame):
        """ Data Visualisation of all data
            
            usage: visual.plot_all_data(data)
        """
        plt.figure(figsize=(10,5))
        sns.lineplot(data)
        plt.xlabel('Data')
        plt.ylabel('Properties')
        plt.title(f"All Data - {self.sample}")
        plt.legend()

    def plot_Te_Tc(self, data:pd.DataFrame):
        """ Data Visualisation of Te vs Tc
            
            usage: visual.plot_Te_Tc(data)
        """
        plt.figure(figsize=(10,5))
        plt.plot(data['Te[K]'], label = 'Te[K]')
        plt.plot(data['Tc[K]'], label = 'Tc[K]')
        plt.xlabel('Te[K]')
        plt.ylabel('Tc[K]')
        plt.title(f"Te[K] vs Tc[K] - {self.sample}")
        plt.legend()

    def plot_eu(self, df_mean:pd.DataFrame, df_std:pd.DataFrame, property:str, point='.k', eu='r'):
        """ Data Visualisation with expanded uncertainty
            
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
    
    