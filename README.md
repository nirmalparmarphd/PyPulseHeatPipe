# PulseHeatPipe

[PyPulseHeatPipe](https://pypi.org/project/PyPulseHeatPipe/) is a Python Library for data analysis and for data plotting/visualisation specifically for PHP experimental data.

### pkg installation
```
pip install PyPulseHeatPipe

# for pkg upgrade
pip install --upgrade PyPulseHeatPipe
```
## Usage:  
### importing the module
    from PyPulseHeatPipe import PulseHeatPipe
### creating the reference variable 
    analysis = PulseHeatPipe("datapath", "sample_name")
### for a class help 
    help(analysis)
### for a function help
    help(analysis.data_etl)
### using a function from the class
    df, df_conv = analysis.data_etl()
### to create blank file
    analysis.blank_file()

## list of available functions
0. blank_file
1. data_etl
2. gibbs_fe
3. data_chop
4. data_stat
5. data_property_avg
6. best_TP
7. plot_all_data
8. plot_Te_Tc
9. plot_eu

Example:
```
# importing module
from PyPulseHeatPipe import PulseHeatPipe
from PyPulseHeatPipe import DataVisualisation

analysis = PulseHeatPipe("datapaht", "sample_name")
visual = DataVisualisation("datapaht", "sample_name")

# calling help
help(analysis.data_etl)
help(visual.plot_all_data)

# using methods eg;
# for ETL
df, df_conv = analysis.data_etl()

# for visulisation of all thermal properties
visual.plot_all_data(df_gfe)

```
**NOTE**: The experimental data file must prepared in '.xlsx' format. The data must contain at least following columns with mentioned titles:

>**samle_data.xlsx format**

| t(min) | Tc[C] | Te[C] | P[bar] | Q[W] | alpha | beta | pulse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 35 | 700 | 80 | 90 | 0 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |

here,

'time'= timestamp,
'Te[C]'= Evaporator Temperature,
'Tc[C]'= Condenser Temperature,
'P[bar]'= Pressure (gauge) of PHP,
'Q[W]'= Power Supply,
'alpha'= Horizontal Angle of PHP,
'beta'= Vertical Angle of PHP, 
'pulse'= Visible pulse generation (y=1/n=0)

