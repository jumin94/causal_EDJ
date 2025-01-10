import xarray as xr
import numpy as np
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
import glob
import netCDF4
import random
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import scipy.stats as stats


def bayes_factor_RD(obs,sl,slmean):
    try:
        RSS0 = np.sum((obs.values - slmean.values)**2)
        RSS1 = np.sum((obs.values - sl.values)**2)
    except:
        RSS0 = np.sum((obs.values - slmean.values[:-1])**2)
        RSS1 = np.sum((obs.values - sl.values[:-1])**2)

    # Compute log-likelihood for Model 1
    n = len(obs.values)
    logL1 = -n/2 * np.log(RSS0/3) - n/2 * np.log(2*np.pi) - n/2
    # Compute BIC for Model 1 - tiene un parametro por lo tanto k1 = 2
    k1 = 2
    BIC1 = k1 * np.log(n) - 2 * logL1

    # Compute log-likelihood for Model 2 -
    logL2 = -n/2 * np.log(RSS1/3) - n/2 * np.log(2*np.pi) - n/2
    # Compute BIC for Model 1  tiene un parametro por lo tanto k1 = 2
    k2 = 2
    BIC2 = k2 * np.log(n) - 2 * logL2

    # Compute the log of the Bayes Factor
    log_BF21 = 0.5 * (BIC1 - BIC2)

    # Compute the Bayes Factor
    BF_MEM_SL = np.exp(log_BF21)
    return(BF_MEM_SL)


def seasonal_data(data,season='DJF'):
    # select DJF
    DA_DJF = data.sel(time = data.time.dt.season==season)

    # calculate mean per year
    DA_DJF = DA_DJF.groupby(DA_DJF.time.dt.year).mean("time")
    DA_DJF = DA_DJF.rename({'year':'time'})
    return DA_DJF

def seasonal_data_months(data, months):
    """
    Selects specified months from an xarray object and averages the data for those months within each year.
    
    Parameters:
    - data: xarray.DataArray or xarray.Dataset
        The input data to process. It should have a 'time' coordinate.
    - months: list of int
        The months to select for averaging (1 = January, 2 = February, ..., 12 = December).
    
    Returns:
    - xarray.DataArray or xarray.Dataset
        The averaged data for the selected months within each year.
    """
    # Ensure 'time' coordinate is in a format that supports .dt accessor
    if np.issubdtype(data['time'].dtype, np.datetime64):
        time_coord = data['time']
    else:
        time_coord = xr.cftime_range(start=data['time'][0].values, periods=data['time'].size, freq='M')
        data = data.assign_coords(time=time_coord)

    # Select specified months
    selected_months_data = data.sel(time=data['time'].dt.month.isin(months))
    
    # Group by year and average the selected months within each year
    averaged_data = selected_months_data.groupby('time.year').mean(dim='time')
    
    return averaged_data.rename({'year':'time'})

def make_xarr(data,time):
    time_series = xr.DataArray(
    data,
    coords=[time],
    dims=["time"],
    name="time_array")
    return time_series
    
def plot_mean_with_shading(ax, data, variable_name, data_source, title, obs, sl_high_high_l, sl_high_low_l, time, time_obs, sl_time, sl_time_long):
    """
    Plots the mean value across the 'ensemble' dimension with shading between
    the highest and lowest values for each time step on a given axis.
    
    Parameters:
    - ax: matplotlib.axes.Axes
        The axis to plot on.
    - data: xarray.Dataset
        The dataset containing the variable to plot.
    - variable_name: str
        The name of the variable to plot.
    - title: str
        The title of the subplot.
    """
    # Compute the mean, max, and min values across the 'ensemble' dimension
    mean_values = data.mean(dim='model')
    max_values = data.max(dim='model')
    min_values = data.min(dim='model')
    
    # Extract time and values for plotting
    mean_data = mean_values.values
    max_data = max_values.values
    min_data = min_values.values

    BF_rd_high = bayes_factor_RD(obs.sel(time=slice('1960', '2022')),
                                 make_xarr(sl_high_high_l.values, mean_values.sel(time=slice('1950', '2099')).time).sel(time=slice('1960', '2022')),
                                 mean_values.sel(time=slice('1960', '2022')))
    BF_rd_low = bayes_factor_RD(obs.sel(time=slice('1960', '2022')),
                                make_xarr(sl_high_low_l.values, mean_values.sel(time=slice('1950', '2099')).time).sel(time=slice('1960', '2022')),
                                mean_values.sel(time=slice('1960', '2022')))

    # Create the title with the Bayes factor text
    formatted_title = (
    f"P({data_source}|high {title})/P({data_source}|MEM)={round(BF_rd_high, 3)},\n"
    f"P({data_source}|low {title})/P({data_source}|MEM)={round(BF_rd_low, 3)}"
)
    ax.set_title(formatted_title, fontsize=14, loc='left')

    # Plot observed data
    if data_source == "JRA55":
        if len(obs.values) == 66:
            ax.plot(time.sel(time=slice('1958','2023')),obs.values, label=data_source, color='green',linewidth=2)
        else: 
            ax.plot(time.sel(time=slice('1958','2022')),obs.values, label=data_source, color='green',linewidth=2)
    else: 
        if len(obs.values) == 74:
            ax.plot(time.sel(time=slice('1950','2023')),obs.values, label=data_source, color='green',linewidth=2)
        elif len(obs.values) == 73:
            ax.plot(time.sel(time=slice('1950','2022')),obs.values, label=data_source, color='green',linewidth=2)
        else: 
            ax.plot(time.sel(time=slice('1950','2021')),obs.values, label=data_source, color='green',linewidth=2)

    # Plot model mean and spread
    if title == 'TW/GW':
        ax.plot(time, mean_data, label='CMIP6 hist+SSP5-8.5 MEM', color='black')
        ax.fill_between(time, min_data, max_data, color='grey', alpha=0.3, label='CMIP6 Spread')
        ax.plot(sl_time_long, sl_high_high_l.values, color='red', label=title + ', high storyline')
        ax.plot(sl_time_long, sl_high_low_l.values, color='blue', label=title + ', low storyline')
    else:
        ax.plot(time, mean_data, color='black')
        ax.fill_between(time, min_data, max_data, color='grey', alpha=0.3)
        ax.plot(sl_time_long, sl_high_high_l.values, color='red', label=title + ', high storyline')
        ax.plot(sl_time_long, sl_high_low_l.values, color='blue', label=title + ', low storyline')

    # Set labels and legend
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel(variable_name, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=12, length=5, width=1.5)
    
    if title == 'SPV/GW':
        ax.legend(fontsize=14, loc='lower right')
    else:
        ax.legend(fontsize=14, loc='upper center')


# Function to compute the 80% probability ellipse bounds
def find_80_percent_ellipse_values(data):
    mean_vector = data.mean()
    cov_matrix = np.cov(data.T)
    chi2_value = stats.chi2.ppf(0.80, df=2)  # 80% confidence level

    result_dict = {}
    for column in data.columns:
        mean = mean_vector[column]
        std_dev = np.sqrt(cov_matrix[data.columns.get_loc(column), data.columns.get_loc(column)])
        
        # Compute the lower and upper bounds
        lower_bound = mean - std_dev * np.sqrt(chi2_value)
        upper_bound = mean + std_dev * np.sqrt(chi2_value)
        
        result_dict[column] = {'mean': mean, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    
    return pd.DataFrame(result_dict).T


def create_figure_with_subplots(dataset, rd, variable_name,data_source,title,obs_dict_ts,storylines_dict_high_high_long, storylines_dict_high_low_long,time,time_obs,time_sl,time_sl_long):
    """
    Creates a figure with four subplots, each plotting the mean value with shading between
    the highest and lowest values for each time step.
    
    Parameters:
    - dataset: xarray.Dataset
        The dataset containing the variable to plot.
    - variable_name: str
        The name of the variable to plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10),dpi=300)
    
    for i in range(4):
        ax = axs.flat[i]
        plot_mean_with_shading(ax, dataset[rd[i]], variable_name[i],data_source[i],title[i],obs_dict_ts[rd[i]],storylines_dict_high_high_long[rd[i]], storylines_dict_high_low_long[rd[i]], time,time_obs,time_sl,time_sl_long)
    
    subplot_labels = ['a','b','c','d']
    # Add labels and grid to each subplot
    for i, ax in enumerate(axs.flat):
        # Set x-axis label only for the bottom row
        # if i in [2, 3]:  # Bottom subplots
        #     ax.set_xlabel('Year', fontsize=12)
        
        # Add subplot label in the top left corner of each subplot
        ax.text(0.05, 0.95, subplot_labels[i], transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='center')
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        if i != 5:
            # Add grid
            ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    return fig


def main(config):
    """Run the diagnostic."""
    #Reanalysis data
    ### Import ERA5 data
    ua_era5 = xr.open_dataset('/home/jmindlin/causal_EDJ/ERA5/ua_ERA5.nc')
    ua_era5 = ua_era5.rename({'latitude':'lat','longitude':'lon'})
    ua_era5_50 = ua_era5.sel(level=50)
    ua_era5_850 = ua_era5.sel(level=850)
    ta_era5 = xr.open_dataset('/home/jmindlin/causal_EDJ/ERA5/ta_ERA5.nc')
    ta_era5 = ta_era5.rename({'latitude':'lat','longitude':'lon'})

    ### Import NCEP data

    ua_ncep = xr.open_dataset('/home/jmindlin/causal_EDJ/NCEP/uwnd.mon.mean.nc')
    ua_ncep_50 = ua_ncep.sel(level=50)
    ua_ncep_850 = ua_ncep.sel(level=850)
    ta_ncep = xr.open_dataset('/home/jmindlin/causal_EDJ/NCEP/air.mon.mean.nc')

    del ua_era5, ua_ncep

    ### Import JRA55 data

    ua_jra55_50 = [xr.open_dataset('/home/jmindlin/causal_EDJ/JRA55/ua/anl_mdl.033_ugrd.reg_tl319.'+str(year)+'01_'+str(year)+'12.mindlin756630_50hPa.nc') for year in np.arange(1958,2024,1)]
    ua_jra55_50_concat = xr.concat(ua_jra55_50,dim='initial_time0_hours')
    ua_jra55_50_concat = ua_jra55_50_concat.rename({'initial_time0_hours':'time','g4_lat_2':'lat','g4_lon_3':'lon'})

    ua_jra55_850 = [xr.open_dataset('/home/jmindlin/causal_EDJ/JRA55/ua/anl_mdl.033_ugrd.reg_tl319.'+str(year)+'01_'+str(year)+'12.mindlin756630_847hPa.nc') for year in np.arange(1958,2024,1)]
    ua_jra55_850_concat = xr.concat(ua_jra55_850,dim='initial_time0_hours')
    ua_jra55_850_concat = ua_jra55_850_concat.rename({'initial_time0_hours':'time','g4_lat_2':'lat','g4_lon_3':'lon'})

    ta_jra55 = [xr.open_dataset('/home/jmindlin/causal_EDJ/JRA55/ta/anl_mdl.011_tmp.reg_tl319.'+str(year)+'01_'+str(year)+'12.mindlin754486.nc') for year in np.arange(1958,2024,1)]
    ta_jra55_concat = xr.concat(ta_jra55,dim='initial_time0_hours')
    ta_jra55_concat = ta_jra55_concat.rename({'initial_time0_hours':'time','g4_lat_2':'lat','g4_lon_3':'lon','lv_HYBL1':'lev'})

    import urllib.request

    # URL of the data file
    url = "https://crudata.uea.ac.uk/cru/data/temperature/HadCRUT5.0Analysis_gl.txt"

    # Fetch the data from the URL
    with urllib.request.urlopen(url) as response:
        lines = response.read().decode('utf-8').splitlines()

    # Parse the lines to extract the data
    data = []
    months = []
    years = []
    for line in lines[::2]:
        values = line.split(' ')[2:-1]
        years.append(line.split(' ')[1])
        for i, value in enumerate(values):
            if value != '':
                data.append(value)
                months.append(i)

    # Convert the list of lists into a NumPy array
    data_array = np.array(data, dtype=float)
    data_array = data_array

    # Print the resulting NumPy array
    print(data_array)

    time = pd.date_range(start='1850-01-01', end='2024-12-01', freq='MS')
    temperature_data = xr.DataArray(
        data_array, 
        coords={'time': time}, 
        dims='time', 
        name='temperature - HadCRU5'
    )

    tas_DJF = seasonal_data_months(temperature_data,[12,1,2])
    tas_DJF_anom = (tas_DJF - np.mean(tas_DJF.sel(time=slice('1950','1979')))).sel(time=slice('1950','2023'))

    u_1979_2019 = xr.open_dataset('/home/jmindlin/causal_EDJ/send_to_LIM/ERA5/ERA5/era5.mon.mean_T42.nc').u.sel(lev=50).drop_vars('lev')
    u_1950_1978 = xr.open_dataset('/home/jmindlin/causal_EDJ/send_to_LIM/ERA5/ERA5/ERA5_monthly_u_wind_n36_rename_regrid.nc').u.sel(plev=5000).drop_vars('plev')
    u_1950_2019 = xr.concat([u_1950_1978,u_1979_2019],'time')
    spv_era5_OND = seasonal_data_months(u_1950_2019,[10,11,12]).sel(lat=slice(-50,-60)).mean(dim='lat').mean(dim='lon').sel(time=slice('1950','2019'))
    spv_era5_1950_2019_OND = spv_era5_OND - spv_era5_OND.sel(time=slice('1950','1979')).mean(dim='time')
    CLIM_spv_era5 = spv_era5_1950_2019_OND.sel(time=slice('1950','2019')).mean(dim='time')

    spv_ncep_OND = seasonal_data_months(ua_ncep_50,[10,11,12]).sel(lat=slice(-50,-60)).mean(dim='lat').mean(dim='lon').sel(time=slice('1950','2023'))
    spv_ncep_1950_2023_OND = spv_ncep_OND - spv_ncep_OND.sel(time=slice('1950','1979')).mean(dim='time')
    CLIM_spv_ncep = spv_ncep_1950_2023_OND.sel(time=slice('1950','2023')).mean(dim='time')

    spv_jra55_OND = seasonal_data_months(ua_jra55_50_concat,[10,11,12]).sel(lat=slice(-50,-60)).mean(dim='lat').mean(dim='lon').sel(time=slice('1950','2023'))
    spv_jra55_1950_2023_OND = spv_jra55_OND - spv_jra55_OND.sel(time=slice('1950','1979')).mean(dim='time')
    CLIM_spv_jra55 = spv_jra55_1950_2023_OND.sel(time=slice('1950','2023')).mean(dim='time')

    ### Extend ERA5 SPV 2019-2023 with NCEP data
    spv_era5_1950_2023_OND = xr.concat([spv_era5_1950_2019_OND,spv_ncep_1950_2023_OND.uwnd.sel(time=slice('2020','2023'))],dim='time')

    stratospheric_polar_vortex_rean = []
    stratospheric_polar_vortex_rean.append(spv_era5_1950_2023_OND)
    stratospheric_polar_vortex_rean.append(spv_ncep_1950_2023_OND)
    stratospheric_polar_vortex_rean.append(spv_jra55_1950_2023_OND)

    tropical_warming = []
    tw_era5_DJF = seasonal_data_months(ta_era5,[12,1,2]).sel(lat=slice(15,-15)).mean(dim='lat').mean(dim='lon').sel(time=slice('1950','2023'))
    tw_era5_1950_2023_DJF = tw_era5_DJF - tw_era5_DJF.sel(time=slice('1950','1979')).mean(dim='time')
    tropical_warming.append(tw_era5_1950_2023_DJF.t)
    CLIM_tw_ncep = tw_era5_1950_2023_DJF.sel(time=slice('1950','2023')).mean(dim='time')

    tw_ncep_DJF = seasonal_data_months(ta_ncep,[12,1,2]).sel(level=250).sel(lat=slice(15,-15)).mean(dim='lat').mean(dim='lon').sel(time=slice('1950','2023')) + 273.15
    tw_ncep_1950_2023_DJF = tw_ncep_DJF - tw_ncep_DJF.sel(time=slice('1950','1979')).mean(dim='time')
    tropical_warming.append(tw_ncep_1950_2023_DJF)
    CLIM_tw_ncep = tw_ncep_1950_2023_DJF.sel(time=slice('1950','2023')).mean(dim='time')

    tw_jra55_DJF = seasonal_data_months(ta_jra55_concat,[12,1,2]).sel(lev=29).sel(lat=slice(15,-15)).mean(dim='lat').mean(dim='lon').sel(time=slice('1950','2023'))
    tw_jra55_1950_2023_DJF = tw_jra55_DJF - tw_jra55_DJF.sel(time=slice('1950','1979')).mean(dim='time')
    tropical_warming.append(tw_jra55_1950_2023_DJF)
    CLIM_tw_jra55 = tw_jra55_1950_2023_DJF.sel(time=slice('1950','2023')).mean(dim='time')

    ### SST data
    sst_ERSST = xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/sst.mnmean_ERSST_2022_KAPLAN_grid.nc') #- xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/sst.mnmean_ERSST_2022_KAPLAN_grid.nc').mean(dim='lon')
    sst_COBE = xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/sst.mon.mean_COBE_2022_KAPLAN_grid.nc')# - xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/sst.mon.mean_COBE_2022_KAPLAN_grid.nc').mean(dim='lon')
    sst_HadISST = xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/HadISST_sst_latest_KAPLAN_grid.nc') #- xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/HadISST_sst_latest_KAPLAN_grid.nc').mean(dim='lon')
    sst_Kaplan = xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/sst.mean.anom_Kaplan_2022_KAPLAN_grid.nc') #- xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/sst.mean.anom_Kaplan_2022_KAPLAN_grid.nc').mean(dim='lon')

    sst_ERSST_CP = sst_ERSST.sel(lon=slice(180,250)).sst.sel(lat=slice(-5,5)).mean(dim='lat').mean(dim='lon') 
    sst_ERSST_CP_DJF = seasonal_data_months(sst_ERSST_CP,[12,1,2])
    sst_ERSST_CP_DJF = sst_ERSST_CP_DJF - sst_ERSST_CP_DJF.sel(time=slice('1950','1979')).mean(dim='time')

    sst_ERSST_EP = sst_ERSST.sel(lon=slice(260,280)).sst.sel(lat=slice(0,10)).mean(dim='lat').mean(dim='lon')
    sst_ERSST_EP_DJF = seasonal_data_months(sst_ERSST_EP,[12,1,2])
    sst_ERSST_EP_DJF = sst_ERSST_EP_DJF - sst_ERSST_EP_DJF.sel(time=slice('1950','1979')).mean(dim='time')

    sst_COBE_CP = sst_COBE.sel(lon=slice(180,250)).sst.sel(lat=slice(-5,5)).mean(dim='lat').mean(dim='lon')
    sst_COBE_CP_DJF = seasonal_data_months(sst_COBE_CP,[12,1,2])
    sst_COBE_CP_DJF = sst_COBE_CP_DJF - sst_COBE_CP_DJF.sel(time=slice('1950','1979')).mean(dim='time')
    #sst_COBE_CP_DJF = sst_COBE_CP_DJF.sel(time=slice('1950','2018'))

    sst_COBE_EP = sst_COBE.sel(lon=slice(260,280)).sst.sel(lat=slice(0,10)).mean(dim='lat').mean(dim='lon')
    sst_COBE_EP_DJF = seasonal_data_months(sst_COBE_EP,[12,1,2])
    sst_COBE_EP_DJF = sst_COBE_EP_DJF - sst_COBE_EP_DJF.sel(time=slice('1950','1979')).mean(dim='time')
    #sst_COBE_EP_DJF = sst_COBE_EP_DJF.sel(time=slice('1950','2018'))

    sst_HadISST_CP = sst_HadISST.sel(lon=slice(180,250)).sst.sel(lat=slice(-5,5)).mean(dim='lat').mean(dim='lon')
    sst_HadISST_CP_DJF = seasonal_data_months(sst_HadISST_CP,[12,1,2])
    sst_HadISST_CP_DJF = sst_HadISST_CP_DJF - sst_HadISST_CP_DJF.sel(time=slice('1950','1979')).mean(dim='time')
    #sst_HadISST_CP_DJF = sst_HadISST_CP_DJF.sel(time=slice('1950','2018'))

    sst_HadISST_EP = sst_HadISST.sel(lon=slice(260,280)).sst.sel(lat=slice(0,10)).mean(dim='lat').mean(dim='lon')
    sst_HadISST_EP_DJF = seasonal_data_months(sst_HadISST_EP,[12,1,2])
    sst_HadISST_EP_DJF = sst_HadISST_EP_DJF - sst_HadISST_EP_DJF.sel(time=slice('1950','1979')).mean(dim='time')
    #sst_HadISST_EP_DJF = sst_HadISST_EP_DJF.sel(time=slice('1950','2018'))

    sst_Kaplan_CP = sst_Kaplan.sel(lon=slice(180,250)).sst.sel(lat=slice(-5,5)).mean(dim='lat').mean(dim='lon')
    sst_Kaplan_CP_DJF = seasonal_data_months(sst_Kaplan_CP,[12,1,2])
    #sst_Kaplan_CP_DJF = sst_Kaplan_CP_DJF.sel(time=slice('1950','2018'))

    sst_Kaplan_EP = sst_Kaplan.sel(lon=slice(260,280)).sst.sel(lat=slice(0,10)).mean(dim='lat').mean(dim='lon')
    sst_Kaplan_EP_DJF = seasonal_data_months(sst_Kaplan_EP,[12,1,2])
    #sst_Kaplan_EP_DJF = sst_Kaplan_EP_DJF.sel(time=slice('1950','2018'))

    sst_CP_obs = [sst_ERSST_CP_DJF,sst_COBE_CP_DJF,sst_HadISST_CP_DJF,sst_Kaplan_CP_DJF]
    sst_EP_obs = [sst_ERSST_EP_DJF,sst_COBE_EP_DJF,sst_HadISST_EP_DJF,sst_Kaplan_EP_DJF]

    obs_dict_ts = {'gw':tas_DJF_anom.sel(time=slice('1950','2023')),'ta':tropical_warming[0].sel(time=slice('1950','2023')),
                   'tos_cp':sst_ERSST_CP_DJF.sel(time=slice('1950','2023')),'tos_ep':sst_ERSST_EP_DJF.sel(time=slice('1950','2023')),
                   'ua50_spv':stratospheric_polar_vortex_rean[0].sel(time=slice('1950','2022'))}

    obs_dict_ts_jra55 = {'gw':tas_DJF_anom.sel(time=slice('1950','2023')),'ta':tropical_warming[2].TMP_GDS4_HYBL_S123.sel(time=slice('1950','2023')),
                   'tos_cp':sst_HadISST_CP_DJF.sel(time=slice('1950','2023')),'tos_ep':sst_HadISST_EP_DJF.sel(time=slice('1950','2023')),
                   'ua50_spv':stratospheric_polar_vortex_rean[2].UGRD_GDS4_HYBL_S123.sel(time=slice('1950','2022'))}

    obs_dict_ts_ncep = {'gw':tas_DJF_anom.sel(time=slice('1950','2023')),'ta':tropical_warming[1].air.sel(time=slice('1950','2023')),
                   'tos_cp':sst_COBE_CP_DJF.sel(time=slice('1950','2023')),'tos_ep':sst_COBE_EP_DJF.sel(time=slice('1950','2023')),
                   'ua50_spv':stratospheric_polar_vortex_rean[1].uwnd.sel(time=slice('1950','2022'))}

    eesc_ts = pd.read_csv('/home/jmindlin/causal_EDJ/send_to_LIM/GW_EESC_polar_ozoneloss.csv')
    for i in range(len(eesc_ts[:8])):
        eesc_ts['EESC_polar'][i] = eesc_ts['EESC_polar'][8]

    eesc_ts_long = eesc_ts - eesc_ts['EESC_polar'][8]
    #eesc_ts['EESC_polar'][10:79] = eesc_ts_long['EESC_polar'][10:79] - eesc_ts_long['EESC_polar'][8]

    storylines_dict_high = {'gw':tas_DJF_anom.sel(time=slice('1950','2023')),'ta':tas_DJF_anom.sel(time=slice('1950','2023'))*1.82,
                   'tos_cp':tas_DJF_anom.sel(time=slice('1950','2023'))*0.95,'tos_ep':tas_DJF_anom.sel(time=slice('1950','2023'))*0.95,
                   'ua50_spv':eesc_ts_long['EESC_polar'][:74]*0.003 + tas_DJF_anom.sel(time=slice('1950','2023'))*2.3}

    storylines_dict_low = {'gw':tas_DJF_anom.sel(time=slice('1950','2023')),'ta':tas_DJF_anom.sel(time=slice('1950','2023'))*1.55,
                   'tos_cp':tas_DJF_anom.sel(time=slice('1950','2023'))*(0.6),'tos_ep':tas_DJF_anom.sel(time=slice('1950','2023'))*(0.6),
                   'ua50_spv':eesc_ts_long['EESC_polar'][:74]*0.001 + tas_DJF_anom.sel(time=slice('1950','2023'))*0.9}

    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    #print(cfg)
    meta_dataset = group_metadata(config["input_data"].values(), "dataset")
    models = []
    rd_list_models = []
    regressors_members = {}
    for dataset, dataset_list in meta_dataset.items(): ####DATASET es el modelo
        meta = group_metadata(config["input_data"].values(), "alias")
        if dataset != 'E3SM-1-0':
            print(f"Evaluate for {dataset}\n")
            models.append(dataset)
            rd_list_members = []
            for alias, alias_list in meta.items(): ###ALIAS son los miembros del ensemble para el modelo DATASET
                ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1950','2099')) -  xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1940','1969')).mean(dim='time') for m in alias_list if (m["dataset"] == dataset) & (m["variable_group"] != 'ua850') & (m["variable_group"] != 'sst') & (m["variable_group"] != 'pr') & (m["variable_group"] != 'tos_zm') }
                if ('gw' in ts_dict.keys()) & (dataset == 'ACCESS-CM2'):
                    rd_list_members.append(ts_dict)
                    time = ts_dict['gw'].sel(time=slice('1950','2099')).time ### Model ensemble Means.time
                    time_obs = ts_dict['gw'].sel(time=slice('1950','2023')).time
                    time_sl = ts_dict['gw'].sel(time=slice('1950','2023')).time
                    time_sl_long = ts_dict['gw'].sel(time=slice('1950','2099')).time
                elif('gw' in ts_dict.keys()):
                    rd_list_members.append(ts_dict)
                else:
                    a = 'nada'

            #Index create data array
            regressor_names = rd_list_members[0].keys()
            regressors_members[dataset] = {}
            for rd in regressor_names:
                list_values = [rd_list_members[m][rd] for m,ensemble in enumerate(rd_list_members)]
                regressors_members[dataset][rd] = xr.concat(list_values, dim='ensemble') # Ensemble for each model 
                regressors_members[dataset][rd]['time'] = time

    regressor_names = rd_list_members[0].keys()
    regressors_members_MEM = {rd: xr.concat([regressors_members[ensemble_mean][rd].mean(dim='ensemble').sel(time=slice('1950','2099'))  for ensemble_mean in models], dim='model') for rd in regressor_names} ### Model ensemble Means
    regressors_members_MMEM = {rd: regressors_members_MEM[rd].mean(dim='model').sel(time=slice('1950','2099')) for rd in regressor_names} ### Model ensemble Means
    print(regressors_members_MEM)
    
    for rd in obs_dict_ts.keys():
        if rd == 'ua50_spv':
            obs_dict_ts[rd]['time'] = time_obs[:-2]
        else:
            obs_dict_ts[rd]['time'] = time_obs

    drivers = pd.read_csv(config["work_dir"]+'/remote_drivers/raw_remote_drivers_tropical_warming_global_warming_scaledGW.csv', index_col=0)
    sl_values = find_80_percent_ellipse_values(drivers)
    high_GW = 1

    storylines_dict_high_high_long = {'ta':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW*sl_values['upper_bound']['ta'],
                   'tos_cp':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW*sl_values['upper_bound']['tos_cp'],'tos_ep':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW*sl_values['upper_bound']['tos_ep'],
                    'ua50_spv':eesc_ts_long['EESC_polar'][:-1]*0.002 + regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW*sl_values['upper_bound']['ua50_spv']}

    storylines_dict_high_low_long = {'ta':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW*sl_values['lower_bound']['ta'],
                   'tos_cp':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW*sl_values['lower_bound']['tos_cp'],'tos_ep':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW*sl_values['lower_bound']['tos_ep'],
                   'ua50_spv':eesc_ts_long['EESC_polar'][:-1]*0.002 + regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW*sl_values['lower_bound']['ua50_spv']}

    regressors_members_MEM_woGW = {rd: xr.concat([regressors_members[ensemble_mean][rd].mean(dim='ensemble').sel(time=slice('1950','2099'))  for ensemble_mean in models], dim='model') for rd in storylines_dict_high_low_long.keys()} ### Model ensemble Means

    fig = create_figure_with_subplots(regressors_members_MEM_woGW,list(storylines_dict_high_low_long.keys()),
                                      ['Tropical Warming [K]','Central Pacific Warming [K]','Eastern Pacific Warming [K]','Stratospheric Polar Vortex [m/s]'],
                                      ['ERA5','ERSSTv5','ERSSTv5','ERA5'],
                                      ['TW/GW','CP/GW','EP/GW','SPV/GW'],obs_dict_ts, storylines_dict_high_high_long, storylines_dict_high_low_long,
                                      time=time,time_obs=time_sl_long.sel(time=slice('1950','2023')),time_sl=time_sl,time_sl_long=time_sl_long)

    fig2 = create_figure_with_subplots(regressors_members_MEM_woGW,list(storylines_dict_high_low_long.keys()),
                                      ['Tropical Warming [K]','Central Pacific Warming [K]','Eastern Pacific Warming [K]','Stratospheric Polar Vortex [m/s]'],
                                      ['JRA55','HadISST','HadISST','JRA55'],
                                      ['TW/GW','CP/GW','EP/GW','SPV/GW'],obs_dict_ts_jra55, storylines_dict_high_high_long, storylines_dict_high_low_long,
                                      time=time,time_obs=time_sl_long.sel(time=slice('1950','2023')),time_sl=time_sl,time_sl_long=time_sl_long)

    fig3 = create_figure_with_subplots(regressors_members_MEM_woGW,list(storylines_dict_high_low_long.keys()),
                                      ['Tropical Warming [K]','Central Pacific Warming [K]','Eastern Pacific Warming [K]','Stratospheric Polar Vortex [m/s]'],
                                      ['NCEP','COBE','COBE','NCEP'],
                                      ['TW/GW','CP/GW','EP/GW','SPV/GW'],obs_dict_ts_ncep, storylines_dict_high_high_long, storylines_dict_high_low_long,
                                      time=time,time_obs=time_sl_long.sel(time=slice('1950','2023')),time_sl=time_sl,time_sl_long=time_sl_long)


    #Create directories to store results
    # os.chdir(config["work_dir"])
    # os.getcwd()
    # os.makedirs("remote_drivers_storylines",exist_ok=True)
    # df = pd.DataFrame(storylines_dict_high_high_long)
    # df.to_csv(config["work_dir"]+'/remote_drivers/remote_drivers_storylines_high_high.csv')
    # df = pd.DataFrame(storylines_dict_high_low_long)
    # df.to_csv(config["work_dir"]+'/remote_drivers/remote_drivers_storylines_high_low.csv')

    os.chdir(config["plot_dir"])
    os.getcwd()
    os.makedirs("remote_drivers",exist_ok=True)
    fig.savefig(config["plot_dir"]+'/remote_drivers/remote_driver_time_series_rean_storylines_nicer_MEMGW_ERA5.png')

    os.chdir(config["plot_dir"])
    os.getcwd()
    os.makedirs("remote_drivers",exist_ok=True)
    fig2.savefig(config["plot_dir"]+'/remote_drivers/remote_driver_time_series_rean_storylines_nicer_MEMGW_JRA55.png')

    os.chdir(config["plot_dir"])
    os.getcwd()
    os.makedirs("remote_drivers",exist_ok=True)
    fig3.savefig(config["plot_dir"]+'/remote_drivers/remote_driver_time_series_rean_storylines_nicer_MEMGW3_NCEP.png')



if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                              
