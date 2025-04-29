import xarray as xr
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import urllib.request
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import cartopy.crs as ccrs
import matplotlib.path as mpath
import matplotlib as mpl
import matplotlib.patches as mpatches
from cartopy.util import add_cyclic_point
from scipy.signal import detrend

def jet_lat_strength(jet_data,lon1=-180,lon2=180):
    if np.max(jet_data.lon.values) > 300:
        ds = jet_data.assign_coords(lon=((jet_data.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")
        jet_data = ds
    else:
        jet_data = jet_data
    jet_30_70 = jet_data.sel(lat=slice(-30,-70)).sel(lon=slice(lon1,lon2)).mean(dim='lon')
    lat = jet_30_70.lat
    jet_lat = (jet_30_70*lat).sum(dim='lat')/(jet_30_70).sum(dim='lat')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(lat=max_lat,method='nearest').sel(lon=slice(lon1,lon2)).mean(dim='lon'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength

def jet_lat_strength_model(jet_data,lon1=-180,lon2=180):
    if np.max(jet_data.lon.values) > 300:
        ds = jet_data.assign_coords(lon=((jet_data.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")
        jet_data = ds
    else:
        jet_data = jet_data
    jet_30_70 = jet_data.sel(lat=slice(-70,-30)).sel(lon=slice(lon1,lon2)).mean(dim='lon')
    lat = jet_30_70.lat
    jet_lat = (jet_30_70*lat).sum(dim='lat')/(jet_30_70).sum(dim='lat')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(lat=max_lat,method='nearest').sel(lon=slice(lon1,lon2)).mean(dim='lon'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength

def anom(x):
    return x - x.sel(time=slice('1950','1979')).mean(dim='time')

def anom_array(x):
    return x - np.mean(x[:30])

def std_anom_past(x):
    std_x = (x - x.sel(time=slice('1950','1979')).mean(dim='time'))/x.sel(time=slice('1950','2023')).std(dim='time')
    return std_x 

def std(x):
    std_x = x/x.std(dim='time')
    return std_x 

def std_bias_corr_xr(x,y):
    std_x = (x - x.sel(time=slice('1950','1979')).mean(dim='time'))/np.std(y)
    return std_x 

def std_bias_corr_np(x,y):
    std_x = (x - np.mean(x[:30]))/np.std(y)
    return std_x 

def make_xarr(data,time):
    time_series = xr.DataArray(
    data,
    coords=[time],
    dims=["time"],
    name="time_array")
    return time_series


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
        The averaged data for the selected months within each year, accounting for months that span across years.
    """
    # Ensure 'time' coordinate is in a format that supports .dt accessor
    if np.issubdtype(data['time'].dtype, np.datetime64):
        time_coord = data['time']
    else:
        time_coord = xr.cftime_range(start=data['time'][0].values, periods=data['time'].size, freq='M')
        data = data.assign_coords(time=time_coord)

    # Select the relevant months and keep track of the original years
    selected_months_data = data.sel(time=data['time'].dt.month.isin(months))

    # Create a new time coordinate for grouping
    new_years = selected_months_data['time'].dt.year.values.copy()

    # Shift the year for December, if necessary
    if 12 in months:
        dec_mask = selected_months_data['time'].dt.month == 12
        new_years[dec_mask] += 1  # Increment year for December

    # Assign the new year as a coordinate to the selected data
    selected_months_data = selected_months_data.assign_coords(new_year=("time", new_years))

    # Now group by the new year and calculate the mean
    averaged_data = selected_months_data.groupby("new_year").mean(dim="time")

    # Rename the new year dimension to 'time' for consistency
    averaged_data = averaged_data.rename({"new_year": "time"})

    return averaged_data

def seasonal_data(data,season='DJF'):
    # select DJF
    DA_DJF = data.sel(time = data.time.dt.season==season)

    # calculate mean per year
    DA_DJF = DA_DJF.groupby(DA_DJF.time.dt.year).mean("time")
    DA_DJF = DA_DJF.rename({'year':'time'})
    return DA_DJF


def evaluate_similarity(x1, x2):
    """
    Evaluates the similarity metric between two time series x1 and x2 using the formula:
    (1 / (sqrt(2 * pi * sigma))) * exp(-((sum((x1 - x2)**2)) / (2 * sigma**2))),
    where sigma is the standard deviation of the detrended time series x1.
    
    Parameters:
        x1 (np.ndarray): The first time series.
        x2 (np.ndarray): The second time series.
        
    Returns:
        float: The result of the similarity evaluation.
    """
    if len(x1) != len(x2):
        raise ValueError("Time series x1 and x2 must have the same length.")
    
    # Detrend x1 by removing its mean
    x1_detrended = x1 - np.mean(x1)
    
    # Compute the standard deviation of the detrended x1
    sigma = np.std(x1_detrended)
    
    if sigma == 0:
        raise ValueError("Standard deviation of the detrended x1 is zero, computation not possible.")
    
    # Compute the similarity metric
    similarity = (1 / (np.sqrt(2 * np.pi * sigma))) * np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma ** 2))
    
    return similarity


def bayes_factor_RD(obs,sl,slmean):
    PS0 = evaluate_similarity(obs,slmean)
    PS1 = evaluate_similarity(obs, sl)

    # Compute the Bayes Factor
    BF_MEM_SL = PS1 / PS0
    return(BF_MEM_SL)


def anom(x):
    return x - np.mean(x[:30])


def detrend_timeseries(da):
    """
    Remove the linear trend from an xarray DataArray.

    Parameters:
    - da: xarray.DataArray
        The input time series to detrend.

    Returns:
    - detrended_da: xarray.DataArray
        The detrended time series.
    """
    # Check if the input is a DataArray
    if not isinstance(da, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray")

    # Ensure that there is a 'time' coordinate
    if 'time' not in da.coords:
        raise ValueError("DataArray must have a 'time' coordinate")

    # Detrend the data along the time axis
    detrended_data = detrend(da, axis=0)

    # Return as a new DataArray, preserving the original coordinates
    detrended_da = xr.DataArray(detrended_data, dims=da.dims, coords=da.coords)
    
    return detrended_da


def standardize_time(ds):
    """Convert dataset time coordinates to a standard calendar."""
    if "time" in ds.coords:
        ds["time"] = xr.decode_cf(ds).time  # Decode time while handling different calendars
    return ds


def plot_mean_with_shading_min_max_lowSL(ax, era5,mean_data,std_data,mean_data_recon,std_data_recon,list_story, time,timeobs,labels_sl):
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
    # Plot mean values
    handles = []
    a = ax.plot(time,mean_data, label='CMIP6 MEM', color='grey')
    handles.append(a)
    
    # Shade between max and min values
    b = ax.fill_between(time,std_data[0], std_data[1], color='k', alpha=0.1, label='CMIP6 spread ')
    handles.append(b)

    
    # Plot recon mean values
    handles = []
    c = ax.plot(time,mean_data_recon[0:91], label='CMIP6 MEM reconstructed', color='r')
    handles.append(c)
    
    # Shade between max and min values
    d = ax.fill_between(time,std_data_recon[0][0:91], std_data_recon[1][0:91], color='r', alpha=0.1, label='CMIP6 reconstructed spread')
    handles.append(d)

    line = ax.plot(timeobs,era5, label='ERA5', color='k')
    handles.append(line)

    line = ax.plot(timeobs,era5*0, label=' ', alpha=0.001, color='k')
    handles.append(line) 

    # Define a custom color palette (8 warm colors, 8 cool colors)
    colors_warm = plt.cm.YlOrRd(np.linspace(0.2, 1, 8))
    colors_cool = plt.cm.PuBu(np.linspace(0.2, 1, 8))
    colors_sl = np.vstack([colors_cool,colors_warm])

    for i,sl in enumerate(list_story):
        if 'TW -' in labels_sl[i]:
            line = ax.plot(time,sl, label=labels_sl[i], color=colors_sl[i],alpha=0.9)
        else:
            line = ax.plot(time,sl, label=labels_sl[i], color=colors_sl[i],alpha=0.9,linestyle=':')
        handles.append(line)

    # line = ax.plot(time,mean_story, label='mean storyline', color='purple')
    # handles.append(line)
    # Set labels and title
    ax.set_xlabel('Time')
    return ax, handles


def create_figure_with_subplots_min_max_lowSLs(era5, mean_data, std_data,mean_data_recon, std_data_recon, list_story,time, timeobs,labels_sl):
    """
    Creates a figure with subplots, each plotting the mean value with shading between
    the highest and lowest values for each time step, and a unified legend below.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 8),dpi=300)
    
    # Initialize empty lists to collect handles and labels
    handles = []
    labels = ['CMIP6 MEM','CMIP6 spread','CMIP6 MEM \n reconstructed','CMIP6 spread \n reconstructed','ERA5',' ']
    labels.extend(labels_sl)

    # Iterate over subplots
    for i, ax in enumerate(axs.flat):
        ax, subplot_handles = plot_mean_with_shading_min_max_lowSL(
            ax, era5[i][:], mean_data[i], std_data[i], mean_data_recon[i], std_data_recon[i], list_story[i], time, timeobs, labels_sl
        )
        
        # Flatten subplot_handles and extend the main handles list
        for h in subplot_handles:
            if isinstance(h, list):  # Handle cases where plot returns a list of Line2D objects
                handles.extend(h)
            else:
                handles.append(h)


    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    y_labels = [
        'Zonal mean \n EDJ latitude [degrees]', 
        'Zonal mean \n EDJ strength [m/s]', 
        'Pacific basin \n EDJ latitude [degrees]', 
        'Pacific basin \n EDJ strength [m/s]',
        'Atlantic-Indian basin \n EDJ latitude [degrees]', 
        'Atlantic-Indian basin \n EDJ strength [m/s]'
    ]

    # Customize each subplot
    for i, ax in enumerate(axs.flat):
        ax.set_ylabel(y_labels[i], fontsize=12)
        if i in [4, 5]:  # Bottom subplots
            ax.set_xlabel('Year', fontsize=12)
        ax.text(0.05, 0.95, subplot_labels[i], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    # Adjust layout to make space for the legend below the subplots
    fig.subplots_adjust(bottom=0.15)

    # Create a single unified legend below all subplots
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_handles.append(h)
            unique_labels.append(l)

    fig.legend(handles=unique_handles, labels=unique_labels, 
               loc='lower center', 
               ncol=5, 
               bbox_to_anchor=(0.5, -0.2))

    plt.tight_layout()
    plt.show()
    return fig


### Functions

def regressor_EESC_GW(gw_ts):
    eesc_ts = pd.read_csv('/home/jmindlin/causal_EDJ/send_to_LIM/GW_EESC_polar_ozoneloss.csv')
    for i in range(len(eesc_ts[:8])):
        eesc_ts['EESC_polar'][i] = eesc_ts['EESC_polar'][8]
    df = pd.DataFrame({'EESC':eesc_ts['EESC_polar'][10:79] - eesc_ts['EESC_polar'][8],'GW':gw_ts})
    regressors_out = sm.add_constant(df.values)
    return regressors_out, df

def stand_detr(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return signal.detrend(anom)

def filtro(dato):
    """Apply a rolling mean of 5 years and remov the NaNs resulting bigining and end"""
    signal = dato - dato.rolling(time=10, center=True).mean()
    signal_out = signal.dropna('time', how='all')
    return signal_out
                          
def stand(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return anom

def replace_nans_with_zero(x):
    return np.where(np.isnan(x), random.random(), x)

def figure(target,predictors):
    fig = plt.figure()
    y = predictors.apply(stand_detr,axis=0).values
    for i in range(len(predictors.keys())):
        plt.plot(y[:,i])
    plt.plot(stand_detr(target))
    return fig

def jet_lat_strength(jet_data,lon1=-180,lon2=180):
    if np.max(jet_data.lon.values) > 300:
        ds = jet_data.assign_coords(lon=((jet_data.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")
        jet_data = ds
    else:
        jet_data = jet_data
    jet_30_70 = jet_data.sel(lat=slice(-30,-70)).sel(lon=slice(lon1,lon2)).mean(dim='lon')
    lat = jet_30_70.lat
    jet_lat = (jet_30_70*lat).sum(dim='lat')/(jet_30_70).sum(dim='lat')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(lat=max_lat,method='nearest').sel(lon=slice(lon1,lon2)).mean(dim='lon'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength

def jet_lat_strength_model(jet_data,lon1=-180,lon2=180):
    if np.max(jet_data.lon.values) > 300:
        ds = jet_data.assign_coords(lon=((jet_data.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")
        jet_data = ds
    else:
        jet_data = jet_data
    jet_30_70 = jet_data.sel(lat=slice(-70,-30)).sel(lon=slice(lon1,lon2)).mean(dim='lon')
    lat = jet_30_70.lat
    jet_lat = (jet_30_70*lat).sum(dim='lat')/(jet_30_70).sum(dim='lat')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(lat=max_lat,method='nearest').sel(lon=slice(lon1,lon2)).mean(dim='lon'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength

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
        The averaged data for the selected months within each year, accounting for months that span across years.
    """
    # Ensure 'time' coordinate is in a format that supports .dt accessor
    if np.issubdtype(data['time'].dtype, np.datetime64):
        time_coord = data['time']
    else:
        time_coord = xr.cftime_range(start=data['time'][0].values, periods=data['time'].size, freq='M')
        data = data.assign_coords(time=time_coord)

    # Select the relevant months and keep track of the original years
    selected_months_data = data.sel(time=data['time'].dt.month.isin(months))

    # Create a new time coordinate for grouping
    new_years = selected_months_data['time'].dt.year.values.copy()

    # Shift the year for December, if necessary
    if 12 in months:
        dec_mask = selected_months_data['time'].dt.month == 12
        new_years[dec_mask] += 1  # Increment year for December

    # Assign the new year as a coordinate to the selected data
    selected_months_data = selected_months_data.assign_coords(new_year=("time", new_years))

    # Now group by the new year and calculate the mean
    averaged_data = selected_months_data.groupby("new_year").mean(dim="time")

    # Rename the new year dimension to 'time' for consistency
    averaged_data = averaged_data.rename({"new_year": "time"})

    return averaged_data


#Across models regression class
class spatial_MLR(object):
    def __init__(self):
        self.what_is_this = 'This performs a regression across models and plots everything'
    
    def regression_data(self,variable,regressors,regressor_names,dataset):
        """Define the regression target variable 
        this is here to be edited if some opperation is needed on the DataArray
        
        :param variable: DataArray
        :return: target variable for the regression  
        """
        self.dataset = dataset
        self.target = variable
        regressor_indices = regressors
        self.regression_y = sm.add_constant(regressors.values)
        self.regressors = regressors.values
        self.rd_num = len(regressor_names) 
        self.regressor_names = regressor_names

    #Regresion lineal
    def linear_regression(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        returns = [res.params[i] for i in range(self.rd_num)]
        return tuple(returns)

    def linear_regression_pvalues(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        returns = [res.pvalues[i] for i in range(self.rd_num)]
        return tuple(returns)
    
    def linear_regression_R2(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        return res.rsquared
    

    def perform_regression(self,path,var): 
        """ Performs regression over all gridpoints in a map and returns and saves DataFrames
        
        :param path: saving path
        :return: none
        """
        
        target_var = xr.apply_ufunc(replace_nans_with_zero, self.target)
        results = xr.apply_ufunc(self.linear_regression,target_var,input_core_dims=[["time"]],
                                 output_core_dims=[[] for i in range(self.rd_num)],
                                 vectorize=True,
                                 dask="parallelized")
        results_pvalues = xr.apply_ufunc(self.linear_regression_pvalues,target_var,input_core_dims=[["time"]],
                                 output_core_dims=[[] for i in range(self.rd_num)],
                                 vectorize=True,
                                 dask="parallelized")
        results_R2 = xr.apply_ufunc(self.linear_regression_R2,target_var,input_core_dims=[["time"]],
                                 output_core_dims=[[]],
                                 vectorize=True,
                                 dask="parallelized")
        
      
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs = results[0].to_dataset(name='const')
            else:
                regression_coefs[self.regressor_names[i]] = results[i]
                
        print('This is regressor_coefs:',regression_coefs)
        if var == 'ua':
            regression_coefs = regression_coefs.rename({'ua':self.regressor_names[0]})
        elif var == 'sst':
            regression_coefs = regression_coefs.rename({'tos':self.regressor_names[0]})
        elif var == 'tas':
            regression_coefs = regression_coefs.rename({'tas':self.regressor_names[0]})
        elif var == 'pr':
            regression_coefs = regression_coefs.rename({'pr':self.regressor_names[0]})
        else:
            'done'
            #regression_coefs = regression_coefs.rename({var:self.regressor_names[0]})
        regression_coefs.to_netcdf(path+'/'+var+'/regression_coefficients_'+self.dataset+'.nc')
        
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs_pvalues = results_pvalues[0].to_dataset(name='const')
            else:
                regression_coefs_pvalues[self.regressor_names[i]] = results_pvalues[i]        
        if var == 'ua':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'ua':self.regressor_names[0]})
        elif var == 'sst':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'tos':self.regressor_names[0]})
        elif var == 'tas':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'tas':self.regressor_names[0]})
        elif var == 'pr':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'pr':self.regressor_names[0]})
        else:
            'done'
            #regression_coefs_pvalues = regression_coefs_pvalues.rename({var:self.regressor_names[0]})
        regression_coefs_pvalues.to_netcdf(path+'/'+var+'/regression_coefficients_pvalues_'+self.dataset+'.nc')
        

        results_R2.to_netcdf(path+'/'+var+'/R2_'+self.dataset+'.nc')
                     
        
    def create_x(self,i,j,dato):
        """ For each gridpoint creates an array and standardizes it 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """    
        x = np.array([])
        for y in range(len(dato.time)):
            aux = dato.isel(time=y)
            x = np.append(x,aux[i-1,j-1].values)
        return stand(x)
     
    
    def open_regression_coef(self,path,var,dataset):
        """ Open regression coefficients and pvalues to plot
        :param path: saving path
        :return maps: list of list of coefficient maps
        :return maps_pval:  list of coefficient pvalues maps
        :return R2: map of fraction of variance
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/'+var+'/regression_coefficients_'+dataset+'.nc')
        coef_pvalues = xr.open_dataset(path+'/'+var+'/regression_coefficients_pvalues_'+dataset+'.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/'+var+'/R2_'+dataset+'.nc')
        return maps, maps_pval, R2    

    def open_lmg_coef(self,path,var):
        """ Open regression coefficients and pvalues to plot
        :param path: saving path
        :return maps: list of list of coefficient maps
        :return maps_pval:  list of coefficient pvalues maps
        :return R2: map of fraction of variance
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/'+var+'/regression_coefficients_relative_importance.nc')
        coef_pvalues = xr.open_dataset(path+'/'+var+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names[1:]]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/'+var+'/R2.nc')
        return maps, maps_pval, R2    
    
    def plot_regression_lmg_map(self,path,var,output_path):
        """ Plots figure with all of 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_lmg_coef(path,var)
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection_stereo = ccrs.SouthPolarStereo(central_longitude=300)
        projection_plate = ccrs.PlateCarree(180)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num-1):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic
            if var == 'ua':
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
                ax.set_extent([0,359.9, -90, 0], crs=data_crs)
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            elif var == 'sst':
                ax = plt.subplot(3,3,k+1,projection=projection_plate)
            else: 
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
            clevels = np.arange(0,40,2)
            im=ax.contourf(lon_c, lat, var_c*100,clevels,transform=data_crs,cmap='OrRd',extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[k+1].min() < 0.05: 
                levels = [maps_pval[k+1].min(),0.05,maps_pval[k+1].max()]
                ax.contourf(maps_pval[k+1].lon, lat, maps_pval[k+1].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[k+1].min() < 0.10:
                levels = [maps_pval[k+1].min(),0.10,maps_pval[k+1].max()]
                ax.contourf(maps_pval[k+1].lon, lat, maps_pval[k+1].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k+1]) 
            plt.title(self.regressor_names[k+1],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
        plt1_ax = plt.gca()
        left, bottom, width, height = plt1_ax.get_position().bounds
        if var == 'ua':
            colorbar_axes1 = fig_coef.add_axes([left+0.5, bottom, 0.01, height*2])
        elif var == 'sst':
            colorbar_axes1 = fig_coef.add_axes([left+0.3, bottom, 0.01, height*2])    
        else:
            colorbar_axes1 = fig_coef.add_axes([left+0.5, bottom, 0.01, height*2])
        cbar = fig_coef.colorbar(im, colorbar_axes1, orientation='vertical')
        cbar.set_label('relative importance',fontsize=14) #rotation = radianes
        cbar.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        if var == 'ua':
            plt.savefig(output_path+'/regression_coefficients_relative_importance_u850',bbox_inches='tight')
        elif var == 'sst':
            plt.savefig(output_path+'/regression_coefficients_relative_importance_sst',bbox_inches='tight')
        else:
            plt.savefig(output_path+'/regression_coefficients_relative_importance_XXX',bbox_inches='tight')   
        plt.clf

        return fig_coef


    def plot_regression_coef_map(self,path,var,output_path):
        """ Plots figure with all of 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_regression_coef(path,var,self.dataset)
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        u_ERA = xr.open_dataset('/home/jmindlin/causal_EDJ/ERA5/ua_ERA5.nc')
        u_ERA = u_ERA.u.sel(level=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.rename({'longitude':'lon','latitude':'lat'})
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection_stereo = ccrs.SouthPolarStereo(central_longitude=300)
        projection_plate = ccrs.PlateCarree(180)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic for winds
            if var == 'u':
                ax = plt.subplot(3,2,k+1,projection=projection_stereo)
                ax.set_extent([0,359.9, -90, 0], crs=data_crs)
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            #Plate Carree map for SST
            elif var == 'sst':
                ax = plt.subplot(3,2,k+1,projection=projection_plate)
            else: 
                ax = plt.subplot(3,2,k+1,projection=projection_stereo)
            if k == 6:
                im0=ax.contourf(lon_c, lat, var_c,transform=data_crs,cmap='OrRd',extend='both')
            else:
                clevels = np.arange(-.6,.7,0.1)
                im=ax.contourf(lon_c, lat, var_c,clevels,transform=data_crs,cmap='RdBu_r',extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[k].min() < 0.05: 
                levels = [maps_pval[k].min(),0.05,maps_pval[k].max()]
                ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[k].min() < 0.10:
                levels = [maps_pval[k].min(),0.10,maps_pval[k].max()]
                ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k]) 
            plt.title(self.regressor_names[k],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            if var == 'ua':
                ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
            elif var == 'sst':
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            else: 
                ax.set_extent([-60, 220, -80, -25], ccrs.PlateCarree(central_longitude=180))
            
        plt1_ax = plt.gca()
        left, bottom, width, height = plt1_ax.get_position().bounds
        if var == 'ua':
            colorbar_axes1 = fig_coef.add_axes([left+0.28, bottom, 0.01, height*2])
            colorbar_axes2 = fig_coef.add_axes([left+0.36, bottom, 0.01, height*2])
        elif var == 'sst':
            colorbar_axes1 = fig_coef.add_axes([left+0.3, bottom, 0.01, height*3])
            colorbar_axes2 = fig_coef.add_axes([left+0.38, bottom, 0.01, height*3])
        else:
            colorbar_axes1 = fig_coef.add_axes([left+0.28, bottom, 0.01, height*2])
            colorbar_axes2 = fig_coef.add_axes([left+0.36, bottom, 0.01, height*2])
        cbar = fig_coef.colorbar(im, colorbar_axes1, orientation='vertical')
        cbar2 = fig_coef.colorbar(im, colorbar_axes2, orientation='vertical')
        if var == 'ua':
            cbar.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
        elif var == 'sst':
            cbar.set_label('K/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('K/std(rd)',fontsize=14) #rotation = radianes
        else:
            cbar.set_label('X/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('X/std(rd)',fontsize=14) #rotation = radianes
        cbar.ax.tick_params(axis='both',labelsize=14)
        cbar2.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        if var == 'ua':
            plt.savefig(output_path+'/regression_coefficients_u850',bbox_inches='tight')
        elif  var == 'sst':
            plt.savefig(output_path+'/regression_coefficients_sst',bbox_inches='tight')
        else:
            plt.savefig(output_path+'/regression_coefficients_unknown_var',bbox_inches='tight')
        
        plt.clf

        return fig_coef


def plot_regression_coef_map_MEM(maps, maps_pval, regressor_names, output_path):
    """Plots figure with regression coefficient maps with two distinct colorbars.
    :param regressor_names: list with strings naming the independent variables
    :param path: saving path
    :return: figure
    """

    # Custom colormap
    cmapU850 = mpl.colors.ListedColormap(['darkblue', 'navy', 'steelblue', 'lightblue',
                                'lightsteelblue', 'white', 'white', 'mistyrose',
                                'lightcoral', 'indianred', 'brown', 'firebrick'])
    cmapU850.set_over('maroon')
    cmapU850.set_under('midnightblue')

    # Load data for contours
    ua_era5 = xr.open_dataset('/home/jmindlin/causal_EDJ/ERA5/ua_ERA5.nc')
    ua_era5 = ua_era5.rename({'latitude': 'lat', 'longitude': 'lon'})
    ua_era5_850 = ua_era5.sel(level=850)
    u_ERA = ua_era5_850.u.sel(time=slice('1979', '2018'))
    u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

    # Create figure and subplots with adjusted size
    fig_coef, axs = plt.subplots(2, 3, figsize=(24, 15), dpi=100,
                                subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=300)})
    plt.subplots_adjust(bottom=0.1, right=0.85, top=0.85, hspace=0.1, wspace=0.25)

    data_crs = ccrs.PlateCarree()

    # Loop over the subplots
    for k, ax in enumerate(axs.flat):
        if k >= len(maps):  # Stop if we have more subplots than data
            break
        
        lat = maps[k].lat
        lon = np.linspace(0, 360, len(maps[k].lon))
        var_c, lon_c = add_cyclic_point(maps[k].values, lon)

        ax.set_extent([0, 359.9, -90, 0], crs=data_crs)
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        # Use different color scales for the first and other subplots
        if k == 0:
            clevels = np.arange(-11, 12, 1)
            # Contour plot
            im0 = ax.contourf(maps[k].lon, maps[k].lat, maps[k].values, clevels, transform=data_crs, cmap=cmapU850, extend='both')
        else:
            clevels = np.arange(-1, 1.1, 0.1)
            # Contour plot
            try:
                im = ax.contourf(maps[k].lon, maps[k].lat, maps[k].values, clevels, transform=data_crs, cmap=cmapU850, extend='both')
            except TypeError:
                print(maps[k])

        # Overlay contour lines for u_ERA
        cnt = ax.contour(u_ERA.lon, u_ERA.lat, u_ERA.values, levels=[8], transform=data_crs,
                        linewidths=1.2, colors='black', linestyles='-')
        ax.clabel(cnt, inline=True, fmt='%1.0f', fontsize=8)

        # Check for significant p-values and hatch regions
        if maps_pval[k].min() < 0.05:
            levels = [maps_pval[k].min(), 0.05, maps_pval[k].max()]
            ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values, levels=levels,
                        transform=data_crs, hatches=["...", " "], alpha=0)
        
        # Plot title
        ax.set_title(regressor_names[k], fontsize=18)
        
        # Add coastlines and borders
        ax.add_feature(cartopy.feature.COASTLINE, alpha=.5)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
        ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
        ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())

    # Create two colorbars outside the grid of subplots

    # Colorbar for the first subplot
    cbar_ax_1 = fig_coef.add_axes([0.87, 0.55, 0.02, 0.25])  # Manually specify position
    cbar_1 = fig_coef.colorbar(im0, cax=cbar_ax_1, orientation='vertical', ticks=np.arange(-11, 12, 1))
    cbar_1.set_label(r'm s$^{-1}$ $\sigma_{RD}^{-1}$', fontsize=14)
    cbar_1.ax.tick_params(axis='both', labelsize=12)

    # Add "panel a" text above the first colorbar
    plt.text(0.87, 0.82, 'panel a', fontsize=14, transform=fig_coef.transFigure, ha='center')

    # Colorbar for the remaining subplots
    cbar_ax_2 = fig_coef.add_axes([0.87, 0.18, 0.02, 0.25])  # Manually specify position
    cbar_2 = fig_coef.colorbar(im, cax=cbar_ax_2, orientation='vertical', ticks=np.arange(-1, 1.1, 0.1))
    cbar_2.set_label(r'm s$^{-1}$ $\sigma_{RD}^{-1}$', fontsize=14)
    cbar_2.ax.tick_params(axis='both', labelsize=12)

    # Add "panels b-f" text above the second colorbar
    plt.text(0.87, 0.45, 'panels b-f', fontsize=14, transform=fig_coef.transFigure, ha='center')

    plt.savefig(output_path, bbox_inches='tight')
    plt.clf()

    return fig_coef



def detrend_timeseries(da):
    """
    Remove the linear trend from an xarray DataArray.

    Parameters:
    - da: xarray.DataArray
        The input time series to detrend.

    Returns:
    - detrended_da: xarray.DataArray
        The detrended time series.
    """
    # Check if the input is a DataArray
    if not isinstance(da, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray")

    # Ensure that there is a 'time' coordinate
    if 'time' not in da.coords:
        raise ValueError("DataArray must have a 'time' coordinate")

    # Detrend the data along the time axis
    detrended_data = detrend(da, axis=0)

    # Return as a new DataArray, preserving the original coordinates
    detrended_da = xr.DataArray(detrended_data, dims=da.dims, coords=da.coords)
    
    return detrended_da


import pandas as pd
import statsmodels.api as sm

def multiple_linear_regression(target, predictors_dict):
    """
    Perform a multiple linear regression on a target time series using a dictionary of predictor time series.

    Parameters:
    - target: xarray.DataArray
        The target time series to predict.
    - predictors_dict: dict
        A dictionary where keys are predictor names and values are xarray.DataArray objects representing predictor time series.

    Returns:
    - results: statsmodels.regression.linear_model.RegressionResultsWrapper
        The results of the regression, including coefficients, p-values, etc.
    """
    # Check if input is a DataArray
    if not isinstance(target, xr.DataArray):
        raise TypeError("Target must be an xarray DataArray")
    
    # Ensure that there is a 'time' coordinate
    if 'time' not in target.coords:
        raise ValueError("Target DataArray must have a 'time' coordinate")

    # Convert the target and predictors to a pandas DataFrame
    df = pd.DataFrame({name: da.to_series() for name, da in predictors_dict.items()})
    
    # Ensure the target time series is aligned with predictors
    df['target'] = target.to_series()
    
    # Drop any rows with NaN values
    df.dropna(inplace=True)

    # Separate the predictors and the target
    X = df.drop(columns='target')
    y = df['target']

    # Add a constant (intercept) to the predictors
    X = sm.add_constant(X)

    # Perform the OLS regression
    model = sm.OLS(y, X)
    results = model.fit()

    return results


def standardize_data(da):
    """
    Standardize an xarray DataArray by subtracting the mean and dividing by the standard deviation.
    
    Parameters:
    da (xarray.DataArray or xarray.Dataset): Input data to standardize.
    
    Returns:
    xarray.DataArray or xarray.Dataset: Standardized data.
    """
    mean = da.mean(dim='time')
    std_dev = da.std(dim='time')
    
    standardized_da = (da - mean) / std_dev
    return standardized_da


class JetAnalysis:
    def __init__(self, jet_data, covariates_dict, lon1=140, lon2=295):
        """
        Initialize the class with the jet data and a dictionary of covariates.
        
        Parameters:
        - jet_data: xarray.DataArray
            The jet data with dimensions lat, lon, and time.
        - covariates_dict: dict
            Dictionary of covariates, where each key is a predictor name and each value is an xarray.DataArray.
        - lon1: int
            The starting longitude for the analysis (default is 140).
        - lon2: int
            The ending longitude for the analysis (default is 295).
        """
        self.jet_data = jet_data
        self.covariates_dict = covariates_dict
        self.lon1 = lon1
        self.lon2 = lon2
        self.jet_lat = None
        self.jet_strength = None
        self.regression_results = None
    
    def jet_lat_strength(self):
        """
        Calculate the jet latitude and jet strength from the input jet data.
        
        Returns:
        - jet_lat: np.array
            Array of calculated jet latitudes over time.
        - jet_strength: np.array
            Array of calculated jet strengths over time.
        """
        jet_30_70 = self.jet_data.sel(lat=slice(-70, -30)).sel(lon=slice(self.lon1, self.lon2)).mean(dim='lon')
        lat = jet_30_70.lat
        jet_lat = (jet_30_70 * lat).sum(dim='lat') / jet_30_70.sum(dim='lat')
        
        strength = []
        for t, max_lat in zip(self.jet_data.time, jet_lat):
            strength.append(self.jet_data.sel(time=t).sel(lat=max_lat, method='nearest').sel(lon=slice(self.lon1, self.lon2)).mean(dim='lon'))
        jet_strength = np.array(strength)
        
        # Store the results as class attributes
        self.jet_lat = np.array(jet_lat.values)
        self.jet_strength = jet_strength
        
        return self.jet_lat, self.jet_strength
    
    def standardize_data(self, da):
        """
        Standardize an xarray DataArray by subtracting the mean and dividing by the standard deviation.
        
        Parameters:
        - da: xarray.DataArray
            Input data to standardize.
        
        Returns:
        - standardized_da: xarray.DataArray
            Standardized data.
        """
        mean = da.mean(dim='time')
        std_dev = da.std(dim='time')
        standardized_da = (da - mean) / std_dev
        return standardized_da
    
    def multiple_linear_regression(self, target, predictors_dict):
        """
        Perform multiple linear regression on a target time series using a dictionary of predictors.
        
        Parameters:
        - target: xarray.DataArray
            The target time series to predict.
        - predictors_dict: dict
            A dictionary where keys are predictor names and values are xarray.DataArray objects representing predictor time series.
        
        Returns:
        - results: statsmodels.regression.linear_model.RegressionResultsWrapper
            The results of the regression, including coefficients, p-values, etc.
        """
        # Convert the target and predictors to a pandas DataFrame
        df = pd.DataFrame({name: da.to_series() for name, da in predictors_dict.items()})
        
        # Ensure the target time series is aligned with predictors
        df['target'] = target.to_series()
        
        # Drop any rows with NaN values
        df.dropna(inplace=True)

        # Separate the predictors and the target
        X = df.drop(columns='target')
        y = df['target']

        # Add a constant (intercept) to the predictors
        X = sm.add_constant(X)

        # Perform the OLS regression
        model = sm.OLS(y, X)
        results = model.fit()

        return results

    def analyze(self):
        """
        Perform the full analysis: calculate metrics, standardize data, perform regression, and save results.
        
        Returns:
        - results_dict: dict
            A dictionary containing the regression coefficients, p-values, and summary.
        """
        # Step 1: Calculate the jet latitude and strength
        jet_lat, jet_strength = self.jet_lat_strength()

        # Step 2: Standardize the data
        standardized_jet_lat = self.standardize_data(xr.DataArray(jet_lat, dims=['time'], coords={'time': self.jet_data.time}))
        standardized_jet_strength = self.standardize_data(xr.DataArray(jet_strength, dims=['time'], coords={'time': self.jet_data.time}))

        # Standardize the predictors
        standardized_predictors = {name: self.standardize_data(da) for name, da in self.covariates_dict.items()}

        # Step 3: Perform multiple linear regression on both metrics
        lat_results = self.multiple_linear_regression(standardized_jet_lat, standardized_predictors)
        strength_results = self.multiple_linear_regression(standardized_jet_strength, standardized_predictors)

        # Save the regression results
        self.regression_results = {
            'jet_lat_regression': lat_results,
            'jet_strength_regression': strength_results
        }

        # Step 4: Compile the results into a dictionary
        results_dict = {
            'jet_lat_coefficients': lat_results.params.to_dict(),
            'jet_lat_pvalues': lat_results.pvalues.to_dict(),
            'jet_lat_summary': lat_results.summary().as_text(),
            'jet_strength_coefficients': strength_results.params.to_dict(),
            'jet_strength_pvalues': strength_results.pvalues.to_dict(),
            'jet_strength_summary': strength_results.summary().as_text()
        }

        return results_dict

def std_anom(x):
    return (x - np.mean(x))/np.std(x)


def calculate_trend(data_array):
    """
    Calculate the linear trend along the 'time' dimension of an xarray.DataArray.
    
    Parameters:
        data_array (xarray.DataArray): Input data with dimensions ('time', 'lat', 'lon').
        
    Returns:
        xarray.DataArray: An array with dimensions ('lat', 'lon') containing the linear trend
                          (slope) for each grid cell.
    """
    # Ensure the data has the correct dimensions
    if not {'time', 'lat', 'lon'}.issubset(data_array.dims):
        raise ValueError("The input data must have 'time', 'lat', and 'lon' dimensions.")
    
    # Prepare an empty array to store the trends
    lat_size = data_array.sizes['lat']
    lon_size = data_array.sizes['lon']
    trends = np.zeros((lat_size, lon_size))
    
    # Loop through each grid cell
    for i in range(lat_size):
        for j in range(lon_size):
            # Extract the time series for this grid cell
            time_series = data_array[:, i, j].values
            
            # Calculate the linear trend (slope) using scipy.stats.linregress
            if np.all(np.isnan(time_series)):
                trends[i, j] = np.nan  # Handle NaNs in the time series
            else:
                time = np.arange(len(time_series))
                slope, _, _, _, _ = linregress(time, time_series)
                trends[i, j] = slope
    
    # Create an output xarray.DataArray with the same lat and lon as the input
    trend_array = xr.DataArray(
        trends,
        coords={"lat": data_array.lat, "lon": data_array.lon},
        dims=["lat", "lon"],
        name="linear_trend"
    )
    
    return trend_array

def calculate_trend_and_pvalues(data_array):
    """
    Calculate the linear trend and p-values along the 'time' dimension of an xarray.DataArray.
    
    Parameters:
        data_array (xarray.DataArray): Input data with dimensions ('time', 'lat', 'lon').
        
    Returns:
        tuple: Two xarray.DataArrays containing the linear trend (slopes) and p-values for
               each grid cell, respectively.
    """
    # Ensure the data has the correct dimensions
    if not {'time', 'lat', 'lon'}.issubset(data_array.dims):
        raise ValueError("The input data must have 'time', 'lat', and 'lon' dimensions.")
    
    # Prepare empty arrays to store the trends and p-values
    lat_size = data_array.sizes['lat']
    lon_size = data_array.sizes['lon']
    trends = np.zeros((lat_size, lon_size))
    pvalues = np.ones((lat_size, lon_size))  # Initialize p-values to 1 (non-significant)
    
    # Loop through each grid cell
    for i in range(lat_size):
        for j in range(lon_size):
            # Extract the time series for this grid cell (corrected indexing)
            time_series = data_array.isel(lat=i, lon=j).values
            
            # Calculate the linear trend (slope) and p-value using scipy.stats.linregress
            if np.all(np.isnan(time_series)):
                trends[i, j] = np.nan  # Handle NaNs in the time series
                pvalues[i, j] = np.nan
            else:
                time = np.arange(len(time_series))
                slope, _, _, p_value, _ = linregress(time, time_series)
                trends[i, j] = slope
                pvalues[i, j] = p_value
    
    # Create output xarray.DataArrays with the same lat and lon as the input
    trend_array = xr.DataArray(
        trends,
        coords={"lat": data_array.lat, "lon": data_array.lon},
        dims=["lat", "lon"],
        name="linear_trend"
    )
    
    pvalue_array = xr.DataArray(
        pvalues,
        coords={"lat": data_array.lat, "lon": data_array.lon},
        dims=["lat", "lon"],
        name="p_value"
    )
    
    return trend_array, pvalue_array


# Define the linear regression function
def linear_regression(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept

