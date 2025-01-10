#I AM HUMAN
# I AM ROBOT
# I AM GAIA
import xarray as xr
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
from sklearn import linear_model
import glob
from scipy import signal
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib as mpl
import random

#Across models regression class
class spatial_MLR(object):
    def __init__(self):
        self.what_is_this = 'This performs a regression across models and plots everything'
    
    def regression_data(self,variable,regressors,regressor_names):
        """Define the regression target variable 
        this is here to be edited if some opperation is needed on the DataArray
        
        :param variable: DataArray
        :return: target variable for the regression  
        """
        self.target = variable
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
    
    def perform_regression(self,path): 
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
                regression_coefs = results[0].to_dataset()
            else:
                regression_coefs[self.regressor_names[i]] = results[i]
                
        regression_coefs = regression_coefs.rename({'ua':self.regressor_names[0]})
        #regression_coefs.to_netcdf(path+'/regression_coefficients.nc')
        save_xarray_to_netcdf(regression_coefs, path+'/regression_coefficients.nc')
        
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs_pvalues = results_pvalues[0].to_dataset()
            else:
                regression_coefs_pvalues[self.regressor_names[i]] = results_pvalues[i]
                
        regression_coefs_pvalues = regression_coefs_pvalues.rename({'ua':self.regressor_names[0]})
        save_xarray_to_netcdf(regression_coefs_pvalues, path+'/regression_coefficients_pvalues.nc')
        #regression_coefs_pvalues.to_netcdf(path+'/regression_coefficients_pvalues.nc')
        
        #results_R2.to_netcdf(path+'/R2.nc')
        save_xarray_to_netcdf(results_R2, path+'/R2.nc')

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
     
    def open_regression_coef(self,path):
        """ Open regression coefficients and pvalues to plot

        :param path: saving path
        :return maps: list of list of coefficient maps
        :return maps_pval:  list of coefficient pvalues maps
        :return R2: map of fraction of variance
        """ 
        maps = []; maps_pval = []
        print(path+'/regression_coefficients.nc')
        coef_maps = xr.open_dataset(path+'/regression_coefficients.nc')
        coef_pvalues = xr.open_dataset(path+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/R2.nc')
        return maps, maps_pval, R2    

    def open_lmg_coef(self,path):
        """ Open regression coefficients and pvalues to plot

        :param path: saving path
        :return maps: list of list of coefficient maps
        :return maps_pval:  list of coefficient pvalues maps
        :return R2: map of fraction of variance
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/regression_coefficients_relative_importance.nc')
        coef_pvalues = xr.open_dataset(path+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names[1:]]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/R2.nc')
        return maps, maps_pval, R2    
    
    def plot_regression_lmg_map(self,path,var,alias,output_path):
        """ Plots figure with all of 

        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_lmg_coef(path+'/'+var+'/'+alias)
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
        projection = ccrs.SouthPolarStereo(central_longitude=300)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num-1):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic
            ax = plt.subplot(3,3,k+1,projection=projection)
            ax.set_extent([0,359.9, -90, 0], crs=data_crs)
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
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
            colorbar_axes1 = fig_coef.add_axes([left+0.23, bottom, 0.01, height*0.6])
            cbar = fig_coef.colorbar(im, colorbar_axes1, orientation='vertical')
            cbar.set_label('relative importance',fontsize=14) #rotation = radianes
            cbar.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        plt.savefig(output_path+'/regression_coefficients_relative_importance_u850_'+alias,bbox_inches='tight')
        plt.clf

        return fig_coef

    def plot_regression_coef_map(self, path, alias, subplot_names, output_path):
        """Plots figure with regression coefficient maps with two distinct colorbars.
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: figure
        """
        maps, maps_pval, R2 = self.open_regression_coef(path+'/'+alias)

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
        fig_coef, axs = plt.subplots(2, 3, figsize=(24, 18), dpi=100,
                                    subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=300)})
        plt.subplots_adjust(bottom=0.1, right=0.85, top=0.85, hspace=0.1, wspace=0.25)

        data_crs = ccrs.PlateCarree()

        # Loop over the subplots
        for k, ax in enumerate(axs.flat):
            if k >= self.rd_num:  # Stop if we have more subplots than data
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
                im0 = ax.contourf(lon_c, lat, var_c, clevels, transform=data_crs, cmap=cmapU850, extend='both')
                #im0 = ax.contourf(maps[k].lon, maps[k].lat, maps[k].values, clevels, transform=data_crs, cmap=cmapU850, extend='both')
            else:
                clevels = np.arange(-0.6, 0.7, 0.1)
                # Contour plot
                try:
                    im = ax.contourf(lon_c, lat, var_c, clevels, transform=data_crs, cmap=cmapU850, extend='both')
                #    im = ax.contourf(maps[k].lon, maps[k].lat, maps[k].values, clevels, transform=data_crs, cmap=cmapU850, extend='both')
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
            ax.set_title(subplot_names[k], fontsize=18)
            
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
        cbar_2 = fig_coef.colorbar(im, cax=cbar_ax_2, orientation='vertical', ticks=np.arange(-0.6, 0.7, 0.2))
        cbar_2.set_label(r'm s$^{-1}$ $\sigma_{RD}^{-1}$', fontsize=14)
        cbar_2.ax.tick_params(axis='both', labelsize=12)

        # Add "panels b-f" text above the second colorbar
        plt.text(0.87, 0.45, 'panels b-f', fontsize=14, transform=fig_coef.transFigure, ha='center')
    
        try:
            plt.savefig(output_path+'/regression_coefficients_u850_'+alias, bbox_inches='tight')
        except TypeError:
            print(alias)
        plt.clf()

        return fig_coef


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


def multiple_linear_regression(target,predictors):
    """Multiple linear regression to estimate the links in  Causal Effect Network
    target: np.array - time series of target variable
    predictors: pandas dataframe with predictor variables
    
    """
    y = predictors.apply(stand_detr_filtro,axis=0).values
    print(predictors.keys())
    res = sm.OLS(stand_detr(target),y).fit()
    coef_output = {var:round(res.params[i],10) for var,i in zip(predictors.keys(),range(len(predictors.keys())))}
    coef_pvalues = {var:round(res.pvalues[i],10) for var,i in zip(predictors.keys(),range(len(predictors.keys())))}
    return coef_output, coef_pvalues
    
def figure(target,predictors):
    fig = plt.figure()
    y = predictors.apply(stand_detr,axis=0).values
    for i in range(len(predictors.keys())):
        plt.plot(y[:,i])
    plt.plot(stand_detr(target))
    return fig

def iod(iod_e,iod_w):
    iod_index = iod_w - iod_e
    return iod_index

def jet_lat_strength(jet_data,lon1=140,lon2=295):
    jet_30_70 = jet_data.sel(lat=slice(-70,-30)).sel(lon=slice(lon1,lon2)).mean(dim='lon')
    lat = jet_30_70.lat
    jet_lat = (jet_30_70*lat).sum(dim='lat')/(jet_30_70).sum(dim='lat')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(lat=max_lat,method='nearest').sel(lon=slice(lon1,lon2)).mean(dim='lon'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength

def save_xarray_to_netcdf(data_array, file_path):
    """
    Saves an xarray DataArray or Dataset to a NetCDF file if it does not exist.

    Parameters:
    - data_array: xarray.DataArray or xarray.Dataset
        The data to save to a NetCDF file.
    - file_path: str
        The path where the NetCDF file will be saved.

    Returns:
    - None
    """
    # Check if the file already exists
    if not os.path.exists(file_path):
        # Save the DataArray or Dataset to the specified file path
        data_array.to_netcdf(file_path)
        print(f"File saved to {file_path}.")
    else:
        # Skip saving if the file already exists
        print(f"File {file_path} already exists. Skipping save.")


def return_dict_time_series_with_residuals(df):
    """
    Returns a dictionary with residuals and a dictionary with smoothed time series
    
    :param df: DataFrame with multiple time series (each column is a time series)
    """
    n = df.shape[1]  # Number of time series
    fig, axs = plt.subplots(n, 2, figsize=(15, 4*n), dpi=100)

    ts_dict_residuals = {}
    ts_dict_smoothed = {}
    for i, column in enumerate(df.columns):
        # Extract the time series
        time_series = df[column]
        
        # Calculate the smoothed time series (5-step running mean)
        smoothed_series = time_series.rolling(window=5, center=True).mean()
        ts_dict_smoothed[column] = smoothed_series
        # Calculate residuals (original - smoothed)
        residuals = time_series - smoothed_series
        ts_dict_residuals[column] = residuals

    return ts_dict_residuals, ts_dict_smoothed


def main(config):
    """Run the diagnostic.""" 
    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    print(cfg)
    meta = group_metadata(config["input_data"].values(), "alias")
    #print(f"\n\n\n{meta}")
    for alias, alias_list in meta.items():
        if (alias != 'MIROC-ES2L') & (alias!= 'NESM3'):
            # Perform multiple linear regression with raw time series and without removing the zonal mean from the Pacific indices
            t0_dict = {m["variable_group"]: xr.open_dataset(m["filename"]).time.dt.year[0].values for m in alias_list}
            t0_dict_time = {m["variable_group"]: xr.open_dataset(m["filename"]).time for m in alias_list}
            variables = list(t0_dict.keys())
            print(t0_dict_time[variables[0]])
            start_year_list = np.array([t0_dict[var] for var in t0_dict.keys()])
            start_year = np.max(start_year_list)
            if start_year < 802:
                start_str = '0'+str(start_year+1)
                end_str = '0'+str(start_year+1+198)
            elif start_year < 1000 and start_year > 802:
                start_str = '0'+str(start_year+1)
                end_str = str(start_year+1+198)
            else:
                start_str = str(start_year+1)
                end_str = str(start_year+1+198)
            
            if start_year < 802:
                start_str_spv = '0'+str(start_year)
                end_str_spv = '0'+str(start_year+198)
            elif start_year < 1000 and start_year > 802:
                start_str_spv = '0'+str(start_year)
                end_str_spv = str(start_year+198)
            else:
                start_str_spv = str(start_year)
                end_str_spv = str(start_year+198)
            
            try:
                ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).values for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tas") & (m["variable_group"] != "pr") & (m["variable_group"] != "sst") & (m["variable_group"] != "psl") & (m["variable_group"] != "ta250") & (m["variable_group"] != 'ua50_spv') & (m["variable_group"] != 'gw')}
                ts_dict_spv = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str_spv,end_str_spv)).values for m in alias_list if (m["variable_group"] == 'ua50_spv')}
                ts_dict_tos_cp = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-5,5)).sel(lon=slice(180,250)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                ts_dict_tos_ep = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(0,10)).sel(lon=slice(260,280)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                ts_dict_tos_zm = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-10,10)).sel(lon=slice(0,360)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                ts_dict['ua50_spv'] = ts_dict_spv['ua50_spv']
                ts_dict['tos_cp'] = ts_dict_tos_cp['sst'] - ts_dict_tos_zm['sst']
                ts_dict['tos_ep'] = ts_dict_tos_ep['sst'] - ts_dict_tos_zm['sst']
                target_wind = [xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)) for m in alias_list if m["variable_group"] == "ua850"]
                target_wind = target_wind[0] if len(target_wind) == 1 else 0
            except ValueError:
                try:
                    start_str = str(start_year+1)
                    end_str = str(start_year+1+198)
                    start_str_spv = str(start_year)
                    end_str_spv = str(start_year+198)
                    ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).values for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tas") & (m["variable_group"] != "pr") & (m["variable_group"] != "sst") & (m["variable_group"] != "psl") & (m["variable_group"] != "ta250") & (m["variable_group"] != 'ua50_spv') & (m["variable_group"] != 'gw')}
                    ts_dict_spv = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str_spv,end_str_spv)).values for m in alias_list if (m["variable_group"] == 'ua50_spv')}
                    ts_dict_tos_cp = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-5,5)).sel(lon=slice(180,250)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict_tos_ep = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(0,10)).sel(lon=slice(260,280)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict_tos_zm = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-10,10)).sel(lon=slice(0,360)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict['ua50_spv'] = ts_dict_spv['ua50_spv']
                    ts_dict['tos_cp'] = ts_dict_tos_cp['sst'] - ts_dict_tos_zm['sst']
                    ts_dict['tos_ep'] = ts_dict_tos_ep['sst'] - ts_dict_tos_zm['sst']
                    target_wind = [xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)) for m in alias_list if m["variable_group"] == "ua850"]
                    target_wind = target_wind[0] if len(target_wind) == 1 else 0    
                except ValueError:
                    print('ENTRE ACA')
                    start_str = '000'+str(start_year+1)
                    end_str = '0'+str(start_year+1+198)   
                    start_str_spv = '000'+str(start_year)
                    end_str_spv = '0'+str(start_year+198)  
                    ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).values for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tas") & (m["variable_group"] != "pr") & (m["variable_group"] != "sst") & (m["variable_group"] != "psl") & (m["variable_group"] != "ta250") & (m["variable_group"] != 'ua50_spv') & (m["variable_group"] != 'gw')}
                    ts_dict_spv = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str_spv,end_str_spv)).values for m in alias_list if (m["variable_group"] == 'ua50_spv')}
                    ts_dict_tos_cp = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-5,5)).sel(lon=slice(180,250)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict_tos_ep = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(0,10)).sel(lon=slice(260,280)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict_tos_zm = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-10,10)).sel(lon=slice(0,360)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict['ua50_spv'] = ts_dict_spv['ua50_spv']
                    ts_dict['tos_cp'] = ts_dict_tos_cp['sst'] - ts_dict_tos_zm['sst']
                    ts_dict['tos_ep'] = ts_dict_tos_ep['sst'] - ts_dict_tos_zm['sst']
                    target_wind = [xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)) for m in alias_list if m["variable_group"] == "ua850"]
                    target_wind = target_wind[0] if len(target_wind) == 1 else 0    

            print(f"Computing spatial regression for {alias}\n")
            for key, value in ts_dict.items():
                print(f"Shape of value for {key}: {len(value)}")
            # Find the minimum length of all arrays in the dictionary
            lengths = [len(value) for value in ts_dict.values()]
            lengths.append(len(target_wind.time))
            min_length = min(lengths)

            # Truncate each array to the minimum length
            for key in ts_dict:
                ts_dict[key] = ts_dict[key][:min_length]

            ts_dict = pd.DataFrame(ts_dict)
            # List of column names in order
            new_column_order = ['ua50_spv', 'ta', 'tos_cp', 'tos_ep']
            # Reorder the DataFrame columns
            ts_dict_order = ts_dict[new_column_order]

            target_wind = target_wind[:min_length]
            MLR = spatial_MLR()
            MLR.regression_data(target_wind,ts_dict_order.apply(stand,axis=0),ts_dict_order.keys().insert(0,'clim'))
            os.chdir(config["work_dir"])
            os.getcwd()
            os.makedirs("u850_RawDrivers_withZMremove_withoutGW",exist_ok=True)
            os.chdir(config["work_dir"]+'/'+"u850_RawDrivers_withZMremove_withoutGW")
            os.makedirs(alias,exist_ok=True)
            #MLR.perform_regression(config["work_dir"]+'/u850_RawDrivers_withZMremove_withoutGW/'+alias)
            #Plot coefficients
            os.chdir(config["plot_dir"])
            os.getcwd()
            os.makedirs("u850_RawDrivers_withZMremove_withoutGW",exist_ok=True)
            subplot_names = ['Climatology','Stratospheric \n Polar Vortex', 'Tropical \n Warming', 'Central Pacific \n aSST', 'Easter Pacific \n aSST']

            MLR.plot_regression_coef_map(config["work_dir"]+'/u850_RawDrivers_withZMremove_withoutGW',alias,subplot_names,config["plot_dir"]+'/u850_RawDrivers_withZMremove_withoutGW')
                
    for alias, alias_list in meta.items():
        if (alias != 'MIROC-ES2L') & (alias!= 'NESM3'):
            # Perform multiple linear regression with raw time series and without removing the zonal mean from the Pacific indices
            t0_dict = {m["variable_group"]: xr.open_dataset(m["filename"]).time.dt.year[0].values for m in alias_list}
            t0_dict_time = {m["variable_group"]: xr.open_dataset(m["filename"]).time for m in alias_list}
            variables = list(t0_dict.keys())
            print(t0_dict_time[variables[0]])
            start_year_list = np.array([t0_dict[var] for var in t0_dict.keys()])
            start_year = np.max(start_year_list)
            if start_year < 802:
                start_str = '0'+str(start_year+1)
                end_str = '0'+str(start_year+1+198)
            elif start_year < 1000 and start_year > 802:
                start_str = '0'+str(start_year+1)
                end_str = str(start_year+1+198)
            else:
                start_str = str(start_year+1)
                end_str = str(start_year+1+198)
            
            if start_year < 802:
                start_str_spv = '0'+str(start_year)
                end_str_spv = '0'+str(start_year+198)
            elif start_year < 1000 and start_year > 802:
                start_str_spv = '0'+str(start_year)
                end_str_spv = str(start_year+198)
            else:
                start_str_spv = str(start_year)
                end_str_spv = str(start_year+198)
            
            try:
                ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).values for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tas") & (m["variable_group"] != "pr") & (m["variable_group"] != "sst") & (m["variable_group"] != "psl") & (m["variable_group"] != "ta250") & (m["variable_group"] != 'ua50_spv') & (m["variable_group"] != 'gw')}
                ts_dict_spv = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str_spv,end_str_spv)).values for m in alias_list if (m["variable_group"] == 'ua50_spv')}
                ts_dict_tos_cp = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-5,5)).sel(lon=slice(180,250)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                ts_dict_tos_ep = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(0,10)).sel(lon=slice(260,280)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                ts_dict_tos_zm = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-10,10)).sel(lon=slice(0,360)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                ts_dict['ua50_spv'] = ts_dict_spv['ua50_spv']
                ts_dict['tos_cp'] = ts_dict_tos_cp['sst'] #- ts_dict_tos_zm['sst']
                ts_dict['tos_ep'] = ts_dict_tos_ep['sst'] #- ts_dict_tos_zm['sst']
                target_wind = [xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)) for m in alias_list if m["variable_group"] == "ua850"]
                target_wind = target_wind[0] if len(target_wind) == 1 else 0
            except ValueError:
                try:
                    start_str = str(start_year+1)
                    end_str = str(start_year+1+198)
                    start_str_spv = str(start_year)
                    end_str_spv = str(start_year+198)
                    ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).values for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tas") & (m["variable_group"] != "pr") & (m["variable_group"] != "sst") & (m["variable_group"] != "psl") & (m["variable_group"] != "ta250") & (m["variable_group"] != 'ua50_spv') & (m["variable_group"] != 'gw')}
                    ts_dict_spv = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str_spv,end_str_spv)).values for m in alias_list if (m["variable_group"] == 'ua50_spv')}
                    ts_dict_tos_cp = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-5,5)).sel(lon=slice(180,250)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict_tos_ep = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(0,10)).sel(lon=slice(260,280)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict_tos_zm = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-10,10)).sel(lon=slice(0,360)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict['ua50_spv'] = ts_dict_spv['ua50_spv']
                    ts_dict['tos_cp'] = ts_dict_tos_cp['sst'] #- ts_dict_tos_zm['sst']
                    ts_dict['tos_ep'] = ts_dict_tos_ep['sst'] #- ts_dict_tos_zm['sst']
                    target_wind = [xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)) for m in alias_list if m["variable_group"] == "ua850"]
                    target_wind = target_wind[0] if len(target_wind) == 1 else 0    
                except ValueError:
                    print('ENTRE ACA')
                    start_str = '000'+str(start_year+1)
                    end_str = '0'+str(start_year+1+198)   
                    start_str_spv = '000'+str(start_year)
                    end_str_spv = '0'+str(start_year+198)  
                    ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).values for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tas") & (m["variable_group"] != "pr") & (m["variable_group"] != "sst") & (m["variable_group"] != "psl") & (m["variable_group"] != "ta250") & (m["variable_group"] != 'ua50_spv') & (m["variable_group"] != 'gw')}
                    ts_dict_spv = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str_spv,end_str_spv)).values for m in alias_list if (m["variable_group"] == 'ua50_spv')}
                    ts_dict_tos_cp = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-5,5)).sel(lon=slice(180,250)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict_tos_ep = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(0,10)).sel(lon=slice(260,280)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict_tos_zm = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)).sel(lat=slice(-10,10)).sel(lon=slice(0,360)).mean(dim='lat').mean(dim='lon').values for m in alias_list if (m["variable_group"] == "sst")}
                    ts_dict['ua50_spv'] = ts_dict_spv['ua50_spv']
                    ts_dict['tos_cp'] = ts_dict_tos_cp['sst'] #- ts_dict_tos_zm['sst']
                    ts_dict['tos_ep'] = ts_dict_tos_ep['sst'] #- ts_dict_tos_zm['sst']
                    target_wind = [xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice(start_str,end_str)) for m in alias_list if m["variable_group"] == "ua850"]
                    target_wind = target_wind[0] if len(target_wind) == 1 else 0    

            print(f"Computing spatial regression for {alias}\n")
            for key, value in ts_dict.items():
                print(f"Shape of value for {key}: {len(value)}")
            # Find the minimum length of all arrays in the dictionary
            lengths = [len(value) for value in ts_dict.values()]
            lengths.append(len(target_wind.time))
            min_length = min(lengths)

            # Truncate each array to the minimum length
            for key in ts_dict:
                ts_dict[key] = ts_dict[key][:min_length]

            ts_dict = pd.DataFrame(ts_dict)
            # List of column names in order
            new_column_order = ['ua50_spv', 'ta', 'tos_cp', 'tos_ep']
            # Reorder the DataFrame columns
            ts_dict_order = ts_dict[new_column_order]

            target_wind = target_wind[:min_length]
            MLR = spatial_MLR()
            MLR.regression_data(target_wind,ts_dict_order.apply(stand,axis=0),ts_dict_order.keys().insert(0,'clim'))
            os.chdir(config["work_dir"])
            os.getcwd()
            os.makedirs("u850_RawDrivers_withoutZMremove_withoutGW",exist_ok=True)
            os.chdir(config["work_dir"]+'/'+"u850_RawDrivers_withoutZMremove_withoutGW")
            os.makedirs(alias,exist_ok=True)
            MLR.perform_regression(config["work_dir"]+'/u850_RawDrivers_withoutZMremove_withoutGW/'+alias)
            #Plot coefficients
            os.chdir(config["plot_dir"])
            os.getcwd()
            os.makedirs("u850_RawDrivers_withoutZMremove_withoutGW",exist_ok=True)
            subplot_names = ['Climatology','Stratospheric \n Polar Vortex', 'Tropical \n Warming', 'Central Pacific \n aSST', 'Easter Pacific \n aSST']
            MLR.plot_regression_coef_map(config["work_dir"]+'/u850_RawDrivers_withoutZMremove_withoutGW',alias,subplot_names,config["plot_dir"]+'/u850_RawDrivers_withoutZMremove_withoutGW')


if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                                    
