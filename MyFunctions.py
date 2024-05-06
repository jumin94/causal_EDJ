import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import os, fnmatch
import statsmodels.api as sm
import itertools
from scipy import linalg
import matplotlib as mpl
from mpl_toolkits import mplot3d
from sklearn import mixture

def cargo_datos(ruta,var):
    os.chdir(ruta)
    os.getcwd()
    dic = {}
    listOfFiles = os.listdir(ruta)
    models = [filename.split('_')[1] for filename in listOfFiles if fnmatch.fnmatch(filename,'*.nc')]
    models = set(models)
    for model in models:
        dic[model] = {}
        dic[model]['Members'] = []
        dic[model]['MemberNum'] = []
        pattern = "*"+model+"*.nc"
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry,pattern):
                dato = xr.open_dataset(ruta+'/'+entry)[var]
                dic[model]['Members'].append(dato)
                dic[model]['MemberNum'].append(entry.split('_')[-3])
                
        ensemble = xr.concat(dic[model]['Members'],dim='member')
        dic[model]['EnsembleMean'] = ensemble.mean(dim="member")
            
    return dic

def detrend(serie):
    years = np.arange(0,len(serie),1)
    y = regressor_GW(years)
    coef = linear_regression(y,serie) 
    detrended_serie = serie - (coef[0] + coef[1]*years)
    return detrended_serie

def sel_box(dato,b):
    return dato.sel(lat=slice(b[0],b[1]),lon=slice(b[2],b[3])).mean(dim=('lat','lon'))

def hemisphere_boxes(dic,b):
    dic_out = {}
    models = dic.keys()
    for model in models:
        dic_out[model] = {}
        dic_out[model]['Members'] = [sel_box(m,b) for m in dic[model]['Members']]
        dic_out[model]['MemberNum'] = dic[model]['MemberNum']
        dic_out[model]['EnsembleMean'] = sel_box(dic[model]['EnsembleMean'],b)
    return dic_out

def signal_noise(datos):
    dic = {}
    for model in datos.keys():
        dic[model] = {}
        noise_1900_1950 = []
        for i in range(len(datos[model]['Members'])):
            noise_1900_1950.append(datos[model]['Members'][i].sel(time=slice('1900','1950')).values)
        
        dic[model]['std'] = np.std(np.array(np.concatenate(noise_1900_1950)))
        dic[model]['mean'] = np.mean(np.array(np.concatenate(noise_1900_1950)))
    
    return dic

def jet_lat_strength(jet_data,lon1=140,lon2=295):
    jet_30_70 = jet_data.sel(lat=slice(-70,-30)).sel(lon=slice(lon1,lon2)).mean(dim='lon')**2
    lat = jet_30_70.lat
    jet_lat = (jet_30_70*lat).sum(dim='lat')/(jet_30_70).sum(dim='lat')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(lat=max_lat,method='nearest').sel(lon=slice(lon1,lon2)).mean(dim='lon'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength

def signal_noise_dato_completo(datos):
    dic = {}
    for model in datos.keys():
        dic[model] = {}
        noise_1900_1950 = []
        for i in range(len(datos[model]['Members'])):
            noise_1900_1950.append(datos[model]['Members'][i].sel(time=slice('1900','1950')))
            
        concat = xr.concat(noise_1900_1950,dim='members')
        
        dic[model]['std'] = concat.std(dim='time').mean('members')
        dic[model]['mean'] = concat.mean(dim='time').mean('members')
    
    return dic

def seasonal_data2(data,season):
    # select DJF
    DA_DJF = data.sel(time = data.time.dt.season=="DJF")

    # calculate mean per year
    DA_DJF = DA_DJF.groupby(DA_DJF.time.dt.year).mean("time")
    return DA_DJF
        
        
def filtro(dato):
    """Apply a rolling mean of 5 years and remov the NaNs resulting bigining and end"""
    signal = dato - dato.rolling(time=5, center=True).mean()
    signal_out = signal.dropna('time', how='all')
    return signal_out

def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def clim_anom(dato):
    return dato - dato.isel(time=slice(0,29)).mean(dim='time')

def clim_anom_std(dato):
    return (dato - dato.isel(time=slice(0,29)).mean(dim='time'))/dato.isel(time=slice(0,29)).std(dim='time')

def time_series_plot(dic,title):
    multimodel_ensemble = [filtro(clim_anom(dic[model]['EnsembleMean'])) for model in dic.keys() if model != "GISS-E2-1-G"]
    gw_multimodel_ensemble = xr.concat(multimodel_ensemble,dim='model')
    gw_multimodel_ensemble_mean = gw_multimodel_ensemble.mean(dim='model')
    gw_multimodel_ensemble_std = gw_multimodel_ensemble.std(dim='model')
    for model in gw_index.keys():
        if model != "GISS-E2-1-G":
            plt.plot(filtro(clim_anom(dic[model]['EnsembleMean'])),color='grey') #time[5:-5]
        else:
            continue
    plt.plot(clim_anom(gw_multimodel_ensemble_mean-gw_multimodel_ensemble_std),color='blue')
    plt.plot(clim_anom(gw_multimodel_ensemble_mean+gw_multimodel_ensemble_std),color='red')
    plt.plot(clim_anom(gw_multimodel_ensemble_mean),color='black')
    plt.xlabel('year')

def cargo_regression_coefs(ruta,var):
    os.chdir(ruta)
    os.getcwd()
    dic = {}
    listOfFolders = os.listdir(ruta)
    models = [filename.split('_')[0] for filename in listOfFolders]
    models = set(models)
    print(models)
    for model in models:
        dic[model] = {}
        dic[model]['Members'] = []
        dic[model]['MemberNum'] = []
        pattern = model+"*"
        for entry in listOfFolders:
            if fnmatch.fnmatch(entry,pattern):
                dato = xr.open_dataset(ruta+'/'+entry+'/regression_coefficients.nc')[var]
                dic[model]['Members'].append(dato)
                try:
                    dic[model]['MemberNum'].append(entry.split('_')[1])
                except:
                    dic[model]['MemberNum'].append('r1i1p1f1')
                
        ensemble = xr.concat(dic[model]['Members'],dim='member')
        dic[model]['EnsembleMean'] = ensemble.mean(dim="member")
        os.chdir('/'.join(ruta.split("/")[:-1]))
        os.getcwd()
        os.makedirs("u850_mean_values_time_regression",exist_ok=True)
        ensemble.to_netcdf(ruta+'/'+entry+'/'+var+'_'+model+'_regression_coefficients.nc')
    return dic

def jet_lat_strength_obs(jet_data,lon1=140,lon2=295):
    jet_30_70 = jet_data.sel(latitude=slice(-30,-70)).sel(longitude=slice(lon1,lon2)).mean(dim='longitude')**2
    lat = jet_30_70.latitude
    jet_lat = (jet_30_70*lat).sum(dim='latitude')/(jet_30_70).sum(dim='latitude')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(latitude=max_lat,method='nearest').sel(longitude=slice(lon1,lon2)).mean(dim='longitude'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength

def cargo_datos_rd(ruta,var):
    os.chdir(ruta)
    os.getcwd()
    dic = {}
    listOfFiles = os.listdir(ruta)
    models = [filename.split('_')[1] for filename in listOfFiles if fnmatch.fnmatch(filename,'*.nc')]
    models = set(models)
    for model in models:
        dic[model] = {}
        dic[model]['Members'] = []
        dic[model]['MemberNum'] = []
        pattern = "*"+model+"*.nc"
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry,pattern):
                dato = xr.open_dataset(ruta+'/'+entry)[var]
                dic[model]['Members'].append(dato)
                dic[model]['MemberNum'].append(entry.split('_')[-3])
                
        ensemble = xr.concat(dic[model]['Members'],dim='member')
        dic[model]['EnsembleMean'] = ensemble.mean(dim="member")
            
    return dic

def jet_lat_strength(jet_data,lon1=140,lon2=295):
    jet_30_70 = jet_data.sel(lat=slice(-70,-30)).sel(lon=slice(lon1,lon2)).mean(dim='lon')**2
    lat = jet_30_70.lat
    jet_lat = (jet_30_70*lat).sum(dim='lat')/(jet_30_70).sum(dim='lat')
    strength = []
    for t,max_lat in zip(jet_data.time,jet_lat):
        strength.append(jet_data.sel(time=t).sel(lat=max_lat,method='nearest').sel(lon=slice(lon1,lon2)).mean(dim='lon'))
    jet_strength = np.array(strength)
    return np.array(jet_lat.values),jet_strength


def regressor_EESC_GW(gw_ts,eesc_ts):
    for i in range(len(eesc_ts[:8])):
        eesc_ts['EESC_polar'][i] = eesc_ts['EESC_polar'][8]
    df = pd.DataFrame({'EESC':(eesc_ts['EESC_polar'][:-1] - eesc_ts['EESC_polar'][8]),'GW':gw_ts})
    regressors_out = sm.add_constant(df.values)
    return regressors_out

def regressor_GW(gw_ts):
    df = pd.DataFrame({'GW':gw_ts})
    regressors_out = sm.add_constant(df.values)
    return regressors_out

def linear_regression(y,x):
    res = sm.OLS(x,y).fit()
    returns = [res.params[i] for i in range(len(res.params))]
    return tuple(returns)

def linear_regression_pvalues(y,x):
    res = sm.OLS(x,y).fit()
    returns = [res.pvalues[i] for i in range(len(res.params))]
    return tuple(returns)

def stand(x):
    return (x - np.mean(x)) / np.std(x)

def regression_driver_gw(dic,gw_index):
    dic_out = {}
    for model in dic.keys():
        dic_out[model] = {}
        gw_ts = gw_index[model]['EnsembleMean'].values  -  np.mean(gw_index[model]['EnsembleMean'].values[:30])
        y = regressor_GW(gw_ts)
        coef = linear_regression(y,dic[model]['EnsembleMean'].values)   
        pval = linear_regression_pvalues(y,dic[model]['EnsembleMean'].values)
        dic_out[model]['coef'] = coef
        dic_out[model]['pval'] = pval  
    
    return dic_out

def regression_driver_eesc_gw(dic,gw_index,eesc_ts):
    dic_out = {}
    for model in dic.keys():
        dic_out[model] = {}
        gw_ts = gw_index[model]['EnsembleMean'].sel(time=slice('1950','2100')).values - np.mean(gw_index[model]['EnsembleMean'].sel(time=slice('1950','2100')).values[:30])
        y = regressor_EESC_GW(gw_ts,eesc_ts)
        coef = linear_regression(y,dic[model]['EnsembleMean'].sel(time=slice('1950','2100')).values)   
        pval = linear_regression_pvalues(y,dic[model]['EnsembleMean'].sel(time=slice('1950','2100')).values)
        dic_out[model]['coef'] = coef
        dic_out[model]['pval'] = pval  
    
    return dic_out

def bayes_factor(SSE_H0,SSE_H1,var_H0,var_H1):
    BF = np.exp(-np.sum(SSE_H1**2)/2*var_H1) / np.exp(-np.sum(SSE_H0**2)/2*var_H0)
    return BF
    
def bayes_factor_sum(SSE_H0,SSE_H1,var_H0,var_H1):
    """Bayes factor where SSE_H0 is an array"""
    BF = np.exp(-SSE_H1/2*var_H1) / np.mean(np.exp(-SSE_H0/2*var_H0))
    return BF

# from mpl_toolkits.mplot3d import Axes3D  # Not needed with Matplotlib 3.6.3
def rotation_x(angle):
    RX = [[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]]
    return RX

def rotation_y(angle):
    X = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]]
    return X

def rotation_z(angle):
    X = [[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    return X

color_iter = itertools.cycle(["navy","green","darkorange", "gold"])

def plot_results(X, Y, means, covariances, index, title):
    fig = plt.figure(figsize=plt.figaspect(1),dpi=300)  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        print(i)
        v, w = linalg.eigh(covar)
        v = 1.73*np.sqrt(v) #one std 1.73, two std = 3.46, three  std 5.20
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        ax.scatter(mean[0],mean[1],mean[2],marker='X',color=color,label='storyline '+str(i+1))
        for j in range(len(Y)):
            if Y[j] == i:
                ax.scatter(X[j,0],X[j,1],X[j,2],color=color)

        # Plot an ellipse to show the Gaussian component
        x_angle = np.arctan(u[1] / u[2])
        x_angle = 360.0 -  180.0 * x_angle / np.pi  # convert to degrees
        y_angle = np.arctan(u[2] / u[0])
        y_angle = 360.0 - 180.0 * y_angle / np.pi  # convert to degrees
        z_angle = np.arctan(u[1] / u[0])
        z_angle = 360.0 - 180.0 * z_angle / np.pi  # convert to degrees
                
        rx, ry, rz = 1/np.sqrt(np.array([v[0],v[1],v[2]]))

        #Rotation angle
        rotx = x_angle
        roty = y_angle
        rotz = z_angle

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v))
        y = ry * np.outer(np.sin(u), np.sin(v))
        z = rz * np.outer(np.ones_like(u), np.cos(v))

        ellipsoid = np.array([x,y,z])
        rotated_ellipsoid = ellipsoid
        for i in range(ellipsoid.shape[1]):
            for j in range(ellipsoid.shape[2]):
                rotated_ellipsoid[:,i,j] = np.matmul(rotation_x(-rotx),np.matmul(rotation_y(-roty),np.matmul(rotation_z(-rotz), ellipsoid[:,i,j]))) + mean
        # Plot:
        ax.plot_surface(rotated_ellipsoid[0], rotated_ellipsoid[1], rotated_ellipsoid[2],  rstride=4, cstride=4, color=color,alpha=0.2)

        # Adjustment of the axes, so that they all have the same span:
        max_radius = max(rx, ry, rz)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    #ax.view_init(0, 0, 0)
    ax.set_xlim(0,3)
    ax.set_ylim(1.3,1.9)
    ax.set_zlim(0.5,1.2)
    ax.set_xlabel('SPV [m/s/K]')
    ax.set_ylabel('TW [K/K]')
    ax.set_zlabel('CP warming [K/K]')
    ax.set_box_aspect(aspect=None, zoom=0.8)
    plt.legend()
    plt.show()



def plot_results_spherical(X, Y, means, covariances, index, title):
    fig = plt.figure(figsize=plt.figaspect(1),dpi=300)  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        ax.scatter(mean[0],mean[1],mean[2],marker='X',color=color,label='MEM')
        for j in range(len(Y)):
            if Y[j] == i:
                ax.scatter(X[j,0],X[j,1],X[j,2],color=color)

        rx, ry, rz = 1/np.sqrt(np.array([covar,covar,covar]))
        
        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v)) + mean[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) + mean[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + mean[2]

        ellipsoid = np.array([x,y,z])

        # Plot:
        ax.plot_surface(ellipsoid[0], ellipsoid[1],ellipsoid[2],  rstride=4, cstride=4, color=color,alpha=0.2)

        # Adjustment of the axes, so that they all have the same span:
        max_radius = max(rx, ry, rz)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    #ax.view_init(0, 0, 0)
    ax.set_xlim(0,3)
    ax.set_ylim(1.3,1.9)
    ax.set_zlim(0.5,1.2)
    ax.set_xlabel('SPV [m/s/K]')
    ax.set_ylabel('TW [K/K]')
    ax.set_zlabel('CP warming [K/K]')
    ax.set_box_aspect(aspect=None, zoom=0.8)
    plt.legend()
    plt.show()

