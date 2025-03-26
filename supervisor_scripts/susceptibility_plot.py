import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, f1_score
from scipy.stats import kde
import sys
import geopandas as gpd
import seaborn as sns
import pandas as pd

def plot_singles():
    # Assuming you have a list of file paths to your TIFF images
    # npy_files = ["Input/Japan/npy_arrays/masked_drought_japan.npy", "Input/Japan/npy_arrays/masked_pga_japan.npy",
    #              "Input/Japan/npy_arrays/masked_extreme_wind_japan.npy", "Output/japan_Flood_Susceptibility_base_model.npy",
    #              "Input/Japan/npy_arrays/masked_heatwave_japan.npy", "Output/Japan_Landslide_Susceptibility_base_model.npy",
    #              "Output/japan_Tsunami_Susceptibility_base_model.npy", "Input/Japan/npy_arrays/masked_volcano_japan.npy",
    #              "Input/Japan/npy_arrays/masked_fire_weather_japan.npy"]
    # npy_files = ["Input/Japan/npy_arrays/masked_drought_japan.npy", "Input/Japan/npy_arrays/masked_jshis_japan.npy",
    #              "Input/Japan/npy_arrays/masked_extreme_wind_japan.npy", "Output/japan_Flood_Susceptibility_base_model.npy",
    #              "Input/Japan/npy_arrays/masked_heatwave_japan.npy", "Output/Japan_Landslide_Susceptibility_base_model.npy",
    #              "Output/japan_Tsunami_Susceptibility_base_model.npy", "Input/Japan/npy_arrays/masked_volcano_japan.npy",
    #              "Input/Japan/npy_arrays/masked_fire_weather_japan.npy"]
    # npy_names = ['Drought', 'Earthquake', 'Extreme wind', 'Flood', 'Heatwave', 'Landslide', 'Tsunami', 'Volcano', 'Wildfire']


    npy_files = ["Input/Europe/npy_arrays/masked_curvature_Europe.npy", "Input/Europe/npy_arrays/masked_drought_Europe.npy"]

    npy_names = ['Curvature', 'Drought']

    # Create a 2x3 subplot grid
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Normalize the colorbar to the values between 0 and 1
    norm = Normalize(vmin=0, vmax=1)

    # Loop through the TIFF files and plot them on the subplot grid
    for i, npy_file in enumerate(npy_files):
        print (npy_names[i])
        # Read data
        npy_data = np.load(npy_file)
        if npy_names[i] == 'Volcano' or npy_names[i] == 'Extreme wind' or 'jshis' in npy_file:  # or npy_names[i] == 'Drought' or npy_names[i] == 'Heatwave':
            if npy_names[i] == 'Extreme wind':
                npy_data[npy_data > 1000] = 1000
            elif npy_names[i] == 'Volcano' or npy_names[i] == 'Wildfire':
                npy_data[npy_data > 10] = 10
            elif 'jshis' in npy_file:
                npy_data[npy_data < 20] = 1
                npy_data[(npy_data < 50) & (npy_data > 10)] = 2
                npy_data[(npy_data < 100) & (npy_data > 10)] = 3
                npy_data[(npy_data < 150) & (npy_data > 10)] = 4
                npy_data[npy_data > 10] = 5
            npy_data_1d = npy_data.reshape(-1, 1)

            # Apply PowerTransformer with a suitable method for your data
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            npy_data_transformed = pt.fit_transform(npy_data_1d)

            # Reshape the transformed data back to its original shape
            npy_data = npy_data_transformed.reshape(npy_data.shape)
        # if npy_names[i] == 'Flood' or npy_names[i] == 'Landslide' or npy_names[i] == 'Tsunami':
        #     npy_data[slope_data < 0.000001] = 0

        if npy_names[i] == 'Wildfire':
            npy_data[npy_data > 10] = 10

        npy_data[npy_data < 0.05] = 0
        npy_data = np.ma.masked_where(npy_data == 0, npy_data)

        # # Normalize the colorbar to the range [0, 1] (adjust as needed)
        norm = Normalize(vmin=0, vmax=np.max(npy_data))
        
        # Plot the data on the subplot grid
        row = i // 3
        col = i % 3
        # im = axs[row, col].imshow(npy_data, cmap='YlOrBr', extent=(128.38875, 146.4443, 30.72236, 46))  # norm=norm, # vmin=0, vmax=1,
        im = axs[row, col].contourf(np.flipud(npy_data), cmap='YlOrBr', extent=(128.38875, 146.4443, 30.72236, 46))  # norm=norm, # vmin=0, vmax=1,
        if npy_names[i] == 'Flood':
            cbar_im = im
                                
        # BoundingBox(left=128.38875000000002, bottom=30.722361111111113, right=146.44430555555556, top=46.00013888888889)

        # Set title for each subplot
        axs[row, col].set_title(npy_names[i], fontsize=16)

        # Set longitude tick labels
        axs[row, col].set_xticks(np.arange(130, 150, 5), crs=ccrs.PlateCarree())
        axs[row, col].xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

        # Set latitude tick labels
        axs[row, col].set_yticks(np.arange(30, 50, 5), crs=ccrs.PlateCarree())
        axs[row, col].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

        if row == 2:
            axs[row, col].set_xlabel('Longitude')
        if col == 0:
            axs[row, col].set_ylabel('Latitude')
        
    # Add coastlines and country borders
    for ax in axs.flat:
        # ax.coastlines()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.1, edgecolor='black')  # Adjust linewidth
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.1)
        ax.add_feature(cfeature.LAND, facecolor='#FFEB3B', alpha=0.1)
        ax.add_feature(cfeature.OCEAN, facecolor='#A6CAE0')
    
    # Add a colorbar for all subplots
    cbar = fig.colorbar(cbar_im, ax=axs, orientation='vertical', fraction=0.05, pad=0.2)
    cbar.set_label('Susceptibility', fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # Set the colorbar ticks and labels
    cbar.set_ticks(np.arange(0, 1.1, 0.1))  # Ticks from 0 to 1 with 0.1 increments
    cbar.set_ticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.1)])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, wspace=0.1)

    # Show the plot
    # plt.show()
    plt.savefig(f'Output/Europe/Individual_Susceptibility-Signals.png', dpi=300)

def plot_mh_map(typemodel):
    npy_files = ["Input/Japan/npy_arrays/masked_drought_japan.npy", "Input/Japan/npy_arrays/masked_jshis_japan.npy",
                "Input/Japan/npy_arrays/masked_extreme_wind_japan.npy", "Output/japan_Flood_Susceptibility_base_model.npy",
                "Input/Japan/npy_arrays/masked_heatwave_japan.npy", "Output/Japan_Landslide_Susceptibility_base_model.npy",
                "Output/japan_Tsunami_Susceptibility_base_model.npy", "Input/Japan/npy_arrays/masked_volcano_japan.npy",
                "Input/Japan/npy_arrays/masked_fire_weather_japan.npy"]
    npy_names = ['Drought', 'Earthquake', 'Extreme wind', 'Flood', 'Heatwave', 'Landslide', 'Tsunami', 'Volcano', 'Wildfire'] 

    elevation = np.load(f'Input/Japan/npy_arrays/masked_elevation_japan.npy')
    valid_cells_count = np.sum(~np.isnan(elevation))
    linear_mh = np.zeros_like(elevation)
    thresholds = pd.read_excel('F1_score_thresholds.xlsx')
    # sys.exit(0)

    for i, npy_file in enumerate(npy_files):
        npy_data = np.load(npy_file)

        # Scale and reshape the transformed data back to its original shape
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_feature = scaler.fit_transform(npy_data.reshape(-1, 1)).reshape(npy_data.shape)
        scaled_feature = np.nan_to_num(scaled_feature, nan=np.nan)

        if npy_names[i] == 'Earthquake' or npy_names[i] == 'Volcano':
            # scaled_feature[scaled_feature < 0.1] = np.nan
            threshold = 0.1
        elif npy_names[i] == 'Landslide' or npy_names[i] == 'Flood' or npy_names[i] == 'Tsunami':
            threshold = np.round(thresholds[thresholds.Variable == npy_names[i]]['Best Threshold'].values, 2)
        else:
            threshold = 0.5
        scaled_feature = np.where(scaled_feature >= threshold, 1, np.nan)
        
        percentage = np.round(np.sum(~np.isnan(scaled_feature)) / valid_cells_count * 100, 2)
        print(f'{npy_names[i]}: {percentage} share of total land')

        scaled_feature[np.isnan(scaled_feature)] = 0
        linear_mh += scaled_feature

    linear_mh[np.isnan(elevation)] = np.nan
    np.save(f'Output/japan_Multihazard_Susceptibility_linear.npy', linear_mh)

    percentage = np.round(np.sum(linear_mh == 0) / valid_cells_count * 100, 2)
    print(f'Low hazards: {percentage} share of total land')

    percentage = np.round(np.sum(linear_mh == 1) / valid_cells_count * 100, 2)
    print(f'Single hazards: {percentage} share of total land')

    percentage = np.round(np.sum(linear_mh > 1) / valid_cells_count * 100, 2)
    print(f'Multiple hazards: {percentage} share of total land')

    neural_mh = np.load(f'Output/japan_Multihazard_Susceptibility_meta_model_{typemodel}.npy')
    neural_mh = np.ma.masked_where(np.isnan(elevation), neural_mh)

    # threshold = np.round(thresholds[thresholds.Variable == 'Multi-hazard']['Best Threshold'].values, 2)
    threshold = 0.5
    neural_mh_binary = np.where(neural_mh >= threshold, 1, np.nan)
    percentage = np.round(np.sum(neural_mh_binary == 1) / valid_cells_count * 100, 2)
    print(f'Multi-hazards: {percentage} share of total land')

    # Create a 1x2 subplot grid
    fig, axs = plt.subplots(1, 2, figsize=(17, 7), subplot_kw={'projection': ccrs.PlateCarree()})

    for i, npy_data in enumerate([linear_mh, neural_mh]):
        if i == 0:
            title = 'Linear Multi-Hazard Map'
            label = 'Number of hazards'
        else:
            title = 'Multi-Hazard Susceptibility Map'
            label = 'Susceptibility'
        im = axs[i].imshow(npy_data, cmap='YlOrBr', extent=(128.38875, 146.4443, 30.72236, 46))
        # im = axs[i].contourf(np.flipud(npy_data), vmin=0, vmax=1, cmap='YlOrBr', extent=(128.38875, 146.4443, 30.72236, 46))

        # Add colorbar to the right
        cbar = plt.colorbar(im, ax=axs[i], orientation='vertical', pad=0.02, shrink=0.69)
        cbar.set_label(label, fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        # if i == 1:
        #     # Define the custom labels for the colorbar
        #     cbar.set_ticklabels([f'{i:.1f}' for i in np.linspace(0.1, 1, 10)])

        # Set title for each subplot
        axs[i].set_title(title, fontsize=32)

        # Set longitude tick labels
        axs[i].set_xticks(np.arange(130, 150, 5), crs=ccrs.PlateCarree())
        axs[i].xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

        # Set latitude tick labels
        axs[i].set_yticks(np.arange(30, 50, 5), crs=ccrs.PlateCarree())
        axs[i].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

        axs[i].set_xlabel('Longitude', fontsize=20)
        axs[i].set_ylabel('Latitude', fontsize=20)
        axs[i].tick_params(axis='x', labelsize=20)
        axs[i].tick_params(axis='y', labelsize=20)

        # Add coastlines and country borders
        # ax.coastlines()
        axs[i].add_feature(cfeature.COASTLINE, linewidth=0.1, edgecolor='black')  # Adjust linewidth
        axs[i].add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.1)
        axs[i].add_feature(cfeature.LAND, facecolor='#FFEB3B', alpha=0.1)
        axs[i].add_feature(cfeature.OCEAN, facecolor='#A6CAE0')

    # Make the layout tight
    plt.tight_layout()

    plt.savefig(f'Output/Multihazard_Susceptibility_2panel_{typemodel}.png', dpi=300)

def corr_heatmap():
    npy_files = ["Input/Japan/npy_arrays/masked_elevation_japan.npy",
                "Input/Japan/npy_arrays/masked_slope_japan.npy",
                "Input/Japan/npy_arrays/masked_curvature_japan.npy",
                "Input/Japan/npy_arrays/masked_aspect_japan.npy",
                "Input/Japan/npy_arrays/masked_accuflux_japan.npy",
                "Input/Japan/npy_arrays/masked_rivers_japan.npy",
                "Input/Japan/npy_arrays/masked_GEM_japan.npy",
                "Input/Japan/npy_arrays/masked_road_japan.npy",
                "Input/Japan/npy_arrays/masked_landcover_japan_flat.npy",
                "Input/Japan/npy_arrays/masked_HWSD_japan.npy",
                "Input/Japan/npy_arrays/masked_GLIM_japan.npy",
                "Input/Japan/npy_arrays/masked_NDVI_japan_flat.npy",
                "Input/Japan/npy_arrays/masked_temperature_daily_japan.npy",
                "Input/Japan/npy_arrays/masked_precipitation_daily_japan.npy",]
    npy_names = ['Elevation', 'Slope', 'Curvature', 'Aspect', 'Accumulated Water Flux', 'Rivers', 'Faultlines', 'Roads', 'Landcover',
                'Soil', 'Lithology', 'NDVI', 'Temperature', 'Precipitation']
    npy_names = ['EL', 'SL', 'CU', 'AS', 'AF', 'RI', 'FA', 'RO', 'LC',
                'SO', 'LI', 'NV', 'TM', 'PR']

    # Load the elevation map
    elevation_map = np.load("Input/Japan/npy_arrays/masked_elevation_japan.npy")

    # Load other raster maps and filter out cells with elevation greater than -9999
    data = []
    for file in npy_files:
        raster_data = np.load(file)
        raster_data_filtered = raster_data[elevation_map > -9999]
        data.append(raster_data_filtered)

    # np.save('Output/Correlation_variables.npy', data)
    # data = np.load('Output/Correlation_variables.npy')

    # Combine the data into a DataFrame
    df = pd.DataFrame(data=np.vstack(data).T, columns=npy_names)
    # df.to_excel("Output/Correlation_variables")
    # sys.exit(0)

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Sample df
    # sampled_df = df.sample(n=1000, random_state=42)
    sampled_df = df.copy()

    # Create figure and axes
    fig, axes = plt.subplots(nrows=len(npy_names), ncols=len(npy_names), figsize=(12, 10))

    # Create scatter plots for lower triangle
    for i in range(len(npy_names)):
        for j in range(len(npy_names)):
            if i > j:
                print(npy_names[i], npy_names[j])
                x = sampled_df[npy_names[j]]
                y = sampled_df[npy_names[i]]

                # Gaussian KDE plot correlated
                nbins = 100
                k = kde.gaussian_kde([x, y])
                xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                axes[i, j].pcolormesh(xi, yi, zi.reshape(xi.shape))
                # axes[i, j].scatter(sampled_df[npy_names[j]], sampled_df[npy_names[i]], alpha=0.7, color='skyblue')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

                if j == 0 and i == len(npy_names) - 1:
                    axes[i, j].set_ylabel(npy_names[i])
                    axes[i, j].set_xlabel(npy_names[j])
                elif j == 0:
                    axes[i, j].set_ylabel(npy_names[i])
                elif i == len(npy_names) - 1:
                    axes[i, j].set_xlabel(npy_names[j])

            elif i == j:
                print(npy_names[i])
                # Gaussian KDE self
                sns.kdeplot(sampled_df[npy_names[i]], ax=axes[i, j], color='k', shade=True, alpha=0.3)
                # axes[i, j].scatter(sampled_df[npy_names[j]], sampled_df[npy_names[i]], alpha=0.7, color='skyblue')
                # axes[i, j].axis('off')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            
                if j == 0:
                    axes[i, j].set_ylabel(npy_names[i])
                    axes[i, j].xaxis.tick_top()
                    axes[i, j].xaxis.set_label_position("top")
                    axes[i, j].set_xlabel(npy_names[j])
                elif i == len(npy_names) - 1:
                    axes[i, j].yaxis.tick_right()
                    axes[i, j].yaxis.set_label_position("right")
                    axes[i, j].set_ylabel(npy_names[i])
                    axes[i, j].set_xlabel(npy_names[j])
                else:
                    axes[i, j].set_ylabel('')
                    axes[i, j].set_xlabel('')

    # Show the correlation matrix in the upper triangle
    for i in range(len(npy_names)):
        for j in range(i+1, len(npy_names)):
            # Determine face color based on correlation value
            corr_value = correlation_matrix.iloc[i, j]

            # Plot heatmap with background shading
            axes[i, j].imshow([[corr_value]], cmap='RdBu_r', aspect='auto', extent=(0, 1, 0, 1), alpha=0.5, vmin=-1, vmax=1)
            axes[i, j].text(0.5, 0.5, '{:.2f}'.format(correlation_matrix.iloc[i, j]), ha='center', va='center', fontsize=12)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0 and j == len(npy_names) - 1:
                axes[i, j].xaxis.tick_top()
                axes[i, j].xaxis.set_label_position("top")
                axes[i, j].set_xlabel(npy_names[j])
                axes[i, j].yaxis.tick_right()
                axes[i, j].yaxis.set_label_position("right")
                axes[i, j].set_ylabel(npy_names[j])
            elif i == 0:
                axes[i, j].xaxis.tick_top()
                axes[i, j].xaxis.set_label_position("top")
                axes[i, j].set_xlabel(npy_names[j])
                axes[i, j].set_ylabel('')
            elif j == len(npy_names) - 1:
                axes[i, j].yaxis.tick_right()
                axes[i, j].yaxis.set_label_position("right")
                axes[i, j].set_ylabel(npy_names[i])
                axes[i, j].set_xlabel('')
            else:
                axes[i, j].set_xlabel('')
                axes[i, j].set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'Output/Correlation_heatmap.png', dpi=300)
    plt.show()
    sys.exit(0)

def plot_input():
    #################### PLOT INPUT MAPS ######################
    npy_files = ["Input/Japan/npy_arrays/masked_elevation_japan.npy",
                "Input/Japan/npy_arrays/masked_slope_japan.npy",
                "Input/Japan/npy_arrays/masked_curvature_japan.npy",
                "Input/Japan/npy_arrays/masked_aspect_japan.npy",
                "Input/Japan/npy_arrays/masked_accuflux_japan.npy",
                "Input/Japan/npy_arrays/masked_rivers_japan.npy",
                "Input/Japan/npy_arrays/masked_GEM_japan.npy",
                "Input/Japan/npy_arrays/masked_road_japan.npy",
                "Input/Japan/npy_arrays/masked_landcover_japan_flat.npy",
                "Input/Japan/npy_arrays/masked_HWSD_japan.npy",
                "Input/Japan/npy_arrays/masked_GLIM_japan.npy",
                "Input/Japan/npy_arrays/masked_NDVI_japan_flat.npy",
                "Input/Japan/npy_arrays/masked_temperature_daily_japan.npy",
                "Input/Japan/npy_arrays/masked_precipitation_daily_japan.npy",
                "Input/Japan/npy_arrays/masked_wind_speed_daily_japan.npy",
                "Input/Japan/npy_arrays/masked_wind_direction_daily_japan.npy",]
    npy_names = ['Elevation', 'Slope', 'Curvature', 'Aspect', 'Accumulated Water Flux', 'Rivers', 'Faultlines', 'Roads', 'Landcover',
                'Soil', 'Lithology', 'NDVI', 'Temperature', 'Precipitation', 'Wind Speed', 'Wind Direction']

    fig, axs = plt.subplots(4, 4, figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    for i, npy_file in enumerate(npy_files):
        print(npy_file)
        # Read data
        npy_data = np.load(npy_file)
        
        if npy_names[i] == 'Curvature':
            npy_data[npy_data == -9999] = np.nan
            npy_data[npy_data < 0] = 0
            npy_data[npy_data > 0] = 1
        elif npy_names[i] == 'Accumulated Water Flux':
            npy_data[npy_data == -9999] = np.nan
            npy_data[npy_data < 3] = 0
            npy_data[npy_data >= 3] = 1
        elif npy_names[i] == 'Faultlines':
            npy_data[npy_data == -9999] = np.nan
            # npy_data[npy_data < 100] = 0
            npy_data[npy_data >= 10] = 10
        elif npy_names[i] == 'Rivers':
            npy_data[npy_data == -9999] = np.nan
            npy_data[npy_data >= 1] = 1

            # scaler = MinMaxScaler(feature_range=(0, 1))
            # scaled_feature = scaler.fit_transform(npy_data.reshape(-1, 1)).reshape(npy_data.shape)
            # npy_data = np.nan_to_num(scaled_feature, nan=-9999)
        # npy_data = np.ma.masked_where(npy_data == 0, npy_data)

        # # Normalize the colorbar to the range [0, 1] (adjust as needed)
        # norm = Normalize(vmin=0, vmax=np.max(npy_data))
        
        # Plot the data on the subplot grid
        row = i // 4
        col = i % 4
        print(row, col)
        im = axs[row, col].imshow(npy_data, cmap='summer', # norm=norm, # vmin=0, vmax=1,
                                extent=(128.38875, 146.4443, 30.72236, 46))
        # BoundingBox(left=128.38875000000002, bottom=30.722361111111113, right=146.44430555555556, top=46.00013888888889)

        # Set title for each subplot
        axs[row, col].set_title(npy_names[i], fontsize=16)

        # Set longitude tick labels
        axs[row, col].set_xticks(np.arange(130, 150, 5), crs=ccrs.PlateCarree())
        axs[row, col].xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

        # Set latitude tick labels
        axs[row, col].set_yticks(np.arange(30, 50, 5), crs=ccrs.PlateCarree())
        axs[row, col].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

        if row == 3 and col == 0:
            axs[row, col].set_xlabel('Longitude')
            axs[row, col].set_ylabel('Latitude')
        elif row == 3:
            axs[row, col].set_xlabel('Longitude')
            axs[row, col].set_yticks([])
        elif col == 0:
            axs[row, col].set_xticks([])
            axs[row, col].set_ylabel('Latitude')
        else:
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
        

    # Add coastlines and country borders
    for ax in axs.flat:
        # ax.coastlines()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.1, edgecolor='black')  # Adjust linewidth
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.1)
        ax.add_feature(cfeature.LAND, facecolor='#FFEB3B', alpha=0.1)
        ax.add_feature(cfeature.OCEAN, facecolor='#A6CAE0')


    # # Add a colorbar for all subplots
    # cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.05, pad=0.2)
    # cbar.set_label('Susceptibility', fontsize=16)
    # cbar.ax.tick_params(labelsize=12)
    # # Define the custom labels for the colorbar
    # cbar.set_ticklabels([f'{i:.1f}' for i in np.linspace(0.1, 1, 10)])

    # Adjust layout for better spacing
    plt.tight_layout()
    # plt.subplots_adjust(right=0.85, wspace=0.1)

    # Show the plot
    # plt.show()
    plt.savefig(f'Output/Input_maps.png', dpi=300)

def plot_stacked(layers='signals', crop=True):
    if layers == 'input':
        npy_files = ["Input/Japan/npy_arrays/masked_ldm_japan.npy",
                    "Input/Japan/npy_arrays/masked_temperature_daily_japan.npy",
                    "Input/Japan/npy_arrays/masked_landcover_japan_flat.npy",
                    "Input/Japan/npy_arrays/masked_GEM_japan.npy",
                    "Input/Japan/npy_arrays/masked_elevation_japan.npy"]
        npy_names = ['Hazard', 'Atmospheric', '(Sub)Surface', 'Distance', 'Terrain']
        pastel = ListedColormap(['#B7E1A1', '#A2D699', '#8EC192', '#7BBF8A', '#69AD83', '#579A7C', '#468877', '#357570', '#24636B', '#135065'])
        figsize = (10, 7)
        ztick = 0.2
        fontsize = 27
    elif layers == 'signals':
        npy_files = ["Input/Japan/npy_arrays/masked_drought_japan.npy", "Input/Japan/npy_arrays/masked_pga_japan.npy",
                    "Input/Japan/npy_arrays/masked_extreme_wind_japan.npy", "Output/japan_Flood_Susceptibility_base_model.npy",
                    "Input/Japan/npy_arrays/masked_heatwave_japan.npy", "Output/Japan_Landslide_Susceptibility_base_model.npy",
                    "Output/japan_Tsunami_Susceptibility_base_model.npy", "Input/Japan/npy_arrays/masked_volcano_japan.npy",
                    "Input/Japan/npy_arrays/masked_fire_weather_japan.npy"]
        npy_files = npy_files[::-1]
        npy_names = ['Drought', 'Earthquake', 'Extreme wind', 'Flood', 'Heatwave', 'Landslide', 'Tsunami', 'Volcano', 'Wildfire']
        npy_names = npy_names[::-1]
        fontsize = 20

        # Calculate the maximum length of the names
        max_length = max(len(name) for name in npy_names)

        # Pad each name with spaces to make them all the same length
        npy_names = [name.rjust(max_length) for name in npy_names]
        print(npy_names)

        colors = [
            (0.98, 0.85, 0.85),  # Very light pastel pink
            (0.96, 0.75, 0.85),    # Light pastel pink
            (0.95, 0.65, 0.85),     # Pastel pink
            (0.8, 0.6, 0.9),     # Light pastel purple
            (0.7, 0.5, 0.8),     # Pastel purple
            (0.6, 0.4, 0.7),     # Dark pastel purple
            (0.5, 0.3, 0.6)      # Very dark pastel purple
        ]

    #     colors = [
    #     (0.98, 0.85, 0.85),  # Very light pastel pink
    #     (0.96, 0.75, 0.85),  # Light pastel pink
    #     (0.95, 0.65, 0.85),  # Pastel pink
    #     (0.85, 0.45, 0.85),  # Light pastel purple
    #     (0.75, 0.35, 0.75),  # Pastel purple
    #     (0.65, 0.25, 0.65),  # Dark pastel purple
    #     (0.55, 0.15, 0.55),  # Very dark pastel purple
    # ]
        pastel = LinearSegmentedColormap.from_list('PastelPinkPurple', colors)
        # pastel = ListedColormap(['#F4C2C2', '#F9D2EB', '#F3C0FA', '#DFCAF4', '#D2E3EF', '#D7EEFB', '#C3F5EF', '#C9F8D1', '#DEFCB5', '#E6F7A9'])
        # pastel = ListedColormap(['#F4C2C2', '#F9D2EB', '#F3C0FA', '#DFCAF4', '#D2E3EF', '#D7EEFB', '#C3F5EF', '#C9F8D1', '#DEFCB5', '#B7C2E2'])
        figsize = (10, 10)
        ztick = 0.1

    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize, facecolor='none')
    ax = fig.add_subplot(111, projection='3d')

    # Define z-height levels
    z_levels = [0, 1, 2]  # Define z-height levels for each layer

    for i, npy_file in enumerate(npy_files):
        print(npy_file)
        # Read data
        npy_data = np.load(npy_file)

        # Define the color levels for each layer
        color_levels = np.linspace(0.01+i/10, 0.5+i/10, 20)
        color_levels = np.linspace(0.1, 0.9, 20)

        # Create colormap with varying shades of green
        colors = [plt.cm.Greens(color_level) for color_level in color_levels]

        # Rotate layers
        if crop:
            npy_data = npy_data[3500:3800, 3600:3900]  # for small grid

        scaler = MinMaxScaler(feature_range=(0, 1))
        npy_data = scaler.fit_transform(npy_data.reshape(-1, 1)).reshape(npy_data.shape)
        if crop:
            npy_data = np.rot90(npy_data, k=1)  # for small grid
        else:
            npy_data = np.rot90(np.flipud(npy_data), k=0)  # For whole Japan
        

        # Define grid coordinates
        x = np.arange(npy_data.shape[1])
        y = np.arange(npy_data.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot each layer at different z-height levels
        # ax.plot(X, Y, np.full_like(npy_data, i), cmap='summer') #, facecolors=plt.cm.summer(npy_data))
        # ax.plot_surface(X, Y, np.full_like(npy_data, i), cmap='summer', facecolors=plt.cm.summer(npy_data))
        color = plt.cm.Greens(i/10+0.1)
        ax.contourf(X, Y, npy_data, cmap=pastel, offset=i*ztick) # colors=colors[::-1]
        if layers == 'input':
            ax.text(npy_data.shape[1], npy_data.shape[0], i*ztick, npy_names[i], va='center', ha='left', fontsize=fontsize)
        elif layers == 'signals':
            ax.text(-240, 0, i*ztick-0.1, npy_names[i], va='center', ha='left', fontsize=fontsize)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if layers == 'input':
        ax.set_box_aspect((1, 1, 1.1))
    elif layers == 'signals':
        ax.set_box_aspect((1, 1, 2))

    # Set transparent background
    ax.patch.set_alpha(0)

    # Remove axis features
    ax.axis('off')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'Output/Stacked_{layers}_maps.png', dpi=300)
    plt.show()

def plot_misc_flowchart():
    # Generate x values
    x = np.linspace(0, 15, 150)

    # Calculate y values for the first line (gradually decrease and then rapidly)
    y1_gradual = np.where(x <= 8, 1 - 0.1 * (x / 8), 0.8 - 0.4 * ((x - 8) / 7))

    # Calculate y values for the second line (gradually increase and then rapidly)
    y2_gradual = np.where(x <= 8, 0.15 * (x / 8), 0.15 + 0.35 * ((x - 8) / 7))

    # Find intersection point
    intersection_index = np.argmax(y1_gradual <= y2_gradual)
    threshold_x = x[intersection_index]
    threshold_y = y1_gradual[intersection_index]

    # Plot the lines
    plt.plot(x, y1_gradual, label='Decreasing Line', color='deepskyblue')
    plt.plot(x, y2_gradual, label='Increasing Line', color='mediumseagreen')

    # Plot red dashed line vertical from threshold
    plt.plot([threshold_x, threshold_x], [0, threshold_y], color='red', linestyle='--')

    # Annotate the intersection point with optimal threshold
    arrow_x = threshold_x - 3
    arrow_y = threshold_y + 0.05
    arrow_origin_y = arrow_y + 0.2  # Adjust the y-axis value of the arrow origin
    plt.annotate('Optimal Threshold', xy=(threshold_x, threshold_y), xytext=(arrow_x, arrow_origin_y),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5', color='black', lw=2, mutation_scale=30))

    # Add labels and title
    plt.xlabel('Threshold')
    plt.ylabel('Score')

    plt.savefig(f'Output/Sensitivity_threshold.png', dpi=300)
    plt.close()

    # Generate data
    x = np.linspace(0, 5, 100)
    y = np.exp(x) - 1  # Exponential increase starting from 0

    # Threshold value
    threshold = 4

    # Plot the exponential curve
    plt.plot(x, y)

    # Fill area under the curve before and after threshold
    plt.fill_between(x, y, where=(x <= threshold), color='lightblue', alpha=0.5)
    plt.fill_between(x, y, where=(x > threshold), color='lightgreen', alpha=0.5)

    # Draw the threshold line
    plt.axvline(x=threshold, color='black', linestyle='-')

    # Draw dashed line at the start of the curve
    plt.axvline(x=x[0], linestyle='--', color='black')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], linestyle='-', color='black', label='Threshold'),
        Line2D([0], [0], linestyle='--', color='black', label='Mean')
    ]

    # Add custom legend
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.07, 1), frameon=False)

    # Remove x and y ticks
    plt.xticks([])
    plt.yticks([])

    # Add labels and title
    plt.xlabel('Hazard Intensity')
    plt.ylabel('Risk')
    # plt.title('Peak Over Threshold Schematic')

    plt.savefig(f'Output/Peak_over_threshold.png', dpi=300)
    plt.close()

    fig = plt.figure(figsize=(10, 10))

    # Define the pastel red colors
    colors = [(1, 1, 1), (1, 0.8, 0.8), (1, 0.6, 0.6), (1, 0.4, 0.4), (1, 0.2, 0.2), (1, 0, 0)]
    colors = [(1, 1, 1), (1, 1, 0.8), (1, 1, 0.6), (1, 1, 0.4), (0.95, 0.95, 0.2), (0.9, 0.9, 0.1), (0.8, 0.8, 0)]
    colors = [(1, 1, 0.8), (1, 1, 0.7), (1, 1, 0.6), (1, 1, 0.5), 
          (0.95, 0.95, 0.4), (0.9, 0.9, 0.3), (0.8, 0.8, 0.2),
          (0.7, 0.7, 0.15), (0.6, 0.6, 0.1), (0.5, 0.5, 0.05)]
    colors = [(1, 1, 0.8), (1, 1, 0.7), (1, 1, 0.6), (1, 1, 0.5),
          (0.95, 0.95, 0.4), (0.9, 0.9, 0.3), (0.85, 0.85, 0.2),
          (0.8, 0.8, 0.15), (0.7, 0.7, 0.1), (0.6, 0.6, 0.05),
          (0.5, 0.5, 0)]
    colors = [(1, 1, 0.8), (1, 1, 0.6), (1, 1, 0.4), (1, 1, 0.2),
          (1, 0.8, 0), (1, 0.6, 0), (1, 0.4, 0), (1, 0.2, 0),
          (0.2, 0.1, 0)]


    # Create a LinearSegmentedColormap
    pastel = LinearSegmentedColormap.from_list('Pastel Reds', colors)
    pastel = LinearSegmentedColormap.from_list('Pastel Yellows', colors)

    npy_data = np.load("Output/japan_Multihazard_Susceptibility_meta_model_MLP.npy")
    # npy_data = npy_data[3500:3700, 3600:3800]
    npy_data = npy_data[700:1000, 5300:5700]

    # Define grid coordinates
    x = np.arange(npy_data.shape[1])
    y = np.arange(npy_data.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create a masked array where 0 values are masked
    masked_data = np.ma.masked_where(npy_data == 0, np.flipud(npy_data))

    plt.contourf(X, Y, masked_data, cmap=pastel)
    # plt.imshow(npy_data, cmap=pastel)

    # Remove x and y ticks
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'Output/MH_flowchart.png', dpi=300)

def plot_region():
    #################### PLOT REGION ######################
    filename = 'gem-global-active-faults-master/geopackage/gem_active_faults_harmonized.gpkg'
    filename = 'gem-global-active-faults-master/gem_selected.gpkg'
    gem = gpd.read_file(filename)

    regions = ['Hokkaido', 'Tohoku', 'Kanto', 'Chubu', 'Kinki', 'Chugoku', 'Shikoku', 'Kyushu']
    adm = gpd.read_file('Japan/jpn_admbnda_adm1_2019.shp')
    adm['ADM1_EN'] = adm['ADM1_EN'].str.strip()

    # Assuming 'adm' is your DataFrame containing the shapefile data
    adm['Region'] = ''

    # Dictionary mapping prefecture codes to region names
    region_mapping = {
        'JP01': 'Hokkaido',
        'JP02': 'Tohoku',
        'JP03': 'Tohoku',
        'JP04': 'Tohoku',
        'JP05': 'Tohoku',
        'JP06': 'Tohoku',
        'JP07': 'Tohoku',
        'JP08': 'Kanto',
        'JP09': 'Kanto',
        'JP10': 'Kanto',
        'JP11': 'Kanto',
        'JP12': 'Kanto',
        'JP13': 'Kanto',
        'JP14': 'Kanto',
        'JP15': 'Chubu',
        'JP16': 'Chubu',
        'JP17': 'Chubu',
        'JP18': 'Chubu',
        'JP19': 'Chubu',
        'JP20': 'Chubu',
        'JP21': 'Chubu',
        'JP22': 'Chubu',
        'JP23': 'Chubu',
        'JP24': 'Kansai',
        'JP25': 'Kansai',
        'JP26': 'Kansai',
        'JP27': 'Kansai',
        'JP28': 'Kansai',
        'JP29': 'Kansai',
        'JP30': 'Kansai',
        'JP31': 'Chugoku',
        'JP32': 'Chugoku',
        'JP33': 'Chugoku',
        'JP34': 'Chugoku',
        'JP35': 'Chugoku',
        'JP36': 'Shikoku',
        'JP37': 'Shikoku',
        'JP38': 'Shikoku',
        'JP39': 'Shikoku',
        'JP40': 'Kyushu',
        'JP41': 'Kyushu',
        'JP42': 'Kyushu',
        'JP43': 'Kyushu',
        'JP44': 'Kyushu',
        'JP45': 'Kyushu',
        'JP46': 'Kyushu',
        'JP47': 'Kyushu'
    }

    # Assigning region names based on prefecture codes
    adm['Region'] = adm['ADM1_PCODE'].map(region_mapping)

    # Dissolve geometries by region
    regions = adm.dissolve(by='Region')

    # Get centroid coordinates for the prefectures
    centroids = adm.centroid

    # Sea points for drawing lines (you may need to adjust these)
    sea_points = {
        'Chiba': (141, 35),
        'Niigata': (138.4, 38.8),
        'Aichi': (137.3, 34.2),
        'Mie': (136.5, 33.8),
        'Ishikawa': (136, 37),
    }

    # Locations of landmarks
    landmarks = {
        'Mount Fuji': (138.7277, 35.3606),
        'Sakurajima': (130.6576, 31.5908),
        'Aso': (131.0866, 32.8841),
        'Asama': (138.5285, 36.4064)
    }

    # Japan Trench, Sagami through, and Nankai trough subduction zones and Izu–Bonin–Mariana Arc convergent boundary

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot GEM
    gem.plot(ax=ax, edgecolor='brown', zorder=0)

    # Set extent
    ax.set_extent((128.38875, 146.4443, 30.72236, 46))

    # Add ocean and land background
    ax.add_feature(cfeature.OCEAN, facecolor='#A6CAE0')
    ax.add_feature(cfeature.LAND, facecolor='#FFEB3B', alpha=0.1)

    # Plot regions with pastel colors
    colors = ['#FFD700', '#FFA07A', '#90EE90', '#AFEEEE', '#FFB6C1', '#87CEFA', '#F0E68C', '#CD5C5C']
    regions.plot(ax=ax, edgecolor='black', facecolor=colors, alpha=1)

    # Add prefecture border outlines
    adm.plot(ax=ax, edgecolor='black', linewidth=0.5, facecolor='none')

    # Create legend
    legend_patches = [mpatches.Patch(color=color, label=region) for region, color in zip(regions.index, colors)]
    ax.legend(handles=legend_patches, loc='upper left', fontsize='x-large')

    # Draw lines and annotate prefecture names
    for prefecture, centroid in zip(adm['ADM1_EN'], centroids):
        centroid_coords = (centroid.x, centroid.y)
        if prefecture in sea_points:
            sea_coords = sea_points[prefecture]
            ax.plot([centroid_coords[0], sea_coords[0]], [centroid_coords[1], sea_coords[1]], color='k', linestyle='-')
            if prefecture == 'Ishikawa':
                ax.text(sea_coords[0]-1.46, sea_coords[1]+0.1, prefecture, fontsize=10, color='k', va='top', weight='bold') # , ha='center'
            elif prefecture == 'Niigata':
                ax.text(sea_coords[0]-1.2, sea_coords[1]+0.1, prefecture, fontsize=10, color='k', va='top', weight='bold') # , ha='center'
            else:
                ax.text(sea_coords[0], sea_coords[1], prefecture, fontsize=10, color='k', va='top', weight='bold') # , ha='center'

            # # Calculate text coordinates
            # text_x = sea_coords[0] + (centroid_coords[0] - sea_coords[0]) * 0.9  # Adjust multiplier as needed
            # text_y = sea_coords[1] + (centroid_coords[1] - sea_coords[1]) * 0.9  # Adjust multiplier as needed
            
            # ax.text(text_x, text_y, prefecture, fontsize=10, color='k', weight='bold')

    # Plot landmarks
    for landmark, coordinates in landmarks.items():
        # Calculate text coordinates
        if landmark == 'Mount Fuji':
            text_x, text_y = coordinates[0], coordinates[1] - 1.8
        elif landmark == 'Aso':
            text_x, text_y = coordinates[0] + 0.9, coordinates[1] - 0.4
        elif landmark == 'Asama':
            text_x, text_y = coordinates[0] + 2.5, coordinates[1]
        elif landmark == 'Sakurajima':
            text_x, text_y = coordinates[0] + 1, coordinates[1] + 0.1

        ax.plot([coordinates[0], text_x], [coordinates[1], text_y], color='k', linestyle='-')
        ax.plot(coordinates[0], coordinates[1], marker='^', color='red', markersize=8, markeredgewidth=1, markeredgecolor='black')
        ax.text(text_x+0.1, text_y, landmark, fontsize=10, ha='left', weight='bold')

    # Japan Trench, Sagami through, and Nankai trough subduction zones and Izu–Bonin–Mariana Arc convergent boundary
    ax.text(143.5, 36.6, "Japan Trench", fontsize=10, ha='left', weight='bold', rotation=70)
    ax.text(134.5, 31.9, "Nankai trough", fontsize=10, ha='left', weight='bold', rotation=23)
    ax.text(139, 34.1, "Sagami through", fontsize=10, ha='left', weight='bold', rotation=-14)
    ax.text(141.6, 30.8, "Izu-Bonin-Mariana Arc", fontsize=10, ha='left', weight='bold', rotation=95)

    # Show plot
    plt.savefig(f'Output/Region_map.png', dpi=300)

def find_best_f1_threshold():
    npy_files = [f'Output/japan_Flood_Susceptibility_base_model.npy', f'Output/japan_Tsunami_Susceptibility_base_model.npy',
                 f'Output/japan_Landslide_Susceptibility_base_model.npy', f'Output/japan_Multihazard_Susceptibility_meta_model_MLP.npy']
    label_files = [f'Input/Japan/npy_arrays/masked_flood_surge_japan.npy', f'Input/Japan/npy_arrays/masked_tsunami_japan.npy',
                   f'Input/Japan/npy_arrays/masked_ldm_japan.npy', f'Input/Japan/npy_arrays/masked_multi_hazard_japan.npy']
    variables = ['Flood', 'Tsunami', 'Landslide', 'Multi-hazard']
    npy_files = [f'Output/japan_Multihazard_Susceptibility_meta_model_MLP.npy']
    label_files = [f'Input/Japan/npy_arrays/masked_multi_hazard_japan.npy']
    variables = ['Multi-hazard']
    elevation = np.load(f'Input/Japan/npy_arrays/masked_elevation_japan.npy')
    land_mask = ~np.isnan(elevation)

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Variable', 'AUC', 'Best Threshold', 'Best F1-score'])

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Iterate over each hazard
    for i, (npy_file, label_file, variable) in enumerate(zip(npy_files, label_files, variables)):
        # if i != 3:
        #     continue
        # Load data
        npy_data = np.load(npy_file)
        label_data = np.load(label_file)
        if variable == 'Multi-hazard':
            label_data[label_data > 0] = 1

        # Mask sea values in label_data
        land_label_data = label_data[land_mask].flatten()

        # Mask sea values in npy_data
        npy_data_masked = npy_data.copy()
        npy_data_masked[np.isnan(label_data)] = np.nan  # Mask sea values

        # Flatten npy_data_masked considering only land values
        land_npy_data = npy_data_masked[land_mask].flatten()

        best_threshold = None
        best_f1 = -1

        thresholds = np.linspace(0, 1, 100)
        # thresholds = [0.5]
        for threshold in thresholds:
            # Convert to binary predictions based on the current threshold
            npy_data_binary = np.where(land_npy_data >= threshold, 1, 0)
            
            # Calculate F1-score
            current_f1 = f1_score(land_label_data, npy_data_binary)
            
            # Update best threshold and F1-score
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
            
            # Print metrics when the current threshold matches a multiple of 0.1
            if np.isclose(threshold % 0.1, 0) and threshold > 0:  # Check if threshold is a multiple of 0.1 (excluding 0.0)
                print(f"Threshold: {threshold:.2f}")

        # Convert predictions to binary using the best threshold
        final_npy_data_binary = np.where(land_npy_data >= best_threshold, 1, 0)

        # Compute ROC curve and AUC using the final binary predictions ##### WHICH ONE OF FOLLOWING SHOULD BE??
        fpr, tpr, _ = roc_curve(land_label_data, final_npy_data_binary)
        fpr, tpr, _ = roc_curve(land_label_data, land_npy_data)
        roc_auc = auc(fpr, tpr)

        # Determine subplot row and column
        row, col = divmod(i, 2)

        # Plot ROC curve
        axs[row, col].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axs[row, col].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[row, col].set_xlabel('False Positive Rate')
        axs[row, col].set_ylabel('True Positive Rate')
        axs[row, col].set_title(variable)
        axs[row, col].legend(loc="lower right")

        # Print AUC and other metric values
        print(f"Hazard: {variable}")
        print(f"AUC: {roc_auc:.4f}")
        print(f"Best Threshold for F1-score: {best_threshold:.2f}")
        print(f"Best F1-score: {best_f1:.4f}")

        # Store results in the DataFrame
        results_df = results_df._append({'Variable': variable, 
                                        'AUC': roc_auc, 
                                        'Best Threshold': best_threshold, 
                                        'Best F1-score': best_f1}, 
                                    ignore_index=True)

    # Save results to Excel file
    results_df.to_excel('Output/F1_score_thresholds.xlsx', index=False)

    # Display the combined plot
    plt.tight_layout()
    plt.savefig(f'Output/ROC_plots.png', dpi=300)

def RSA(hazard):
    # Check for Residual Spatial Autocorrelation
    # from libpysal.weights import lat2W, KNN
    # from esda.moran import Moran
    # from esda.moran import Moran_Local
    # from scipy.spatial.distance import pdist, squareform
    # from shapely.geometry import Point

    # Load predicted susceptibility
    predicted = np.load(f'Output/japan_{hazard}_Susceptibility_base_model.npy')

    # Load labelled data (ground truth)
    ground_truth = np.load(f'Input/Japan/npy_arrays/masked_flood_surge_japan.npy')

    # Load elevation for masking
    elevation = np.load(f'Input/Japan/npy_arrays/masked_elevation_japan.npy')

    # Compute residuals (errors)
    residuals = predicted - ground_truth

    # Convert to 1D (ignore NaNs for land-only data)
    valid_mask = ~np.isnan(elevation)
    residuals_flat = residuals[valid_mask]

    # Plot Residuals
    print('Residuals')
    plt.figure(figsize=(10, 5))
    sns.heatmap(residuals, cmap="RdBu", center=0)
    plt.title("Residuals Map (Predicted - Ground Truth)")
    plt.savefig(f'Output/Residual_map.png', dpi=300)

    # # Compute Variogram (distance-based spatial autocorrelation)
    # print('Variogram')
    # coords = np.argwhere(valid_mask)
    # distances = squareform(pdist(coords))  # Pairwise distances
    # print('distances')
    # errors = residuals_flat[:, None] - residuals_flat  # Error differences
    # print('values')
    # variogram_values = np.var(errors, axis=0)

    # # Plot Variogram
    # print('plotting')
    # plt.figure(figsize=(8, 5))
    # plt.plot(np.sort(distances.flatten()), np.sort(variogram_values), marker="o", linestyle="None")
    # plt.xlabel("Distance")
    # plt.ylabel("Variance of Errors")
    # plt.title("Variogram of Residuals")
    # plt.savefig(f'Output/Variogram_residuals.png', dpi=300)

    # Create spatial weight matrix
    print('Moran')
    # height, width = residuals.shape
    # w = lat2W(height, width, rook=False)  # Queen contiguity (diagonal neighbors too)

    # w = KNN.from_array(np.argwhere(valid_mask), k=8)
    print('coords')
    coords = np.argwhere(valid_mask) / np.max(residuals.shape)  # Normalize to (0,1)
    print('w')
    w = KNN.from_array(coords, k=8)

    # Compute Moran’s I (global spatial autocorrelation)
    print('moran')
    moran = Moran(residuals_flat, w)
    print(f"Moran's I: {moran.I}, p-value: {moran.p_norm}")

    # Compute Local Moran’s I
    local_moran = Moran_Local(residuals_flat, w)

    # Convert results to GeoDataFrame for visualization
    gdf = gpd.GeoDataFrame(
        {"Moran_I": local_moran.Is, "p_value": local_moran.p_sim},
        geometry=[Point(x, y) for x, y in coords]
    )

    # Filter significant hotspots (p < 0.05)
    gdf["Hotspot"] = np.where((gdf["p_value"] < 0.05) & (gdf["Moran_I"] > 0), "Hotspot", "Not Significant")
    gdf["Coldspot"] = np.where((gdf["p_value"] < 0.05) & (gdf["Moran_I"] < 0), "Coldspot", "Not Significant")

    # Plot Local Moran's I (Hotspots/Coldspots)
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(column="Moran_I", cmap="bwr", legend=True, ax=ax, markersize=1)
    plt.title("Local Moran’s I (Hotspots & Coldspots)")
    plt.savefig(f'Output/Moran_map.png', dpi=300)


# plot_singles()
# plot_mh_map('MLP')
# plot_mh_map('Logistic')
# plot_mh_map('base_model')
# corr_heatmap()
# plot_input()
# plot_stacked(layers='signals', crop=True)
# plot_stacked(layers='input', crop=True)
# plot_misc_flowchart()
# plot_region()
# find_best_f1_threshold()
# label_data = np.load(f'Input/Japan/npy_arrays/masked_multi_hazard_japan.npy')
# label_data[label_data > 0] = 1
RSA('Flood')
# sys.exit(0)



# npy_files = ["Input/Japan/npy_arrays/masked_jshis_japan.npy",
#             "Input/Japan/npy_arrays/masked_drought_japan.npy",
#             "Input/Japan/npy_arrays/masked_extreme_wind_japan.npy",
#             "Input/Japan/npy_arrays/masked_fire_weather_japan.npy",
#             "Input/Japan/npy_arrays/masked_heatwave_japan.npy",
#             "Input/Japan/npy_arrays/masked_volcano_japan.npy",
#             "Input/Japan/npy_arrays/masked_multi_hazard_japan.npy",]

# for npy_file in npy_files:
#     npy_data = np.load(npy_file)
#     npy_data = npy_data[4100:4800, 1300:2400]
#     npy_file = npy_file.replace('japan', 'Shikoku')
#     npy_file = npy_file.replace('Japan', 'Shikoku')
#     np.save(npy_file, npy_data)

sys.exit(0)
