import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_npy(npy_file_path: str, output_file_path: str) -> None:
    """
    Normalizes the values in a .npy file to the range [0, 1] using MinMaxScaler.
    Saves the normalized array to the specified output path.

    Args:
        npy_file_path (str): Path to the input .npy file.
        output_file_path (str): Path to save the normalized .npy file.
    """
    # Load the array
    data = np.load(npy_file_path)

    # Reshape for MinMaxScaler
    reshaped_data = data.reshape(-1, 1)

    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(reshaped_data)

    # Reshape back to original shape
    normalized_data = normalized_data.reshape(data.shape)

    # Save the normalized array
    np.save(output_file_path, normalized_data)


if __name__ == "__main__":
    npy_files = [
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_soil_moisture_root_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_soil_moisture_surface_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_NDVI_Europe_flat.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_landcover_Europe_flat.npy",
        # "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wildfire_Europe.npy",
        # "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_multi_hazard_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wind_direction_daily_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wind_speed_daily_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_temperature_daily_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_precipitation_daily_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_fire_weather_Europe.npy",
        # "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_heatwave_Europe.npy",
        # "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_drought_Europe.npy",
        # "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_extreme_wind_Europe.npy",
        # "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_volcano_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_HWSD_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_pga_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_accuflux_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_coastlines_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_rivers_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_slope_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_strahler_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_GLIM_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_GEM_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_aspect_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_elevation_Europe.npy",
        "/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_curvature_Europe.npy",
    ]
    for npy_file in npy_files:
        # Get the file name without the path
        npy_name = npy_file.split("/")[-1].split(".")[0]
        # Define the output file path
        output_file_path = f"/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/normalized_{npy_name}.npy"
        
        # Normalize the .npy file
        normalize_npy(npy_file, output_file_path)