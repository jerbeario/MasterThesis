
import numpy as np
import os
from multiprocessing import Pool, Manager

def generate_mask(data, sample_percentage=0.01):
    flat_data = data.flatten()
    non_nan_indices = np.where(~np.isnan(flat_data))[0]
    num_samples = int(len(non_nan_indices) * sample_percentage)
    sampled_indices = np.random.choice(non_nan_indices, num_samples, replace=False)
    mask = np.full(flat_data.shape, np.nan)
    mask[sampled_indices] = flat_data[sampled_indices]
    mask = mask.reshape(data.shape)
    return mask, sampled_indices

def apply_mask(data, mask, sampled_indices):
    masked_data = np.where(np.isnan(mask), np.nan, data)
    downsampled_data = masked_data.flatten()[sampled_indices]
    return downsampled_data

def process_file(args):
    path, mask, sampled_indices = args
    try:
        data = np.load(path)
        downsampled = apply_mask(data, mask, sampled_indices)

        output_dir = path.replace("npy_arrays", "downsampled_arrays")
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        output_path = output_dir.replace(".npy", "_downsampled.npy")

        np.save(output_path, downsampled)
        print(f"✅ Saved downsampled file to: {output_path}")
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")

if __name__ == "__main__":
    file_paths = [
        '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_accuflux_Europe.npy',
        '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_aspect_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_coastlines_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_curvature_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_drought_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_elevation_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_extreme_wind_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_fire_weather_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_GEM_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_GLIM_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_heatwave_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_HWSD_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_landcover_Europe_flat.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_multi_hazard_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_NDVI_Europe_flat.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_pga_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_precipitation_daily_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_rivers_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_slope_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_soil_moisture_root_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_soil_moisture_surface_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_strahler_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_temperature_daily_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_volcano_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wildfire_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wind_direction_daily_Europe.npy',
        # '/projects/FWC2/MYRIAD/Susceptibility/Input/Europe/npy_arrays/masked_wind_speed_daily_Europe.npy',
    ]   

    # Generate the shared mask from the first file

    reference_data = np.load(file_paths[0])

    mask, sampled_indices = generate_mask(reference_data)
    print(f"Generated mask with {len(sampled_indices)} samples")
    # Prepare arguments for multiprocessing
    args_list = [(path, mask, sampled_indices) for path in file_paths]
    print(args_list)

    with Pool(processes=10) as pool:
        pool.map(process_file, args_list)
