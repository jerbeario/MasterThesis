import logging
import logging.handlers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sympy import primerange
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Layer
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.initializers import he_normal
import time
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback  # COMMENT OUT WITH OLD VERSION
import warnings
warnings.filterwarnings("ignore")
# tf.get_logger().setLevel("ERROR")

if os.environ.get('CONDA_DEFAULT_ENV') == 'hydrology':
    import shap
else:
    tf.keras.mixed_precision.set_global_policy('float32')  # COMMENT OUT WITH OLD VERSION
    tf.keras.backend.set_floatx('float32')

# sys.exit(0)

class ModelMgr:
    def __init__(self, test='japan', prep='model', hazard='Landslide', hyper=False, model_choice='base', partition='spatial'):
        self.hazard = hazard
        self.name_model = 'susceptibility'
        self.missing_data_value = 0
        self.sample_ratio = 0.8
        self.test_split = 0.15
        self.neighborhood_size = 5
        self.hyper = hyper
        self.test = test
        self.model_choice = model_choice
        self.partition = partition
        if self.hazard == 'Landslide':
            self.variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation', 'accuflux', 'HWSD', 'road', 'GEM', 'curvature', 'GLIM']
            self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'continuous', 'continuous', 'categorical', 'label', 'continuous', 'continuous', 'categorical']
            # self.variables = ['elevation', 'slope', 'landcover', 'NDVI', 'precipitation', 'HWSD']
            # self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'categorical']
        elif self.hazard == 'Flood':
            self.variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation', 'accuflux']
            self.var_types = ['continuous', 'continuous', 'categorical', 'continuous', 'continuous', 'continuous', 'continuous']
        elif self.hazard == 'Tsunami':
            self.variables = ['elevation', 'coastlines', 'GEM']
            self.var_types = ['continuous', 'continuous', 'continuous']
            # self.variables = ['coastlines']
            # self.var_types = ['continuous']
        elif self.hazard == 'Multihazard':
            # self.variables = ['drought', 'extreme_wind', 'fire_weather', 'heatwave', 'pga', 'volcano', 'Flood_base_model', 'Landslide_base_model', 'Tsunami_base_model']
            self.variables = ['drought', 'extreme_wind', 'fire_weather', 'heatwave', 'jshis', 'volcano', 'Flood_base_model', 'Landslide_base_model', 'Tsunami_base_model']
            self.var_types = ['continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous', 'continuous']
        self.prep = prep
        self.ensemble_nr = 5  # 5
        self.seed = 43

        self.logger, self.ch = self.set_logger()

        physical_devices = tf.config.list_physical_devices('GPU')
        self.logger.info(f"Num GPUs Available: {len(physical_devices)}")
        self.logger.info(f"GPU Devices: {physical_devices}")
        self.logger.info(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")

        # # Configure memory growth for both GPUs to avoid memory errors
        # for gpu in physical_devices:
        #     tf.config.experimental.set_memory_growth(gpu, True)

        self.logger.info(f"TensorFlow version: {tf.__version__}")
        self.logger.info(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

        # Test simple GPU operation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            self.logger.info(c)
        # sys.exit(0)

        self.base_model_instance = BaseModel(self)
        self.ensemble_model_instance = EnsembleModel(self)
        self.meta_model_instance = MetaModel(self)

        if not (self.hyper and self.partition == 'random'):
            self.preprocess()
        # self.preprocess()

    def set_logger(self, verbose=True):
        """
        Set-up the logging system, exit if this fails
        """
        # assign logger file name and output directory
        datelog = time.ctime()
        datelog = datelog.replace(':', '_')
        reference = f'CNN_ls_susc_{self.test}'

        logfilename = ('logger' + os.sep + reference + '_logfile_' + 
                    str(datelog.replace(' ', '_')) + '.log')

        # create output directory if not exists
        if not os.path.exists('logger'):
            os.makedirs('logger')

        # create logger and set threshold level, report error if fails
        try:
            logger = logging.getLogger(reference)
            logger.setLevel(logging.DEBUG)
        except IOError:
            sys.exit('IOERROR: Failed to initialize logger with: ' + logfilename)

        # set formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s -'
                                    '%(levelname)s - %(message)s')

        # assign logging handler to report to .log file
        ch = logging.handlers.RotatingFileHandler(logfilename,
                                                maxBytes=10*1024*1024,
                                                backupCount=5)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # assign logging handler to report to terminal
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logger.addHandler(console)

        # start up log message
        logger.info('File logging to ' + logfilename)

        return logger, ch

    def preprocess(self):
        # Load data from .npy files
        # Prepare data for the CNN
        input_data = []
        spatial_split = False
        if self.prep == 'model':
            for var, var_type in zip(self.variables, self.var_types):
                input_data.append(self.load_normalize(var, var_type=var_type))
        elif self.prep == 'stack':
            for i in range(self.ensemble_nr):
                input_data.append(self.load_normalize(f'model_{i}', var_type='label', crop=False)[0])
        elif self.prep == 'multi':
            for var, var_type in zip(self.variables, self.var_types):
                input_data.append(self.load_normalize(var, var_type=var_type))
        elevation = self.load_normalize('elevation', var_type='mask')
        input_data = np.array(input_data)

        if self.hazard == 'Landslide':
            labels, output_shape, spatial_split = self.load_normalize('ldm', var_type='label')
        elif self.hazard == 'Flood':
            labels, output_shape, spatial_split = self.load_normalize('flood_surge', var_type='label')
        elif self.hazard == 'Tsunami':
            labels, output_shape, spatial_split = self.load_normalize('tsunami', var_type='label')
        elif self.hazard == 'Multihazard':
            labels, output_shape = self.load_normalize('multi_hazard', var_type='continuous')

        # List to store the indices
        self.logger.info('Extracting indices')
        indices_with_values = []
        original_shape = labels.shape
        self.logger.info(f"Input shape: {input_data.shape}")
        self.logger.info(f"Label shape: {labels.shape}")
        if spatial_split is not False:
            self.logger.info(f"Spatial shape: {spatial_split.shape}")
        self.logger.info(f"Elevation shape: {elevation.shape}")

        # Iterate over the array   ############## THIS SHOULD BE DONE IN LOAD NORMALIZE
        for idx, data_map in enumerate(elevation):
            if np.any(data_map > -9999): ###### SO WOULD NOT BE BETTER TO CHECK ALL MAPS AND MAKE NODATA=0???? FOR MIN MAX SCALER***
                indices_with_values.append(idx)

        # Extract data based on the indices
        input_data = input_data[:, indices_with_values]
        labels = labels[indices_with_values]
        if spatial_split is not False:
            spatial_split = spatial_split[indices_with_values]

        self.logger.info(f"Min value INPUT: {np.min(input_data)}")
        self.logger.info(f"Max value INPUT: {np.max(input_data)}")
        self.logger.info(f"Min value LABEL: {np.min(labels)}")
        self.logger.info(f"Max value LABEL: {np.max(labels)}")
        self.logger.info(f"Input shape: {input_data.shape}")
        self.logger.info(f"Label shape: {labels.shape}")
        if spatial_split is not False:
            self.logger.info(f"Spatial shape: {spatial_split.shape}")
        # for i in range(len(input_data)):
        #     variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation', 'accuflux', 'HWSD', 'road', 'GEM', 'curvature', 'GLIM']
        #     self.logger.info(f"Variable: {variables[i]}")
        #     self.logger.info(f"Min value INPUT: {np.min(input_data[i])}")
        #     self.logger.info(f"Max value INPUT: {np.max(input_data[i])}")
        # sys.exit(0)

        if self.partition == 'random':
            # Generate random indices from the first axis
            if not os.path.exists(f'Output/{region}/{self.hazard}/{self.hazard}_Susceptibility_{model_choice}_model_rnd_ind_{self.test}.npy') or self.hyper:
                train_indices = random.sample(range(input_data.shape[1]), int(input_data.shape[1] * self.sample_ratio))
                train_indices = np.save(f'Output/{region}/{self.hazard}/{self.hazard}_Susceptibility_{model_choice}_model_rnd_ind_{self.test}.npy', train_indices)

            train_indices = np.load(f'Output/{region}/{self.hazard}/{self.hazard}_Susceptibility_{model_choice}_model_rnd_ind_{self.test}.npy')

            # Create the test set of indices
            all_indices = set(range(input_data.shape[1]))
            complement_indices = list(all_indices - set(train_indices))

            test_indices = random.sample(complement_indices, int(input_data.shape[1] * self.test_split))
        elif self.partition == 'spatial':
            if not os.path.exists(f'Output/{region}/Susceptibility_spatial_partitioning_train.npy'):
                self.logger.info('INDICES')
                train_indices = np.where(spatial_split == 1)[0]
                self.logger.info(train_indices.shape)
                val_indices = np.where(spatial_split == 2)[0]
                self.logger.info(val_indices.shape)
                test_indices = np.where(spatial_split == 3)[0]
                self.logger.info(test_indices.shape)
                other_indices = np.where(spatial_split == 0)[0]
                self.logger.info(other_indices.shape)
                
                train_indices = np.save(f'Output/{region}/Susceptibility_spatial_partitioning_train.npy', train_indices)
                val_indices = np.save(f'Output/{region}/Susceptibility_spatial_partitioning_val.npy', val_indices)
                test_indices = np.save(f'Output/{region}/Susceptibility_spatial_partitioning_test.npy', test_indices)
            
            train_indices = np.load(f'Output/{region}/Susceptibility_spatial_partitioning_train.npy')
            val_indices = np.load(f'Output/{region}/Susceptibility_spatial_partitioning_val.npy')
            test_indices = np.load(f'Output/{region}/Susceptibility_spatial_partitioning_test.npy')

        self.input_data = input_data
        self.labels = labels

        # Store the selected indices in a new array
        model_inputs = input_data[:, train_indices]
        model_labels = labels[train_indices]

        test_data = input_data[:, test_indices]
        test_labels = labels[test_indices]

        self.train_indices = train_indices
        self.model_inputs = model_inputs
        self.input_data = input_data
        self.model_labels = model_labels
        self.labels = labels
        self.indices_with_values = indices_with_values
        self.original_shape = original_shape
        self.output_shape = output_shape
        self.test_data = test_data
        self.test_labels = test_labels

        if self.partition == 'spatial':
            val_data = input_data[:, val_indices]
            val_labels = labels[val_indices]
            self.val_data = val_data
            self.val_labels = val_labels

    def load_normalize(self, var, var_type='continuous', crop=True):
        self.logger.info(f'Loading {var}')
        if var == 'landcover' or var == 'NDVI':
            feature_data = np.load(f'Input/Japan/npy_arrays/masked_{var}_japan_flat.npy').astype(np.float32)
        elif var == 'precipitation':
            feature_data = np.load(f'Input/Japan/npy_arrays/masked_{var}_daily_japan.npy').astype(np.float32)
        elif 'base_model' in var:
            feature_data = np.load(f'Output/{region}/{var[:-11]}/{self.test}_{var[:-11]}_Susceptibility_base_model.npy').astype(np.float32)
            crop = False
        elif 'model' in var:
            feature_data = np.load(f'Output/{region}/{self.hazard}/{self.test}_{self.hazard}_Susceptibility_ensemble_{var}.npy').astype(np.float32)
        else:
            feature_data = np.load(f'Input/Japan/npy_arrays/masked_{var}_japan.npy').astype(np.float32)
        
        if crop:
            if self.test == 'hokkaido':
                feature_data = feature_data[150:1700,3800:-200]
            elif self.test == 'sado':
                feature_data = feature_data[2755:2955,3525:3675]
        
        # factor_x, factor_y = int(feature_data.shape[0] / tile), int(feature_data.shape[1] / tile)
        output_shape = feature_data.shape
        
        # Initialize the scaler, fit and transform the data
        if var_type == 'continuous':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_feature = scaler.fit_transform(feature_data.reshape(-1, 1)).reshape(feature_data.shape)
            scaled_feature = np.nan_to_num(scaled_feature, nan=self.missing_data_value)
        
        elif var_type == 'categorical':
            feature_data = np.nan_to_num(feature_data, nan=0)
            # Initialize the OneHotEncoder
            encoder = LabelEncoder()
            # Fit and transform the landcover data
            scaled_feature = encoder.fit_transform(feature_data.reshape(-1, 1)).reshape(feature_data.shape)

        elif var_type == 'label':
            scaled_feature = np.nan_to_num(feature_data, nan=self.missing_data_value)  # Convert nan to a specific value
            partition_map = np.load('Region/Japan/japan_prefecture_partitions_with_buffer.npy')
            partition_map = partition_map[0:5500,2300:8800]
            self.test_prefectures = [2, 6, 16, 10, 18, 34, 43, 39]
            self.val_prefectures = [7, 17, 23, 26, 32, 37, 44]
            self.train_prefectures = [i for i in range(1, 48) if i not in self.test_prefectures and i not in self.val_prefectures]
            spatial_split = []
        
        elif var_type == 'mask':
            scaled_feature = feature_data

        # Iterate through the array to extract sub-arrays
        scaled_feature_reshape = []
        for i in range(self.neighborhood_size, scaled_feature.shape[0] - self.neighborhood_size):
            for j in range(self.neighborhood_size, scaled_feature.shape[1] - self.neighborhood_size):
                ####### HERE SHOULD BE THE CHECK WITH ELEVATION
                sub_array = scaled_feature[i - self.neighborhood_size: i + self.neighborhood_size + 1, j - self.neighborhood_size: j + self.neighborhood_size + 1]
                if (var_type == 'label' and var is not 'road') | (var == 'multi_hazard') | (self.prep == 'multi' and var_type is not 'mask'):
                    center_value = sub_array[self.neighborhood_size, self.neighborhood_size]
                    scaled_feature_reshape.append(center_value)
                    if var_type == 'label':
                        if partition_map[i,j] in self.train_prefectures:
                            spatial_split.append(1)
                        elif partition_map[i,j] in self.val_prefectures:
                            spatial_split.append(2)
                        elif partition_map[i,j] in self.test_prefectures:
                            spatial_split.append(3)
                        else:
                            spatial_split.append(0)
                    # if var == 'HWSD':
                    #     print('check')
                    #     sys.exit(0)
                else:
                    scaled_feature_reshape.append(sub_array)

        # Convert the list of arrays to a numpy array
        scaled_feature_reshape = np.array(scaled_feature_reshape).astype(np.float32)
        
        # scaled_feature_reshape = scaled_feature.reshape((factor_x * factor_y, int(scaled_feature.shape[0] / factor_x), int(scaled_feature.shape[1] / factor_y), 1))
        
        if (var_type == 'label' and var is not 'road'):
            return scaled_feature_reshape.reshape(-1, 1, 1), output_shape, np.array(spatial_split)
        elif var == 'multi_hazard':
            return scaled_feature_reshape.reshape(-1, 1, 1), output_shape
        else:
            return np.expand_dims(scaled_feature_reshape, axis=-1)

    def train_base_model(self):
        if self.prep != 'stack':
            if self.hyper:
                self.base_model_instance.HypParOpt()
            else:
                self.base_model_instance.run()
                self.base_model = self.base_model_instance.base_model
        else:
            self.logger.info('Only works when prep!=stack')
        
    def xload_base_model(self):
        if self.prep != 'stack':
            self.base_model = keras.models.load_model(os.path.join(f'Output/{region}', self.hazard, f'base_model_{self.test}.tf'))
        else:
            self.logger.info('Only works when prep!=stack')
    
    def train_ensemble_model(self):
        if self.prep != 'stack':
            if self.hyper:
                self.ensemble_model_instance.HypParOpt()
            else:
                self.ensemble_model_instance.run()
                self.combined_model = self.ensemble_model_instance.combined_model
        else:
            self.logger.info('Only works when prep!=stack')

    def train_meta_model(self):
        if (self.prep == 'stack') | (self.prep == 'multi'):
            if self.hyper:
                self.meta_model_instance.HypParOpt()
            else:
                self.meta_model_instance.run()
                self.meta_model = self.meta_model_instance.meta_model
        else:
            self.logger.info('Only works when prep=stack | prep=multi')

    def load_meta_model(self):
        if self.prep != 'stack':
            self.meta_model = keras.models.load_model(os.path.join(f'Output/{region}', self.hazard, f'meta_model_MLP_{self.test}.tf'))
        else:
            self.logger.info('Only works when prep!=stack')

    def learning_to_stack(self):
        if self.prep == 'model':
            self.prep = 'stack'
            self.preprocess()
        else:
            self.logger.info('Only works when prep=model')

    def plot(self, data, name='scaled_feature'):
        fig = plt.figure()
        plt.imshow(data, cmap='viridis')
        plt.title(name)
        plt.colorbar()
        plt.savefig(f'Output/{region}/{self.hazard}/{name.replace(" ", "_")}.png', dpi=1000)
        return fig

    def plot_val_loss(self, history, name='scaled_feature'):
        # Visualize the training and validation loss
        fig = plt.figure()
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'Output/{region}/{self.hazard}/{name.replace(" ", "_")}.png', dpi=300)
        return fig

class BaseModel:
    def __init__(self, ModelMgr_instance):
        self.ModelMgr_instance = ModelMgr_instance
        self.name_model = 'basemodel'
        self.filters = 32
        self.n_layers = 3
        self.activation = 'relu'
        self.dropout = True
        self.drop_value = 0.41
        self.kernel_size = 3
        self.pool_size = 2 # CHANGED FROM 2, SHOULD BE TESTED, ADDED 10-03-2025
        self.learning_rate = 0.0001
        self.neighborhood_size = self.ModelMgr_instance.neighborhood_size
        self.hazard = self.ModelMgr_instance.hazard

        if self.ModelMgr_instance.hyper:
            self.batch_size = 8192
            self.epochs = 5
        else:
            self.batch_size = 8192 # 2048 256
            self.epochs = 5

        if self.ModelMgr_instance.test != 'sado':
            if os.path.exists(os.path.join(f'Output/{region}', self.hazard, f'Sweep_results_BaseModel_{self.ModelMgr_instance.test}.csv')):
                df = pd.read_csv(os.path.join(f'Output/{region}', self.hazard, f'Sweep_results_BaseModel_{self.ModelMgr_instance.test}.csv'))
                row = df.sort_values(by="val_loss", ascending=True).iloc[0]  # "val_loss"
                self.filters = int(row['filters'])
                self.n_layers = int(row['layers'])
                self.drop_value = np.round(row['dropout'], 3)
                self.learning_rate = np.round(row['lr'], 5)
        self.ModelMgr_instance.logger.info(f"fi:{self.filters} ly:{self.n_layers} dv:{self.drop_value} lr:{self.learning_rate}")
    
    def design_basemodel(self):
        def safe_binary_crossentropy(y_true, y_pred):
            # Handle potential NaN inputs
            y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
            y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
            # Clip predictions to avoid numerical instability
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        def safe_mse(y_true, y_pred):
            # Handle potential NaN inputs explicitly
            y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
            y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
            # Clip predictions to avoid numerical instability
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            return tf.reduce_mean(tf.square(y_pred - y_true))
            # return tf.keras.metrics.mean_squared_error(y_true, y_pred)  # OLD VERSION
        def safe_mae(y_true, y_pred):
            # Handle potential NaN inputs explicitly
            y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
            y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
            # Clip predictions to avoid numerical instability
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            # Debugging: print if NaNs appear
            tf.debugging.check_numerics(y_pred, "NaN in predictions")
            tf.debugging.check_numerics(y_true, "NaN in true values")
            return tf.reduce_mean(tf.abs(y_pred - y_true))
            # return tf.keras.metrics.mean_absolute_error(y_true, y_pred)  # OLD VERSION

        # Define the model architecture
        self.base_model = self.CNN()

        # Compile the model
        self.ModelMgr_instance.logger.info('Compiling model')
        # optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0, epsilon=1e-7)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, clipnorm=1.0, epsilon=1e-8) # , clipvalue=1.0
        # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9, nesterov=True, clipnorm=1.0, clipvalue=0.5)
        self.base_model.compile(optimizer=optimizer, loss=safe_binary_crossentropy, metrics=[safe_mse, safe_mae]) 

        # Provide model summary
        self.base_model.summary()

    def CNN(self):
        def safe_sigmoid(x):
            return tf.keras.activations.sigmoid(tf.clip_by_value(x, -15.0, 15.0))
        self.ModelMgr_instance.logger.info('Building architecture')
        # Define the model architecture
        input_shape = (self.ModelMgr_instance.input_data[0].shape[1], self.ModelMgr_instance.input_data[0].shape[2], 1)  # Define the input shape for the Conv2D layer

        merge_list, input_list = [], []  # Initialize lists for merging and input data
        for i, var in enumerate(self.ModelMgr_instance.variables):  # Iterate through the variables
            cnn_input = keras.Input(shape=input_shape, name=f'input_{i+1}')  # Define the input layer

            # Spatial attention after for resubmission.
            cnn1 = layers.Conv2D(self.filters, kernel_size=(self.kernel_size, self.kernel_size), padding='same', activation=self.activation, 
                                 kernel_initializer=he_normal(seed=42), kernel_regularizer=tf.keras.regularizers.l2(1e-4))(cnn_input)  # Apply Conv2D layer
            cnn1 = SpatialAttentionLayer()(cnn1)
            cnn1 = layers.MaxPooling2D(pool_size=(self.pool_size, self.pool_size), padding="same")(cnn1)
            
            for i in range(self.n_layers - 1):  # Iterate through additional convolutional layers
                cnn1 = layers.Conv2D(self.filters * 2, kernel_size=(self.kernel_size, self.kernel_size), padding='same', activation=self.activation,
                                     kernel_initializer=he_normal(seed=42), kernel_regularizer=tf.keras.regularizers.l2(1e-4))(cnn1)  # Apply Conv2D layer
                if i == 1 or i == 3 or i == self.n_layers - 1:
                    cnn1 = layers.MaxPooling2D(pool_size=(self.pool_size, self.pool_size), padding="same")(cnn1)  # Apply MaxPooling2D layer
                
            if len(self.ModelMgr_instance.variables) > 1:  # Check if there are multiple variables
                merge_list.append(cnn1)  # Append to the merge list
                input_list.append(cnn_input)  # Append to the input list
            else:
                merge_list = cnn1  # Set the merge list
                input_list = cnn_input  # Set the input list

        if len(self.ModelMgr_instance.variables) > 1:  # Check if there are multiple variables
            merge_list = layers.concatenate(merge_list)  # Concatenate the merge list
        # merge_list = layers.Flatten()(merge_list)
        merge_list = layers.GlobalAveragePooling2D()(merge_list) # instead of line flatten above 
        x = layers.Dense(1024, kernel_initializer=he_normal(seed=42), # 1024 128  activation=self.activation, 
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(merge_list)  # self.filters * 2 
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x) 
        if self.dropout:  # Check if dropout is enabled
            x = layers.Dropout(self.drop_value)(x)  # Apply Dropout layer
        outputs = layers.Dense(1, kernel_regularizer=regularizers.l2(0.00001), kernel_initializer=he_normal(seed=42), activation='sigmoid')(x)  # Add the output layer

        model = keras.Model(inputs=input_list, outputs=outputs, name=self.name_model)  # Define the model

        return model

    def train(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)  # , start_from_epoch=3
        # callbacks = [wandb.keras.WandbCallback(), early_stopping, TerminateOnNaN()]   # OLD VERSION
        callbacks=[WandbMetricsLogger(), early_stopping, TerminateOnNaN()]  # , WandbModelCheckpoint("models") WandbMetricsLogger() WandbCallback() CustomWandbCallback()

        # Train the model
        self.ModelMgr_instance.logger.info('Fitting model')
        if self.ModelMgr_instance.partition == 'random':
            history = self.base_model.fit({'input_' + str(i+1): self.ModelMgr_instance.model_inputs[i] for i in range(len(self.ModelMgr_instance.model_inputs))},
                                        self.ModelMgr_instance.model_labels, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                                        callbacks=[wandb.keras.WandbCallback(), early_stopping, TerminateOnNaN()])
        elif self.ModelMgr_instance.partition == 'spatial':
            history = self.base_model.fit({'input_' + str(i+1): self.ModelMgr_instance.model_inputs[i] for i in range(len(self.ModelMgr_instance.model_inputs))},
                                        self.ModelMgr_instance.model_labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                                        validation_data=({'input_' + str(i+1): self.ModelMgr_instance.val_data[i] for i in range(len(self.ModelMgr_instance.val_data))}, self.ModelMgr_instance.val_labels),
                                        callbacks=callbacks)  # , TerminateOnNaN()  GradientMonitorCallback

        fig = self.ModelMgr_instance.plot_val_loss(history, name=f'{self.ModelMgr_instance.test} {self.hazard} Loss base model')
        wandb.log({"Loss_val": wandb.Image(fig)})
        wandb.log({"bce_val": history.history['val_loss'][-1]})
        self.bce_val = history.history['val_loss'][-1]

        # if self.ModelMgr_instance.hyper:
        #     if self.bce_val < self.bce_val_best:

        for var in self.base_model.trainable_variables:
            tf.print(var.name, "max weight:", tf.reduce_max(var), "min weight:", tf.reduce_min(var), summarize=10)
        
    def predict(self):
        # Evaluate the model on the validation data
        self.ModelMgr_instance.logger.info('Predicting')
        susceptibility = self.base_model.predict({'input_' + str(i+1): self.ModelMgr_instance.input_data[i] for i in range(len(self.ModelMgr_instance.input_data))})
       
        self.ModelMgr_instance.logger.info('Reshaping output')
        susc_map = np.zeros(self.ModelMgr_instance.original_shape)
        # Fill the new_data_array with the values at the specified indices
        for i, index in enumerate(self.ModelMgr_instance.indices_with_values):
            susc_map[index, :, :] = susceptibility[i]

        susc_shape = (self.ModelMgr_instance.output_shape[0] - self.neighborhood_size*2, self.ModelMgr_instance.output_shape[1] - self.neighborhood_size*2)
        susc_map = susc_map.reshape(susc_shape)

        # Create a new array with shape of output and fill it with zeros
        susc_map_reshape = np.zeros(self.ModelMgr_instance.output_shape)

        # Copy the original array into the center of the new array
        susc_map_reshape[self.neighborhood_size:self.neighborhood_size+susc_shape[0], self.neighborhood_size:self.neighborhood_size+susc_shape[1]] = susc_map

        # Plot the susceptibility map
        self.ModelMgr_instance.logger.info('plotting and saving')
        fig = self.ModelMgr_instance.plot(susc_map_reshape, name=f'{self.ModelMgr_instance.test} {self.hazard} Susceptibility base model')
        np.save(f'Output/{region}/{self.hazard}/{self.ModelMgr_instance.test}_{self.hazard}_Susceptibility_base_model.npy', susc_map_reshape)
        np.save(f'Output/{region}/{self.hazard}/{self.hazard}_Susceptibility_base_model_rnd_ind_{self.ModelMgr_instance.test}.npy', self.ModelMgr_instance.train_indices)

        # Estimate feature importance
        if not self.ModelMgr_instance.hyper:
            # Compute the minimum and maximum values of the array
            self.min_value = np.min(susceptibility)
            self.max_value = np.max(susceptibility)

            # Perform min-max scaling to scale the array between 0 and 1
            self.susceptibility = (susceptibility - self.min_value) / (self.max_value - self.min_value)
            self.baseline_accuracy = accuracy_score(np.squeeze(self.ModelMgr_instance.labels, axis=2), (self.susceptibility > 0.5).astype(int))
            # self.permutation_feature_importance()

        # Save the base model weights
        # self.base_model.save(os.path.join(f'Output/{region}', self.hazard, f'base_model_{self.ModelMgr_instance.test}.tf'), save_format='tf') # OLD VERSION
        self.base_model.save(os.path.join(f'Output/{region}', self.hazard, f'base_model_{self.ModelMgr_instance.test}.keras'))
        wandb.log({"Susceptibility basemodel": wandb.Image(fig)})

    def permutation_feature_importance(self):
        self.ModelMgr_instance.logger.info('Feature Importance')
        # Compute permutation feature importance for each input feature set
        num_features = len(self.ModelMgr_instance.input_data)
        feature_importances = {}
        feature_importances['Baseline'] = self.baseline_accuracy

        for feature_index in range(num_features):
            # Copy the original input data for the selected feature set
            shuffled_input_data = self.ModelMgr_instance.input_data.copy()

            # Shuffle the feature values (permutation)
            shuffled_input_data[feature_index] = np.random.shuffle(shuffled_input_data[feature_index])

            # Compute model predictions with shuffled feature values
            predictions = self.base_model.predict({'input_' + str(i+1): shuffled_input_data[i] for i in range(len(shuffled_input_data))})
            predictions = (predictions - self.min_value) / (self.max_value - self.min_value)
            shuffled_accuracy = accuracy_score(np.squeeze(self.ModelMgr_instance.labels, axis=2), (predictions > 0.5).astype(int))

            # Compute permutation feature importance
            permutation_importance = self.baseline_accuracy - shuffled_accuracy
            feature_importances[self.ModelMgr_instance.variables[feature_index]] = permutation_importance
        
        # Save to dataframe
        df = pd.DataFrame.from_dict(feature_importances, orient='index')
        df.to_excel(f"Output/{region}/{self.hazard}/Permutation_Importance_{self.ModelMgr_instance.test}.xlsx")

    def testing(self):
        # Evaluate the model on the validation data
        self.ModelMgr_instance.logger.info('Testing')

        y_pred = self.base_model.predict({'input_' + str(i+1): self.ModelMgr_instance.test_data[i] for i in range(len(self.ModelMgr_instance.test_data))})
        y_pred = np.squeeze(y_pred, axis=(1))
        y_true = np.squeeze(self.ModelMgr_instance.test_labels, axis=(1,2))

        # Calculate Binary Cross-Entropy
        y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true_tf, y_pred_tf)
        self.bce_test = tf.reduce_mean(bce).numpy()
        self.ModelMgr_instance.logger.info(f"BCE: {self.bce_test}")

        # Metrics
        self.mae = mean_absolute_error(y_true, y_pred)
        self.mse = mean_squared_error(y_true, y_pred)
        self.ModelMgr_instance.logger.info(f"MAE: {self.mae}")
        self.ModelMgr_instance.logger.info(f"MSE: {self.mse}")

        # Create a dictionary to store values with names
        metrics_dict = {'MAE': [self.mae], 'MSE': [self.mse]}
        df = pd.DataFrame(metrics_dict)

        # Write the values to a text file
        df.to_csv(f'Output/{region}/{self.hazard}/config_{self.ModelMgr_instance.test}_basemodel.csv', index=False)

        # Store in W&B
        wandb.log({"MAE_test": self.mae})
        wandb.log({"MSE_test": self.mse})
        wandb.log({"BCE_test": self.bce_test})

    def HypParOpt(self):
        # Set seed
        np.random.seed(self.ModelMgr_instance.seed)
        tf.random.set_seed(self.ModelMgr_instance.seed)
        self.bce_val_best = 1000

        # Define sweep config
        sweep_configuration = {
            "method": "bayes",
            "name": f"{self.ModelMgr_instance.test}_BaseModel",
            "metric": {"goal": "minimize", "name": "bce_val"}, # "val_loss"
            "parameters": {
                "layers": {"values": [3, 4, 5]},
                "filters": {"values": [32, 64, 96, 128]},
                "lr": {"max": 0.001, "min": 0.00001},
                "dropout": {"max": 0.5, "min": 0.1},
            },
        }

        # Initialize sweep by passing in config.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"{self.hazard}-sweep")
        self.hyper_df = pd.DataFrame(columns=["layers", "filters", "lr", "dropout", "MAE", "MSE", "val_loss"], dtype=float)

        # Start sweep job.
        wandb.agent(sweep_id, function=self.main, count=20)

    def run(self):
        # Set seed
        np.random.seed(self.ModelMgr_instance.seed)
        tf.random.set_seed(self.ModelMgr_instance.seed)

        wandb.init(project=self.hazard, entity="timothy-tiggeloven", group=self.ModelMgr_instance.test, job_type="BaseModel",
                config={"learning_rate": self.learning_rate,
                        "epochs": self.epochs,
                        "filters": self.filters,
                        "layers": self.n_layers,
                        "seed": self.ModelMgr_instance.seed,
                        "dropout": self.drop_value,
                        "NN_cells": self.neighborhood_size,
                        "variables": self.ModelMgr_instance.variables,
                        "sample_ratio": self.ModelMgr_instance.sample_ratio})

        # Develop a basemodel
        self.main()

        # Store Weights and Biases
        wandb.finish()
    
    def main(self):
        if self.ModelMgr_instance.hyper:
            self.base_model = False
            if self.ModelMgr_instance.partition == 'random':
                self.ModelMgr_instance.preprocess() 
            wandb.init()
            self.n_layers = wandb.config.layers
            self.filters = wandb.config.filters
            self.learning_rate = wandb.config.lr
            self.drop_value = wandb.config.dropout

        # Develop a basemodel
        self.design_basemodel()
        self.train()
        self.predict()
        self.testing()

        if self.ModelMgr_instance.hyper:
            new_row = pd.DataFrame([{
                "layers": wandb.config.layers,
                "filters": wandb.config.filters,
                "lr": wandb.config.lr,
                "dropout": wandb.config.dropout,
                "val_loss": self.bce_val,
                "MAE": self.mae,
                "MSE": self.mse,
            }])
            self.hyper_df = pd.concat([self.hyper_df, new_row], ignore_index=True)
            self.hyper_df.to_csv(f"Output/{region}/{self.hazard}/Sweep_results_BaseModel_{self.ModelMgr_instance.test}.csv", index=False)
            if self.bce_val < self.bce_val_best:
                self.bce_val_best = self.bce_val
        self.ModelMgr_instance.logger.info(f"Main done")

class EnsembleModel:
    def __init__(self, ModelMgr_instance):
        self.ModelMgr_instance = ModelMgr_instance
        self.name_model = 'ensemblemodel'
        self.filters = 16
        self.n_layers = 2
        self.activation = 'relu'
        self.dropout = True
        self.drop_value = 0.2
        self.ensemble_nr = self.ModelMgr_instance.ensemble_nr
        self.kernel_size = (3,3)
        self.pool_size = (2, 2)
        self.epochs = 5
        self.batch_size = 10000
        self.sample_ratio = 0.8
        self.neighborhood_size = self.ModelMgr_instance.neighborhood_size
        self.hazard = self.ModelMgr_instance.hazard
        self.learning_rate = 0.001

    def transfer_learning(self):
        input_layer = keras.Input(shape=model_instance.base_model.layers[-5].output.shape[1:], name='input_ens')  # Define the input layer

        # Add additional layers or modify architecture for fine-tuning
        x = Conv2D(self.filters, self.kernel_size, kernel_regularizer=l2(0.01), padding='same', activation='relu')(input_layer) #(truncated_base_model.output)
        x = BatchNormalization()(x)

        # Residual blocks
        for i in range(self.n_layers): 
            x = ResidualBlock(self.filters, self.kernel_size)(x)
            x = MaxPooling2D(pool_size=self.pool_size, padding="same")(x)

        # Dense to output
        x = Flatten()(x)
        x = Dense(256, kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)

        # Combine the additional layers
        return models.Model(inputs=input_layer, outputs=output)

    def design_ensemblemodel(self):
        # Truncate the last two layers
        truncated_base_model = models.Model(inputs=model_instance.base_model.input, outputs=model_instance.base_model.layers[-5].output, name=f'truncate_{self.ens}')
        model_ens = self.transfer_learning()

        combined_output = model_ens(truncated_base_model.output)

        # Create a new model with the combined output
        self.combined_model = models.Model(inputs=truncated_base_model.input, outputs=combined_output, name=f'ensemble_{self.ens}')

        # Set all layers except the last one to trainable=False
        # for layer in self.combined_model.layers[:-1]:
        #     layer.trainable = False

        # Loop over all layers in the model
        # Create a custom optimizer with different learning rates for each layer
        # optimizers_and_layers = [(Adam(learning_rate=1e-4), self.combined_model.layers[:-2]), (Adam(learning_rate=1e-2), self.combined_model.layers[-1])]
        # custom_optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

        # Compile the model
        self.ModelMgr_instance.logger.info('Compiling Ensemble model')
        # self.combined_model.compile(optimizer=optimizers_and_layers, loss='binary_crossentropy', metrics=['accuracy'])
        self.combined_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['mse', 'mae'])

        # Manually set the learning rate for specific layers
        for layer in self.combined_model.layers[:-1]:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer.kernel_learning_rate = 0.0001

        # Provide model summary
        model_ens.summary()

    def train(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)  # , start_from_epoch=3

        # Train the model
        self.ModelMgr_instance.logger.info('Fitting model')
        history = self.combined_model.fit(self.ens_model_inputs, self.ens_model_labels, validation_split=0.2, epochs=self.epochs,
                                          batch_size=self.batch_size, shuffle=True, callbacks=[wandb.keras.WandbCallback(), early_stopping])
        
        fig = self.ModelMgr_instance.plot_val_loss(history, name=f'{self.ModelMgr_instance.test} {self.hazard} Loss base model')
        wandb.log({"Loss_val": wandb.Image(fig)})
        wandb.log({"bce_val": history.history['val_loss'][-1]})
        self.bce_val = history.history['val_loss'][-1]

    def predict(self):
        # Evaluate the model on the validation data
        self.ModelMgr_instance.logger.info('Predicting')
        # input_data = {'input_' + str(i+1): self.ModelMgr_instance.input_data[i] for i in range(len(self.ModelMgr_instance.input_data))}
        susceptibility = self.combined_model.predict({'input_' + str(i+1): self.ModelMgr_instance.input_data[i] for i in range(len(self.ModelMgr_instance.input_data))})

        self.ModelMgr_instance.logger.info('Reshaping output')
        susc_map = np.zeros(self.ModelMgr_instance.original_shape)
        # Fill the new_data_array with the values at the specified indices
        for i, index in enumerate(self.ModelMgr_instance.indices_with_values):
            susc_map[index, :, :] = susceptibility[i]

        susc_shape = (self.ModelMgr_instance.output_shape[0] - self.neighborhood_size*2, self.ModelMgr_instance.output_shape[1] - self.neighborhood_size*2)
        susc_map = susc_map.reshape(susc_shape)

        # Create a new array with shape of output and fill it with zeros
        susc_map_reshape = np.zeros(self.ModelMgr_instance.output_shape)

        # Copy the original array into the center of the new array
        susc_map_reshape[self.neighborhood_size:self.neighborhood_size+susc_shape[0], self.neighborhood_size:self.neighborhood_size+susc_shape[1]] = susc_map

        # sys.exit(0)
        # susc_map_reshape = susc_map.reshape((int(susc_map.shape[1] * factor_x), int(susc_map.shape[2] * factor_y)))

        # # Plot the susceptibility map
        self.ModelMgr_instance.logger.info('plotting and saving')
        fig = self.ModelMgr_instance.plot(susc_map_reshape, name=f'{self.ModelMgr_instance.test} {self.hazard} Susceptibility ensemble model {self.ens}')
        np.save(f'Output/{region}/{self.hazard}/{self.ModelMgr_instance.test}_{self.hazard}_Susceptibility_ensemble_model_{self.ens}.npy', susc_map_reshape)

        # Save the base model weights
        self.combined_model.save(os.path.join(f'Output/{region}', self.hazard, f'{self.ModelMgr_instance.test}_ensemble_model_{self.ens}.tf'), save_format='tf')
        wandb.log({f"Susceptibility ensemble {self.ens}": wandb.Image(fig)})
    
    def testing(self):
        # Evaluate the model on the validation data
        self.ModelMgr_instance.logger.info('Testing')

        y_pred = self.combined_model.predict({'input_' + str(i+1): self.ModelMgr_instance.test_data[i] for i in range(len(self.ModelMgr_instance.test_data))})
        y_pred = np.squeeze(y_pred, axis=(1))
        y_true = np.squeeze(self.ModelMgr_instance.test_labels, axis=(1,2))

        # Calculate Binary Cross-Entropy
        y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true_tf, y_pred_tf)
        self.bce_test = tf.reduce_mean(bce).numpy()
        self.ModelMgr_instance.logger.info(f"BCE: {self.bce_test}")

        # Metrics
        self.mae = mean_absolute_error(y_true, y_pred)
        self.mse = mean_squared_error(y_true, y_pred)
        self.ModelMgr_instance.logger.info(f"MAE: {self.mae}")
        self.ModelMgr_instance.logger.info(f"MSE: {self.mse}")

        # Create a dictionary to store values with names
        metrics_dict = {'MAE': [self.mae], 'MSE': [self.mse]}
        df = pd.DataFrame(metrics_dict)

        # Write the values to a text file
        df.to_csv(f'Output/{region}/{self.hazard}/config_{self.ModelMgr_instance.test}_ensemble_{self.ens}.csv', index=False)

        # Store in W&B
        wandb.log({"MAE_test": self.mae})
        wandb.log({"MSE_test": self.mse})
        wandb.log({"BCE_test": self.bce_test})

    def HypParOpt(self):
        primes = np.array(list(primerange(20, 1000)))
        seeds = primes[np.arange(0, len(primes), 5)[:model_instance.ensemble_nr]]

        self.ModelMgr_instance.preprocess()
                                       
        for self.ens in range(self.ensemble_nr):
            np.random.seed(seeds[self.ens])
            tf.random.set_seed(seeds[self.ens])

            # Define sweep config
            sweep_configuration = {
                "method": "bayes",
                "name": f"{self.ModelMgr_instance.test}_EnsembleModel_{self.ens}",
                "metric": {"goal": "minimize", "name": "val_loss"},
                "parameters": {
                    "layers": {"values": [1, 2, 3, 4]},
                    "filters": {"values": [32, 64, 128, 264]},
                    "lr": {"max": 0.1, "min": 0.0001},
                    "dropout": {"max": 0.5, "min": 0.0001},
                },
            }

            # Initialize sweep by passing in config.
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"{self.hazard}-sweep")
            self.hyper_df = pd.DataFrame(columns=["layers", "filters", "lr", "dropout", "val_loss"], dtype=float)

            # Start sweep job.
            wandb.agent(sweep_id, function=self.main, count=10)

    def run(self):
        primes = np.array(list(primerange(20, 1000)))
        seeds = primes[np.arange(0, len(primes), 5)[:model_instance.ensemble_nr]]
                                       
        for self.ens in range(self.ensemble_nr):
            np.random.seed(seeds[self.ens])
            tf.random.set_seed(seeds[self.ens])

            if os.path.exists(os.path.join(f'Output/{region}', self.hazard, f'Sweep_results_EnsembleModel_{self.ens}_{self.ModelMgr_instance.test}.csv')):
                df = pd.read_csv(os.path.join(f'Output/{region}', self.hazard, f'Sweep_results_EnsembleModel_{self.ens}_{self.ModelMgr_instance.test}.csv'))
                row = df.sort_values(by="val_loss", ascending=True).iloc[0]  # "val_loss"
                self.filters = int(row['filters'])
                self.n_layers = int(row['layers'])
                self.drop_value = np.round(row['dropout'], 4)
                self.learning_rate = np.round(row['lr'], 4)

            wandb.init(project=self.hazard, entity="timothy-tiggeloven", group=self.ModelMgr_instance.test, job_type=f"EnsembleModel_{self.ens}",
                   config={"learning_rate": 0.001,
                           "epochs": self.epochs,
                           "filters": self.filters,
                           "layers": self.n_layers,
                           "seed": seeds[self.ens],
                           "dropout": self.drop_value,
                           "NN_cells": self.neighborhood_size,
                           "variables": self.ModelMgr_instance.variables,
                           "sample_ratio": self.sample_ratio})
            
            # Develop a ensemblemodel
            self.main()
            wandb.finish()

    def main(self):
        if self.ModelMgr_instance.hyper:
            self.base_model = False
            wandb.init()
            self.n_layers = wandb.config.layers
            self.filters = wandb.config.filters
            self.learning_rate = wandb.config.lr
            self.drop_value = wandb.config.dropout
            random_indices = random.sample(range(self.ModelMgr_instance.input_data.shape[1]), int(self.ModelMgr_instance.input_data.shape[1] * self.sample_ratio))

            # Store the selected indices in a new array 
            start = int(self.sample_ratio * len(random_indices) * self.ens)
            end = int(self.sample_ratio * len(random_indices) * self.ens + self.sample_ratio * len(random_indices))
            self.ens_model_inputs = self.ModelMgr_instance.input_data[:, start:end, :, :, :]
            self.ens_model_labels = self.ModelMgr_instance.labels[start:end]
        else:
            # Generate random indices from the first axis
            random_indices = np.load(f'Output/{region}/{self.hazard}/{self.hazard}_Susceptibility_base_model_rnd_ind_{self.ModelMgr_instance.test}.npy')
        
            # Store the selected indices in a new array 
            start = int(self.sample_ratio * len(random_indices) * self.ens)
            end = int(self.sample_ratio * len(random_indices) * self.ens + self.sample_ratio * len(random_indices))
            self.ens_model_inputs = self.ModelMgr_instance.model_inputs[:, start:end, :, :, :]
            self.ens_model_labels = self.ModelMgr_instance.model_labels[start:end]

        # Assign input names
        self.ens_model_inputs = {'input_' + str(i+1): self.ens_model_inputs[i] for i in range(len(self.ens_model_inputs))}

        # THIS IS ONLY FOR SADO AT THE MOMENT TO TEST BECAUSE TOO SMALL DATASET ANYWAYS
        if self.ModelMgr_instance.test == 'sado':
            self.ens_model_inputs = {'input_' + str(i+1): self.ModelMgr_instance.model_inputs[i] for i in range(len(self.ModelMgr_instance.model_inputs))}
            self.ens_model_labels = self.ModelMgr_instance.model_labels

        # Develop a basemodel
        self.design_ensemblemodel()
        self.train()
        #self.predict()
        #self.testing()

        if self.ModelMgr_instance.hyper:
            new_row = pd.DataFrame([{
                "layers": wandb.config.layers,
                "filters": wandb.config.filters,
                "lr": wandb.config.lr,
                "dropout": wandb.config.dropout,
                "val_loss": self.bce_val,
                #"MAE": self.mae,
                #"MSE": self.mse,
            }])
            self.hyper_df = pd.concat([self.hyper_df, new_row], ignore_index=True)
            self.hyper_df.to_csv(f"Output/{region}/{self.hazard}/Sweep_results_EnsembleModel_{self.ens}_{self.ModelMgr_instance.test}.csv", index=False)
        self.ModelMgr_instance.logger.info(f"Main done")

# Add this as a custom callback
class NaNDetector(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            weights = layer.get_weights()
            for w in weights:
                if np.isnan(w).any():
                    print(f"NaN detected in weights of layer {layer.name}")
                    break
        
        if logs and any(np.isnan(v) for k, v in logs.items()):
            print(f"NaN detected in logs: {logs}")

class CustomWandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            wandb.log(logs)

class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        # Assuming channels_in is the last dimension
        channels_in = input_shape[-1]

        # Define layers in the build method
        self.conv1 = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding='same', name='Res_conv_1')
        self.batch_norm1 = BatchNormalization(name='Res_bn_1')
        self.activation1 = Activation('relu', name='Res_relu_1')
        self.conv2 = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding='same', name='Res_conv_2')
        self.batch_norm2 = BatchNormalization(name='Res_bn_2')
        self.activation2 = Activation('relu', name='Res_relu_2')
        super(ResidualBlock, self).build(input_shape)

    def call(self, x):
        shortcut = x
        y = self.conv1(shortcut)
        y = self.batch_norm1(y)
        y = self.activation1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        y = self.activation2(y)

        residual = Add()([shortcut, y])
        z = Activation('relu', name='Res_relu_extra')(residual)

        return z

    def compute_output_shape(self, input_shape):
        return input_shape

class SpatialAttentionLayer(Layer):
    def __init__(self):
        super(SpatialAttentionLayer, self).__init__()

    def build(self, input_shape):
        channels = input_shape[-1]
        self.conv1x1_theta = layers.Conv2D(channels, (1, 1), activation='relu', padding='same', 
                                           kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))
        self.conv1x1_phi = layers.Conv2D(channels, (1, 1), activation='relu', padding='same', 
                                         kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))
        self.conv1x1_g = layers.Conv2D(channels, (1, 1), activation='sigmoid', padding='same',
                                       kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0001))

    def call(self, x):
        theta = self.conv1x1_theta(x)
        phi = self.conv1x1_phi(x)
        g = self.conv1x1_g(x)

        theta_phi = layers.multiply([theta, phi])
        attention = layers.multiply([theta_phi, g])
        attended_x = layers.add([x, attention])

        return attended_x

class MetaModel:
    def __init__(self, ModelMgr_instance):
        self.ModelMgr_instance = ModelMgr_instance
        self.name_model = 'metamodel'
        self.neurons = 256
        self.n_layers = 2
        self.activation = 'relu'
        self.dropout = True
        self.drop_value = 0.2
        self.ensemble_nr = self.ModelMgr_instance.ensemble_nr
        self.kernel_size = (3,3)
        self.pool_size = (2, 2)
        self.epochs = 30  # 30
        self.sample_ratio = 0.1
        self.learning_rate = 0.001
        self.neighborhood_size = self.ModelMgr_instance.neighborhood_size
        self.hazard = self.ModelMgr_instance.hazard

        if os.path.exists(os.path.join(f'Output/{region}', self.hazard, f'Sweep_results_MetaModel_{self.ModelMgr_instance.test}.csv')):
            df = pd.read_csv(os.path.join(f'Output/{region}', self.hazard, f'Sweep_results_MetaModel_{self.ModelMgr_instance.test}.csv'))
            row = df.sort_values(by="val_loss", ascending=True).iloc[0]  # "val_loss"
            self.neurons = int(row['filters'])
            self.n_layers = int(row['layers'])
            self.drop_value = np.round(row['dropout'], 4)
            self.learning_rate = np.round(row['lr'], 4)
            # self.ModelMgr_instance.logger.info(f"fi:{self.filters} ly:{self.n_layers} dv:{self.drop_value} lr:{self.learning_rate}")

        if self.ModelMgr_instance.prep == 'multi':
            self.axistr = (2)
        else:
            self.axistr = (2, 3)
    
    def calc_weighted_avg(self):
        self.ModelMgr_instance.logger.info('Calculating Weighted Average Ensemble')
        
        # Load MAE and MSE values from CSV files
        dfs = [pd.read_csv(f"Output/{region}/{self.hazard}/config_{self.ModelMgr_instance.test}_ensemble_{ens}.csv") for ens in range(self.ModelMgr_instance.ensemble_nr)]

        # Calculate weights based on the inverse of MAE or MSE, and normalize
        weights_mae = 1 / np.array([df['MAE'].values for df in dfs])
        weights_mae_normalized = np.squeeze((weights_mae / np.sum(weights_mae)), axis=1)

        weights_mse = 1 / np.array([df['MSE'].values for df in dfs])
        weights_mse_normalized = np.squeeze((weights_mse / np.sum(weights_mse)), axis=1)

        # Load predicted maps from ensemble models
        ensemble_maps = np.array([np.load(f"Output/{region}/{self.hazard}/{self.ModelMgr_instance.test}_{self.hazard}_Susceptibility_ensemble_model_{ens}.npy") for ens in range(self.ModelMgr_instance.ensemble_nr)])

        # Perform weighted average using the calculated weights
        weighted_average_mae = np.average(ensemble_maps, axis=0, weights=weights_mae_normalized)
        weighted_average_mse = np.average(ensemble_maps, axis=0, weights=weights_mse_normalized)

        # Plot the susceptibility map
        self.ModelMgr_instance.logger.info('plotting and saving')
        fig = self.ModelMgr_instance.plot(weighted_average_mae, name=f'{self.ModelMgr_instance.test} {self.hazard} Susceptibility meta model MAE')
        fig = self.ModelMgr_instance.plot(weighted_average_mse, name=f'{self.ModelMgr_instance.test} {self.hazard} Susceptibility meta model MSE')
        np.save(f'Output/{region}/{self.hazard}/{self.ModelMgr_instance.test}_{self.hazard}_Susceptibility_meta_model_MAE.npy', weighted_average_mae)
        np.save(f'Output/{region}/{self.hazard}/{self.ModelMgr_instance.test}_{self.hazard}_Susceptibility_meta_model_MSE.npy', weighted_average_mse)

    def design_metamodel(self, model_type='logistic'):
        # input_layer = keras.Input(shape=self.ModelMgr_instance.input_data.shape[0], name=f'MetaModel_{model_type}')  # Define the input layer

        if model_type == 'logistic':
            self.meta_model = models.Sequential([
                Dense(1, activation='sigmoid', input_shape=(self.ModelMgr_instance.input_data.shape[0],))
            ], name=f'MetaModel_{model_type}')

        elif model_type == 'MLP':
            if self.n_layers == 1:
                # Define and compile a simple TensorFlow/Keras model
                self.meta_model = models.Sequential([
                    Dense(self.neurons, input_shape=(self.ModelMgr_instance.input_data.shape[0],),
                          kernel_initializer=he_normal(seed=42), kernel_regularizer=regularizers.l2(1e-4)),
                    BatchNormalization(),
                    Activation('relu'),
                    Dropout(self.drop_value),

                    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer=he_normal(seed=42))
                ], name=f'MetaModel_{model_type}')
            elif self.n_layers == 2:
                # Define and compile a simple TensorFlow/Keras model
                self.meta_model = models.Sequential([
                    Dense(self.neurons, input_shape=(self.ModelMgr_instance.input_data.shape[0],),
                          kernel_initializer=he_normal(seed=42), kernel_regularizer=regularizers.l2(1e-4)),
                    BatchNormalization(),
                    Activation('relu'),
                    Dropout(self.drop_value),

                    Dense(int(self.neurons / 2), kernel_initializer=he_normal(seed=42), kernel_regularizer=regularizers.l2(1e-4)),
                    BatchNormalization(),
                    Activation('relu'),
                    Dropout(self.drop_value),

                    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer=he_normal(seed=42))
                ], name=f'MetaModel_{model_type}')
            elif self.n_layers == 3:
                # Define and compile a simple TensorFlow/Keras model
                self.meta_model = models.Sequential([
                    Dense(self.neurons, input_shape=(self.ModelMgr_instance.input_data.shape[0],),
                          kernel_initializer=he_normal(seed=42), kernel_regularizer=regularizers.l2(1e-4)),
                    BatchNormalization(),
                    Activation('relu'),
                    Dropout(self.drop_value),

                    Dense(int(self.neurons / 2), kernel_initializer=he_normal(seed=42), kernel_regularizer=regularizers.l2(1e-4)),
                    BatchNormalization(),
                    Activation('relu'),
                    Dropout(self.drop_value),

                    Dense(int(self.neurons / 4), kernel_initializer=he_normal(seed=42), kernel_regularizer=regularizers.l2(1e-4)),
                    BatchNormalization(),
                    Activation('relu'),
                    Dropout(self.drop_value),

                    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.00001), kernel_initializer=he_normal(seed=42))
                ], name=f'MetaModel_{model_type}')

        # Compile the model with an optimizer, loss function, and metrics
        if self.ModelMgr_instance.model_choice == 'lr':
            self.meta_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['mae', 'mse'])
        else:
            self.meta_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae', metrics=['mse'])

        # Provide model summary
        self.meta_model.summary()

    def train(self, model_type='logistic'):
        # Train the model
        self.ModelMgr_instance.logger.info('Fitting model')
        if self.ModelMgr_instance.prep == 'multi':
            axistr = (2)
        else:
            axistr = (2, 3)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1) 
        callbacks=[WandbMetricsLogger(), early_stopping, TerminateOnNaN()]

        if self.ModelMgr_instance.partition == 'random':
            history = self.meta_model.fit(np.transpose(np.squeeze(self.ModelMgr_instance.model_inputs, axis=self.axistr)), self.ModelMgr_instance.model_labels,
                                        validation_split=0.2, epochs=self.epochs, batch_size=150, shuffle=True, callbacks=[wandb.keras.WandbCallback()])

        elif self.ModelMgr_instance.partition == 'spatial':
            history = self.meta_model.fit(np.transpose(np.squeeze(self.ModelMgr_instance.model_inputs, axis=self.axistr)), self.ModelMgr_instance.model_labels,
                                        validation_data=(np.transpose(np.squeeze(self.ModelMgr_instance.val_data, axis=self.axistr)),
                                        self.ModelMgr_instance.val_labels), epochs=self.epochs, batch_size=256, shuffle=True, callbacks=callbacks)
        
        fig = self.ModelMgr_instance.plot_val_loss(history, name=f'{self.ModelMgr_instance.test} {self.hazard} Loss base model')
        wandb.log({f"Loss_val {model_type}": wandb.Image(fig)})
        wandb.log({"loss_val": history.history['val_loss'][-1]})
        self.loss_val = history.history['val_loss'][-1]

    def predict(self, model_type='logistic'):
        # Evaluate the model on the validation data
        self.ModelMgr_instance.logger.info('Predicting')
        susceptibility = self.meta_model.predict(np.transpose(np.squeeze(self.ModelMgr_instance.input_data, axis=self.axistr)))
        
        self.ModelMgr_instance.logger.info('Reshaping output')
        susc_map = np.zeros(self.ModelMgr_instance.original_shape)
        # Fill the new_data_array with the values at the specified indices
        for i, index in enumerate(self.ModelMgr_instance.indices_with_values):
            susc_map[index, :, :] = susceptibility[i]

        susc_shape = (self.ModelMgr_instance.output_shape[0] - self.neighborhood_size*2, self.ModelMgr_instance.output_shape[1] - self.neighborhood_size*2)
        susc_map = susc_map.reshape(susc_shape)

        # Create a new array with shape of output and fill it with zeros
        susc_map_reshape = np.zeros(self.ModelMgr_instance.output_shape)

        # Copy the original array into the center of the new array
        susc_map_reshape[self.neighborhood_size:self.neighborhood_size+susc_shape[0], self.neighborhood_size:self.neighborhood_size+susc_shape[1]] = susc_map

        # Plot the susceptibility map
        self.ModelMgr_instance.logger.info('plotting and saving')
        fig = self.ModelMgr_instance.plot(susc_map_reshape, name=f'{self.ModelMgr_instance.test} {self.hazard} Susceptibility meta model {model_type}')
        np.save(f'Output/{region}/{self.hazard}/{self.ModelMgr_instance.test}_{self.hazard}_Susceptibility_meta_model_{model_type}.npy', susc_map_reshape)

        # Save the base model weights
        self.meta_model.save(os.path.join(f'Output/{region}', self.hazard, f'meta_model_{model_type}_{self.ModelMgr_instance.test}.keras'))
        wandb.log({f"Susceptibility  {model_type}": wandb.Image(fig)})

    def testing(self, model_type='logistic'):
        # Evaluate the model on the validation data
        self.ModelMgr_instance.logger.info('Testing')

        y_pred = self.meta_model.predict(np.transpose(np.squeeze(self.ModelMgr_instance.test_data, axis=self.axistr)))
        y_pred = np.squeeze(y_pred, axis=(1))
        y_true = np.squeeze(self.ModelMgr_instance.test_labels, axis=(1,2))

        # Calculate Binary Cross-Entropy
        y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true_tf, y_pred_tf)
        self.bce_test = tf.reduce_mean(bce).numpy()
        self.ModelMgr_instance.logger.info(f"BCE: {self.bce_test}")

        # Metrics
        self.mae = mean_absolute_error(y_true, y_pred)
        self.mse = mean_squared_error(y_true, y_pred)
        self.ModelMgr_instance.logger.info(f"MAE: {self.mae}")
        self.ModelMgr_instance.logger.info(f"MSE: {self.mse}")

        # Create a dictionary to store values with names
        metrics_dict = {'MAE': [self.mae], 'MSE': [self.mse]}
        df = pd.DataFrame(metrics_dict)

        # Write the values to a text file
        df.to_csv(f'Output/{region}/{self.hazard}/config_{self.ModelMgr_instance.test}_metamodel_{model_type}.csv', index=False)

        # Store in W&B
        wandb.log({f"BCE_test_{model_type}": self.bce_test})
        wandb.log({f"MAE_test_{model_type}": self.mae})
        wandb.log({f"MSE_test_{model_type}": self.mse})

    def shap_computation(self, model_type='logistic'):
        # Compute shap values and plot
        self.ModelMgr_instance.logger.info('SHAP')
        X_train = np.transpose(np.squeeze(self.ModelMgr_instance.model_inputs, axis=self.axistr))[:9000]
        X_test = np.transpose(np.squeeze(self.ModelMgr_instance.test_data, axis=self.axistr))[:900]
        variables = ['Drought', 'Extreme Wind', 'Wildfire', 'Heatwave', 'Earthquake', 'Volcano', 'Flood', 'Landslide', 'Tsunami']

        # Create an explainer object
        explainer = shap.DeepExplainer(self.meta_model, data=X_train) 

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)

        # **Summary plots**
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, features=X_test, plot_type='bar', feature_names=variables, show=False)
        ax.set_xlabel('Mean(|SHAP value|)')
        plt.savefig(f'Output/{region}/{self.hazard}/SHAP_values_MEAN_{model_type}.png', dpi=300)
        wandb.log({f"SHAP_bar {model_type}": wandb.Image(fig)})
        plt.close()

        # fig, ax = plt.subplots(figsize=(10, 8))
        # shap.summary_plot(shap_values[0], X_test, feature_names=variables, show=False)
        # ax.set_xlabel('SHAP value')
        # plt.savefig(f'Output/{self.hazard}/SHAP_values_{model_type}.png', dpi=300)
        # wandb.log({f"Shap_labelO {model_type}": wandb.Image(fig)})
        # plt.close()

        # fig, ax = plt.subplots(figsize=(10, 8))
        # # expected_value = explainer.expected_value
        # # shap.decision_plot(expected_value, shap_values[0], X_train, feature_names=variables)
        # shap.decision_plot(0.5, shap_values[0], X_train, feature_names=variables, show=False)
        # plt.savefig(f'Output/{self.hazard}/SHAP_decision_{model_type}.png', dpi=300)
        # wandb.log({f"Shap_decision {model_type}": wandb.Image(fig)})
        # plt.close()

        # Create a figure with an adjusted size
        fig = plt.figure(figsize=(24, 8))  # Adjust width for better visibility

        # Define a list of pastel colors, similar to 'plasma' but lighter
        pastel_colors = [
            (1.0, 0.8, 0.9),  # soft pink
            (0.9, 0.7, 0.9),  # lavender pink
            (0.8, 0.7, 1.0),  # pastel purple
            (0.7, 0.85, 0.95), # light turquoise
            (0.7, 0.9, 0.8),  # mint green
            (0.8, 1.0, 0.7),  # light lime
            (1.0, 1.0, 0.6),  # pastel yellow
        ]

        # Create a custom LinearSegmentedColormap
        pastel_cmap = LinearSegmentedColormap.from_list("custom_pastel", pastel_colors, N=256)

        # First panel for SHAP summary plot
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        shap.summary_plot(shap_values[0], X_test, feature_names=variables, show=False, color_bar=False, cmap=pastel_cmap)
        plt.title('SHAP Summary Plot', fontsize=20)  # Optional title for clarity
        plt.xticks(fontsize=16)  # Set x-tick label font size
        plt.yticks(fontsize=16)  # Set x-tick label font size
        plt.xlabel('SHAP value (impact on model output)', fontsize=16)

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_aspect(80)
        cbar.set_label('Feature value', fontsize=16)
        cbar.ax.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, labelright=False)
        cbar.ax.text(1.7, 1.1, 'High', ha='center', va='center', fontsize=16, color='black', transform=cbar.ax.transAxes)
        cbar.ax.text(1.6, -0.1, 'Low', ha='center', va='center', fontsize=16, color='black', transform=cbar.ax.transAxes)
        cbar.outline.set_visible(False)

        # Second panel for SHAP decision plot
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        shap.decision_plot(0.5, shap_values[0], X_train, feature_names=variables, show=False, plot_color=pastel_cmap)
        plt.title('SHAP Decision Plot', fontsize=20)  # Optional title for clarity
        plt.xticks(fontsize=16)  # Set x-tick label font size
        plt.yticks(fontsize=16)  # Set x-tick label font size
        plt.xlabel('Model output value', fontsize=16)

        # Adjust size like this, but in hindsight would have been better with auto size turned of
        plt.gcf().set_size_inches(24, 8)
        plt.tight_layout()
        # print(f'New size: {plt.gcf().get_size_inches()}')

        # Adjust layout to prevent text cutoff
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'Output/{region}/{self.hazard}/SHAP_combined_{model_type}.png', dpi=300)
        wandb.log({f"Shap_combined {model_type}": wandb.Image(fig)})
        plt.close()

        # **Dependency plot**
        fig, axes = plt.subplots(nrows=len(variables), ncols=len(variables), figsize=(20, 20))

        # Create scatter plots for lower triangle
        for i in range(len(variables)):
            for j in range(len(variables)):
                if i == j:
                    interaction_value = None
                else:
                    interaction_value = j
                # print(variables[i], variables[j])
                shap.dependence_plot(i, shap_values[0], X_test, interaction_index=interaction_value, ax=axes[i, j], show=False, dot_size=3, cmap=pastel_cmap)  # plt.get_cmap('plasma'))
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_ylabel('')
                axes[i, j].set_xlabel('')

                if i == 0 and j == 1:
                    cbar_save = axes[0, 1].collections[0]
                if i != j:
                    axes[i, j].collections[0].colorbar.remove()

                if j == 0 and i == len(variables) - 1:
                    axes[i, j].set_ylabel(variables[i])
                    axes[i, j].set_xlabel(variables[j])
                elif j == 0:
                    axes[i, j].set_ylabel(variables[i])
                elif i == len(variables) - 1:
                    axes[i, j].set_xlabel(variables[j])
                # else:
                #     axes[i, j].axis('off')

        # Create a common legend for all subplots on the right side
        fig.subplots_adjust(right=0.9)  # Adjust layout to make space for legend
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.6])  # Define position for legend
        cbar = fig.colorbar(cbar_save, cax=cbar_ax)  # Use first subplot's colorbar for legend
        cbar.set_label('SHAP Value')  # Set legend label

        plt.savefig(f'Output/{region}/{self.hazard}/SHAP_dependency_{model_type}.png', dpi=300)
        wandb.log({f"Shap_Dependency {model_type}": wandb.Image(fig)})
        plt.close()

    def meta_pipeline(self, model_type='logistic'):
        wandb.init(project=self.hazard, entity="timothy-tiggeloven", group=self.ModelMgr_instance.test, job_type=f"MetaModel_{model_type}",
                   config={"learning_rate": self.learning_rate,
                           "epochs": self.epochs,
                           "filters": self.neurons,
                           "layers": self.n_layers,
                           "seed": self.ModelMgr_instance.seed,
                           "dropout": self.drop_value,
                           "NN_cells": self.neighborhood_size,
                           "variables": self.ModelMgr_instance.variables,
                           "sample_ratio": self.ModelMgr_instance.sample_ratio})
        
        self.design_metamodel(model_type=model_type)
        self.train(model_type=model_type)
        self.predict(model_type=model_type)
        self.testing(model_type=model_type)
        if self.ModelMgr_instance.prep == 'multi' and self.ModelMgr_instance.model_choice == 'meta' and model_type=='MLP':
            self.shap_computation(model_type=model_type)

        wandb.finish()

    def run(self):
        # Set seed
        np.random.seed(self.ModelMgr_instance.seed)
        tf.random.set_seed(self.ModelMgr_instance.seed)

        if self.ModelMgr_instance.prep == 'stack':
            self.calc_weighted_avg()
        self.meta_pipeline(model_type='logistic')
        if self.ModelMgr_instance.model_choice == 'meta':
            self.meta_pipeline(model_type='MLP')

    def HypParOpt(self):
        # Set seed
        np.random.seed(self.ModelMgr_instance.seed)
        tf.random.set_seed(self.ModelMgr_instance.seed)

        # Define sweep config
        sweep_configuration = {
            "method": "bayes",
            "name": f"{self.ModelMgr_instance.test}_MetaModel_MLP",
            "metric": {"goal": "minimize", "name": "loss_val"},
            "parameters": {
                "layers": {"values": [1, 2, 3]},
                "neurons": {"values": [64, 128, 256, 512]},
                "lr": {"max": 0.001, "min": 0.00001},
                "dropout": {"max": 0.5, "min": 0.1},
            },
        }

        # Initialize sweep by passing in config.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"{self.hazard}-sweep")
        self.hyper_df = pd.DataFrame(columns=["layers", "neurons", "lr", "dropout", "MAE", "MSE", "loss_val"], dtype=float)

        # Start sweep job.
        wandb.agent(sweep_id, function=self.main, count=20)

    def main(self):
        self.base_model = False
        if self.ModelMgr_instance.partition == 'random':
            self.ModelMgr_instance.preprocess()
        wandb.init()
        self.n_layers = wandb.config.layers
        self.neurons = wandb.config.neurons
        self.learning_rate = wandb.config.lr
        self.drop_value = wandb.config.dropout

        # Develop a basemodel
        self.design_metamodel(model_type='MLP')
        self.train(model_type='MLP')
        self.predict(model_type='MLP')
        self.testing(model_type='MLP')
        # if self.ModelMgr_instance.prep == 'multi':
        #     self.shap_computation(model_type='MLP')

        new_row = pd.DataFrame([{
            "layers": wandb.config.layers,
            "filters": wandb.config.neurons,
            "lr": wandb.config.lr,
            "dropout": wandb.config.dropout,
            "val_loss": self.loss_val,
            "MAE": self.mae,
            "MSE": self.mse,
        }])
        self.hyper_df = pd.concat([self.hyper_df, new_row], ignore_index=True)
        self.hyper_df.to_csv(f"Output/{region}/{self.hazard}/Sweep_results_MetaModel_{self.ModelMgr_instance.test}.csv", index=False)
        self.ModelMgr_instance.logger.info(f"Main done")

region = 'Japan'
test = 'sado'  # Set test to 'sado', 'hokkaido' or 'japan' as needed
hazard = 'Landslide'  # Set hazard to 'Landslide', 'Flood', 'Tsunami', or 'Multihazard' as needed
hyper = 'False'
# hyper = True
model_choice = 'base'
# model_choice = 'lr'
test = sys.argv[1]
hazard = sys.argv[2]
hyper = sys.argv[3]
model_choice = sys.argv[4]

if hyper == 'False':
    hyper = False
else:
    hyper = True

if model_choice == 'base' or model_choice == 'ensemble':
    prep = 'model'
elif model_choice == 'meta' and hazard == 'Multihazard':
    prep = 'multi'
elif model_choice == 'lr':
    prep = 'multi'
elif model_choice == 'meta':
    prep = 'stack'
else:
    print('Model choice should be base, ensemble or meta')
    sys.exit(1)

# Instantiate and run the BaseModel
model_instance = ModelMgr(test=test, hazard=hazard, hyper=hyper, prep=prep, model_choice=model_choice)  # Set test to 'sado' or 'hokkaido' as needed
# sys.exit(0)

if model_choice == 'base':
    model_instance.train_base_model()
elif model_choice == 'ensemble':
    model_instance.load_base_model()
    model_instance.train_ensemble_model()
elif model_choice == 'meta' or model_choice == 'lr':
    # model_instance.learning_to_stack()
    model_instance.train_meta_model()
# model_instance.load_meta_model()

sys.exit(0)
