

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import loadmat

# Generate Toeplitz matrix
def create_toeplitz_matrix(time_series):
    N = len(time_series)
    return linalg.toeplitz(time_series)


# Plot a matrix as a 2D image (heatmap) and save it
def plot_matrix_as_image(matrix, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.grid(False)
    plt.savefig(filename)  # Save the image
    plt.show()

# Path to the folder containing the files

path = ""
# Directory to save images

save_path = ""

# Ensure the save directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

files = os.listdir(path)
filtered_files = [f for f in files if f != '.DS_Store']
 
#for file in os.listdir(path):    
for file in filtered_files:   
    print(file)
    df = loadmat(path + file)
    #time_series = df.iloc[:, 0].values[:200]

    pv_clear = df.get('PV_clear', None)
    op_clear = df.get('OP_clear', None)

    # Ensure PV_clear was found and is not empty
    if pv_clear is None:
        raise ValueError("The variable 'PV_clear' was not found in the .mat file.")
    if op_clear is None:
        raise ValueError("The variable 'OP_clear' was not found in the .mat file.")
    else:
    # Select the first 200 points
        #SQRT+
        
        pv_clear_reduced =pv_clear[:200] 
        pv_clear_reduced_mean = np.mean(pv_clear_reduced)
        op_clear_reduced =op_clear[:200]  
        op_clear_reduced_mean = np.mean(op_clear_reduced)
     
        time_series1 = np.sqrt((pv_clear_reduced-pv_clear_reduced_mean)**2 + (op_clear_reduced-op_clear_reduced_mean)**2)
        time_series2 = np.sqrt((pv_clear_reduced-pv_clear_reduced_mean)**2 * (op_clear_reduced-op_clear_reduced_mean)**2)
        
        #Delta
        
        delta_PV = pv_clear_reduced [:-1] - pv_clear_reduced [1:]
        delta_OP = op_clear_reduced [:-1] - op_clear_reduced [1:]
        time_series = []
        for i in range(len(delta_PV)):
            if delta_PV[i] != 0:
                time_series.append(op_clear_reduced[i] / delta_PV[i])
            else:
                time_series.append(np.array([0]))
        # Handle division by zero
        
        time_series3 = np.array(time_series)
        
        
        time_series1=time_series1.reshape(-1)
        time_series2=time_series2.reshape(-1)
        time_series3=time_series3.reshape(-1)
        
     
    # Compute matrices
    toeplitz_matrix1 = create_toeplitz_matrix(time_series1)
    toeplitz_matrix2 = create_toeplitz_matrix(time_series2)
    toeplitz_matrix3 = create_toeplitz_matrix(time_series3)
    toeplitz_matrix = np.stack((toeplitz_matrix1[:199,:199],toeplitz_matrix2[:199,:199],toeplitz_matrix3), axis = -1)

    
    # Create filenames for the images
    toeplitz_filename = os.path.join(save_path, f'toeplitz_matrix_{file[:-4]}.png')
    #circulant_filename = os.path.join(save_path, f'circulant_matrix_{file[:-5]}.png')
    
    # Save and plot the Toeplitz matrix
    plot_matrix_as_image(toeplitz_matrix, "Toeplitz Matrix", toeplitz_filename)
    
    # Save and plot the Circulant matrix
    #plot_matrix_as_image(circulant_matrix, "Circulant Matrix", circulant_filename)
