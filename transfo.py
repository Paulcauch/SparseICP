
#Transformations function for clouds

#Imports
import numpy as np
import os 
import open3d as o3d

from ply import write_ply,read_ply





def noise_and_transformation(data_path,sigma,t,axis):
    """
    Apply noise and a transformation to a PLY file and recreate another PLY file: 
    Inputs : 
        data_path = path of an (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        sigma = std of the noise added
        t = scalar translation parameter
    """
    data_ply=read_ply(data_path)
    data = np.vstack((data_ply['x'], data_ply['y'], data_ply['z']))
    data+=np.random.normal(0,sigma,data.shape)
    for ax in axis : 
        data[ax]+=np.ones(data.shape[1])*t[ax]

    directory = os.path.dirname(data_path)  # Get the directory of the input file.
    path = os.path.join(directory, 'noisy_data.ply')
    write_ply(path, [data.T], ['x', 'y', 'z'])

def crop_random_points(data,percentage):
    """
    Remove a percentage of columns (points) from a matrix at random.
    
    Parameters:
        data = A numpy array (d x N) where "N" is the number of points and "d" the dimension.
        percentage = The percentage of points to remove, as a float between 0 and 100.
    
    Returns:
        reduced_matrix = A numpy array with the requested percentage of columns removed.
    """
 
    num_points = data.shape[1]
    num_remove = int(num_points * (percentage / 100.0))

    # Generate a random selection of indices to remove
    remove_indices = np.random.choice(num_points, num_remove, replace=False)
    
    # Remove the selected indices
    remaining_indices = np.delete(np.arange(num_points), remove_indices)
    reduced_matrix = data[:, remaining_indices]

    return reduced_matrix


def decimated(data_path,factor):

    """
    Decimate a cloud by a factor and create a new file for the decimated cloud

    Inputs : 
        data_path = path of an (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        factor = scalar representing of how much you want to downsize the cloud
    
    """
    data_ply=read_ply(data_path)
    data = np.vstack((data_ply['x'], data_ply['y'], data_ply['z'])) 

    data_decimated=data[:,0::factor]

    directory = os.path.dirname(data_path)  # Get the directory of the input file.
    path = os.path.join(directory, 'data_decimated.ply')
    write_ply(path, [data_decimated.T], ['x', 'y', 'z'])

def add_abberant_points(data,number,sigma):
    """
    Add outliers/abberant point to a cloud by adding points at a Gaussian distance from a random point in data

    Inputs : 
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        number = scalar, the number of outliers added
        sigma = scalar, the std of the Gaussian 
    Returns : 
        data_abberant = the initial data with outliers added
    
    """
    num_points=data.shape[1]

    data_abberant=data
    for i in range(number):
        #choose a point 
        ind=np.random.randint(num_points)
        #create aberrant point 
        abberant=data[:,ind]+np.random.normal(0,sigma,size=data.shape[0])
        data_abberant=np.concatenate((data_abberant,abberant[:,None]),axis=-1)
    
    return data_abberant

def transfo(data,trans):
    """
    Add a linear term on the wanted dimensions of data

    Inputs : 
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        trans = list of length d, where trans[i] is the translation for the i_th dimension
    Returns : 
        data_trans = the initial data translated
    """
    data_trans=data
    for i,t in enumerate(trans) : 
        data_trans[i]+=np.ones(data.shape[1])*t
    return data_trans