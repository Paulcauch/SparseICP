# Tests with differents dataset

#Imports
import numpy as np
from matplotlib import pyplot as plt

from ply import read_ply,write_ply
from utils import *
from ICPS import * 
from visu import * 
from transfo import *


#BUNNY
def bunny_classical_SICP(p,bunny,ite):

    # Cloud paths
    bunny_o_path = 'data /bunny_original.ply'
    
    if bunny=='p':
        bunny_path = 'data /bunny_perturbed.ply'
    if bunny=='vp':
        bunny_path='data /bunny_very_perturbed.ply'
    if bunny=='r':
        bunny_path='data /bunny_returned.ply'

    # Load clouds
    bunny_o_ply = read_ply(bunny_o_path)
    bunny_ply = read_ply(bunny_path)

    bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
    bunny = np.vstack((bunny_ply['x'], bunny_ply['y'], bunny_ply['z']))

    # Apply ICP
    bunny_p_opt, R_list, T_list, neighbors_list, RMS_list=Sparse_ICP(bunny,bunny_o,p=p,mu=10,max_iter_icp=ite,RMS_threshold=1e-4)

    # Show ICP
    show_ICP(bunny, bunny_o, R_list, T_list, neighbors_list)
    
    # Plot RMS
    plt.plot(RMS_list)
    plt.show()


#OWLS
def owls_SICP_plane(p,ite):

    owl_right_path='data /owl_pointcloud/owl_segment_left.ply'
    owl_left_path='data /owl_pointcloud/owl_segment_right.ply'

    owl_left_ply=read_ply(owl_left_path)
    owl_right_ply=read_ply(owl_right_path)

    owl_left=np.vstack((owl_left_ply['x'],owl_left_ply['y'],owl_left_ply['z']))
    owl_right=np.vstack((owl_right_ply['x'],owl_right_ply['y'],owl_right_ply['z']))

    per=90
    
    owl_left_original=crop_random_points(owl_left,per)
    oo=np.copy(owl_left_original)
    owl_right=crop_random_points(owl_right,per)
    print(f'We cropped {per}% of the cloud. the new shape of owl_left is {owl_left.shape}, the new shape of owl_right is {owl_right.shape}.')
    owl_left=transfo(oo,[-3,0,0])

    all_eigenvalues_r, all_eigenvectors_r = compute_local_PCA_knn(owl_right.T, owl_right.T, 30)
    normals_owl_right = all_eigenvectors_r[:, :, 0].T
    print('normal calculated.')

    
    # Apply ICP

    owl_p_opt, R_list, T_list, neighbors_list, RMS_lis2=Sparse_ICP_point_to_plane(owl_left,owl_right,normal_ref=normals_owl_right,p=p,mu=10,max_iter_icp=ite,RMS_threshold=1e-4)
   

    write_ply('data /owl_pointcloud/owl_left_décalé',owl_left.T,['x','y','z'])
    write_ply(f'data /owl_pointcloud/owl_SICP_opt_p={p}',owl_p_opt.T,['x','y','z'])
    
    # Show ICP
    show_ICP(owl_left, owl_right, R_list, T_list, neighbors_list)
    
    #Compute the real RMS
    RMS=RMSE_with_real(data_base=owl_left,real=owl_left_original,R_list=R_list,T_list=T_list,nbre_ite=ite)

    # Plot RMS
    plt.plot(RMS,color='b',label='SICP point to plane')
    plt.xlabel('number of iterations')
    plt.ylabel('RMS')
    plt.title('Evolution of the RMS for the SICP')
    plt.legend()
    plt.show()

def bunny_SICP_plane(p,ite):

    # Cloud paths
    bunny_top_path = 'data /bunny/bunny_top_segment.ply'
    bunny_bottom_path = 'data /bunny/bunny_bottom_segment.ply'

    # Load clouds
    bunny_top_ply = read_ply(bunny_top_path)
    bunny_bottom_ply = read_ply(bunny_bottom_path)


    bunny_top = np.vstack((bunny_top_ply['x'], bunny_top_ply['y'], bunny_top_ply['z']))
    bunny_bottom_real = np.vstack((bunny_bottom_ply['x'], bunny_bottom_ply['y'], bunny_bottom_ply['z']))
    oo=np.copy(bunny_bottom_real)

    bunny_bottom=transfo(oo,[0.2,0,0])


    all_eigenvalues_r, all_eigenvectors_r = compute_local_PCA_knn(bunny_top.T, bunny_top.T, 30)
    normals_bunny_top = all_eigenvectors_r[:, :, 0].T
    print('normal calculated.')

    
    # Apply ICP
    bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = Sparse_ICP_point_to_plane(data=bunny_bottom,ref=bunny_top,normal_ref=normals_bunny_top,p=0.4,mu=10,max_iter_icp=ite,RMS_threshold=1e-4)
    
    write_ply('data /bunny/bunny_bottom_décalé',bunny_bottom.T,['x','y','z'])
    write_ply(f'data /bunny/SICP_p={p}',bunny_p_opt.T,['x','y','z'])
   
    RMS_plane=RMSE_with_real(data_base=bunny_bottom,real=bunny_bottom_real,R_list=R_list,T_list=T_list,nbre_ite=ite)


    # Show ICP
    show_ICP(bunny_bottom, bunny_top, R_list, T_list, neighbors_list)
 
    # Plot RMS
    plt.semilogy(RMS_plane,color='r',label=f'SICP p={p}')
    plt.xlabel('number of iterations')
    plt.ylabel('RMS')
    plt.title('Evolution of the RMS for the SICP')
    plt.legend()
    plt.show()


