
#Best rigrid transformation, ICPS algorithms and Sparse ICPS algorithms


#Imports
import numpy as np
from sklearn.neighbors import KDTree
import scipy

from utils import shrink,diagonal_box





#############################################
###### BEST RIGID TRANSFORM FUNCTION ########
#############################################


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    p=np.mean(ref,axis=1).reshape(-1,1)
    p_prime=np.mean(data,axis=1).reshape(-1,1)
    Q=ref-p
    Q_prime=data-p_prime
    H=Q_prime@Q.T
    U,_,V=np.linalg.svd(H)
    V=V.T
    R = V@U.T
    T = p-R@p_prime

    return R, T

def best_rigid_transform_weighted(data,ref,weight):
    '''
    Computes the weighted least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
        ref = (d x N) matrix where "N" is the number of points and "d" the dimension
        weight = (N) or (Nx1) vector of the weights
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    W=np.sum(weight)

    p=((1/W)*ref@weight).reshape(-1,1)
    p_prime=((1/W)*data@weight).reshape(-1,1)

    Q=ref-p
    Q_prime=data-p_prime

    H=Q_prime@Q.T

    U,_,V=np.linalg.svd(H)
    V=V.T
    R = V@U.T

    T = p-R@p_prime

    return R, T

def best_rigid_transform_plane(data, ref, normal_ref,transfo_appro=False):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref with a plane strategy.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
        ref = (d x N) matrix where "N" is the number of points and "d" the dimension
        normal_ref = (d x N) matrix containing the normal at each points of the ref cloud
        transfo_appro = boolean : True for the approximate version
    Returns :
        R = (d x d) rotation matrix
        T = (d x 1) translation vector
        Such that R * data + T is aligned on ref
    '''


    N=normal_ref
    X=data 
    Y=ref 

    U=X-Y

    c=np.cross(X,N,axis=0) #cross product 

    #creation of b 
    b=np.zeros(6)
    for i in range(6):
        if i <3 : 
            b[i]=-np.sum(np.einsum('ij,ji->i', N, (c[i]*U).T))
        else : 
            b[i]=-np.sum(np.einsum('ij,ji->i', N, (N[i-3]*U).T))

    C=np.concatenate((c,N))@np.concatenate((c,N)).T #Big matrix C 

    #solve the system with Cholesky facto
    v, w= scipy.linalg.cho_factor(C)
    alpha,beta,gamma,t_x,t_y,t_z = scipy.linalg.cho_solve((v, w), b)

    T=np.array([[t_x],[t_y],[t_z]])
    if transfo_appro : 
        R=np.array(([1,-gamma,beta],[gamma,1,-alpha],[-beta,alpha,1]))
    else : 
        Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
        ])

        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        R=Rz@Ry@Rx

    return R,T

def best_rigid_transform_plane_SICP(data,ref,normal_ref,u,transfo_appro=False):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref with a plane strategy for SICP.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
        ref = (d x N) matrix where "N" is the number of points and "d" the dimension
        normal_ref = (d x N) matrix containing the normal at each points of the ref cloud
        transfo_appro = boolean : True for the approximate version
    Returns :
        R = (d x d) rotation matrix
        T = (d x 1) translation vector
        Such that R * data + T is aligned on ref
    '''

    X=data
    Y=ref
    N=normal_ref

    LHS = np.zeros((6, 6))
    RHS = np.zeros(6)

    C = np.cross(X, N, axis=0)

    for i in range(X.shape[1]):
        ci = C[:, i][:, np.newaxis]
        ni = N[:, i][:, np.newaxis]

        TL = ci @ ci.T 
        TR = ci @ ni.T 
        BR = ni @ ni.T 
        LHS[:3, :3] += TL
        LHS[:3, 3:] += TR
        LHS[3:, 3:] += BR
        dist_to_plane = -((X[:, i] - Y[:, i]) @ N[:, i] - u[i]) 
        RHS[:3] += ci.flatten() * dist_to_plane
        RHS[3:] += ni.flatten() * dist_to_plane
    
    v, w = scipy.linalg.cho_factor(LHS)
    alpha,beta,gamma,t_x,t_y,t_z = scipy.linalg.cho_solve((v, w), RHS)
    T=np.array([[t_x],[t_y],[t_z]])
    if transfo_appro : 
        R=np.array(([1,-gamma,beta],[gamma,1,-alpha],[-beta,alpha,1]))
    else : 
        Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
        ])

        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        R=Rz@Ry@Rx

    return R,T




######################################
###### ICPS ALGORTIHMS ###############
######################################






def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_cumulative = np.eye(data.shape[0]) 
    T_cumulative = np.zeros((data.shape[0], 1)) 
    R_list = [R_cumulative]
    T_list = [T_cumulative]

    neighbors_list = []
    RMS_list = []
    RMS=100
    ite=0
    while RMS>RMS_threshold and ite<max_iter:
        #we construct a tree on the reference cloud 
        kd_tree=KDTree(data=ref.T,leaf_size=100,metric="l2")

        #We searsh for the nearest neighboor for each point in data
        dis,ind=kd_tree.query(data_aligned.T,k=1)
        ind=np.squeeze(ind)
        neighbors_list.append(ind)

        #find the matched point
        matched_data=ref[:,ind]
        R,T=best_rigid_transform(data=data_aligned, ref=matched_data)
        data_aligned=R.dot(data_aligned) + T

        #compute cumulative for show_ICP
        R_cumulative = R @ R_cumulative
        T_cumulative = R @ T_cumulative + T
        R_list.append(R_cumulative)
        T_list.append(T_cumulative)

        #RMS
        RMS = np.sqrt(np.mean(dis**2))
        RMS_list.append(RMS)
        print(f'The error for the {ite}th iteration is : {RMS}')

        ite+=1

    return data_aligned, R_list, T_list, neighbors_list, RMS_list




def icp_point_to_point_correspondances_prunning(data, ref, delta, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy with correspondences prunning.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        delta = scalar between 0 and 1 percentage of diagonal bouding box pruning
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
        percentage_of_points = percentage of points taken in account for the computation of the transform, by iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_cumulative = np.eye(data.shape[0]) 
    T_cumulative = np.zeros((data.shape[0], 1)) 
    R_list = [R_cumulative]
    T_list = [T_cumulative]

    neighbors_list = []
    RMS_list = []
    percentage_of_points=[]
    RMS=100
    ite=0
    while RMS>RMS_threshold and ite<max_iter:
        #we construct a tree on the reference cloud 
        kd_tree=KDTree(data=ref.T,leaf_size=100,metric="minkowski")
        

        #We searsh for the nearest neighboor for each point in data
        dis,ind=kd_tree.query(data_aligned.T,k=1)

        match=ref[:,ind]
        cut=diagonal_box(ref=match.T,delta=delta)
        ind=ind[dis<cut] # We only use the indices for which the distance is below delta*diagonal

        neighbors_list.append(ind)
        percentage_of_points.append(len(ind)/data_aligned.shape[1])

        #find the matched point
        matched_data=ref[:,ind]
        data_aligned_transform=data_aligned[:,np.where(dis<cut)[0]] # We only use the pointd for which the distance is below delta*diagonal

        R,T=best_rigid_transform(data=data_aligned_transform, ref=matched_data)

        data_aligned=R.dot(data_aligned) + T #we compute the transformation for all the cloud

        #compute cumulative for show_ICP
        R_cumulative = R @ R_cumulative
        T_cumulative = R @ T_cumulative + T
        R_list.append(R_cumulative)
        T_list.append(T_cumulative)

        #RMS
        RMS = np.sqrt(np.mean(dis**2))
        RMS_list.append(RMS)
        print(f'The error for the {ite}th iteration is : {RMS}')
        print(f'The percentage of points considered for the {ite}th iteration is : {percentage_of_points[ite]*100}%')

        ite+=1

    return data_aligned, R_list, T_list, neighbors_list, RMS_list 



def icp_point_to_point_weighted(data, ref, p, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point and a weighted strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        p = degree of the norm use for the weights
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_cumulative = np.eye(data.shape[0]) 
    T_cumulative = np.zeros((data.shape[0], 1)) 
    R_list = [R_cumulative]
    T_list = [T_cumulative]

    neighbors_list = []
    RMS_list = []
    RMS=100
    ite=0

    weight=np.ones(data_aligned.shape[1])

    while RMS>RMS_threshold and ite<max_iter:
        #we construct a tree on the reference cloud 
        kd_tree=KDTree(data=ref.T,leaf_size=100,metric="l2")

        #We searsh for the nearest neighboor for each point in data
        dis,ind=kd_tree.query(data_aligned.T,k=1)
        ind=np.squeeze(ind)
        neighbors_list.append(ind)

        #find the matched point
        matched_data=ref[:,ind]
        R,T=best_rigid_transform_weighted(data=data_aligned, ref=matched_data,weight=weight)
        data_aligned=R.dot(data_aligned) + T

        weight=np.linalg.norm(data_aligned-matched_data,axis=0)**(p-2)
        print(np.min(weight,axis=-1))

        #compute cumulative for show_ICP
        R_cumulative = R @ R_cumulative
        T_cumulative = R @ T_cumulative + T
        R_list.append(R_cumulative)
        T_list.append(T_cumulative)

        #RMS
        RMS = np.sqrt(np.mean(dis**2))
        RMS_list.append(RMS)
        print(f'The error for the {ite}th iteration is : {RMS}')

        ite+=1

    return data_aligned, R_list, T_list, neighbors_list, RMS_list



def icp_point_to_plane(data, ref, normal_ref, max_iter, RMS_threshold,transfo_appro=False):
    '''
    Iterative closest point algorithm with a point to plane strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        normal_ref = (d X N_ref) matrix containing the normal at each point of the ref cloud
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        transfo_appro = boolean : True for the approximate version
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_cumulative = np.eye(data.shape[0]) 
    T_cumulative = np.zeros((data.shape[0], 1)) 
    R_list = [R_cumulative]
    T_list = [T_cumulative]

    neighbors_list = []
    RMS_list = []
    RMS=100
    ite=0
    while RMS>RMS_threshold and ite<max_iter:
        #we construct a tree on the reference cloud 
        kd_tree=KDTree(data=ref.T,leaf_size=100,metric="l2")

        #We searsh for the nearest neighboor for each point in data
        dis,ind=kd_tree.query(data_aligned.T,k=1)
        ind=np.squeeze(ind)
        neighbors_list.append(ind)

        #find the matched point
        matched_data=ref[:,ind]
        matched_normal=normal_ref[:,ind]

        R,T=best_rigid_transform_plane(data=data_aligned, ref=matched_data, normal_ref=matched_normal,transfo_appro=transfo_appro)
        data_aligned=R.dot(data_aligned) + T

        #compute cumulative for show_ICP
        R_cumulative = R @ R_cumulative
        T_cumulative = R @ T_cumulative + T
        R_list.append(R_cumulative)
        T_list.append(T_cumulative)

        #RMS
        RMS = np.sqrt(np.mean(dis**2))
        RMS_list.append(RMS)
        print(f'The error for the {ite}th iteration is : {RMS}')

        ite+=1

    return data_aligned, R_list, T_list, neighbors_list, RMS_list








#############################################
###### SPARSE ICPS ALGORTIHMS ###############
#############################################




def Sparse_ICP(data, ref, p, mu, max_iter_icp, RMS_threshold,max_iter_in=1,max_iter_out=50,alpha=1.1,mu_max=1000,use_penalty=True,print_all_transfo=False):
    
    '''
    Sparse iterative closest point algorithm with a point to point strategy (using L^p norm and ADMM).
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        p = order of the norm used for optimization
        mu = penalty weight for the Augmented Lagrangian
        max_iter_icp = stop condition on the number of iterations for the ICP
        RMS_threshold = stop condition on the distance
        max_iter_in, max_iter_out = stop condition on the number of iterations for the optimization (ADMM)
        alpha = multiplication factor of the regularization parameter mu
        mu_max = max value for the regularization parameter mu 
        use_penalty = Boolean to use penalty or not
        print_all_transfo = Boolean : True if you want the complete R_list and T_list through the inner iteration of SICP
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    #Attribution of diferents variables
    mu_init=mu
    Z=np.zeros_like(data_aligned)
    C=np.zeros_like(data_aligned) 
    X=data_aligned
    Y=ref

    # Initiate lists
    R=np.eye(data.shape[0]) 
    T=np.zeros((data.shape[0], 1)) 
    R_cumulative = R
    T_cumulative = T 
    R_list = [R_cumulative]
    T_list = [T_cumulative]


    neighbors_list = []
    RMS_list = []
    RMS=100
    ite_icp=0

    #we construct a tree on the reference cloud 
    kd_tree=KDTree(data=ref.T,leaf_size=100,metric="l2")

    while (RMS>RMS_threshold and ite_icp<max_iter_icp) :

        #STEP 1 : Coresspondances 

        #We searsh for the nearest neighboor for each point in data
        dis,ind=kd_tree.query(X.T,k=1)
        ind=np.squeeze(ind)
        neighbors_list.append(ind)

        matched_data=Y[:,ind] #nuage ref 
        Q=matched_data

        #STEP 2 : Alignment

        #Re-initialization of parameters for the ADMM loop 
        Z=np.zeros_like(data_aligned)
        C=np.zeros_like(data_aligned)
        mu=mu_init
        #Start of the ADMM loop
        for i in range(max_iter_out) : 
            for j in range(max_iter_in) : 
                #(1)
                H = X-Q+C/mu
                Z = shrink(H,mu,p,iter=3)

                #(2)
                U = Q+Z-C/mu
                R,T = best_rigid_transform(data=X, ref=U)

                #compute cumulative for show_ICP
                R_cumulative = R @ R_cumulative
                T_cumulative = R @ T_cumulative + T
                if print_all_transfo :
                    R_list.append(R_cumulative)
                    T_list.append(T_cumulative)
            #(3)
            X=R.dot(X)+T #MAJ of X for the ADMM loop 
            delta = X-Q-Z

            if use_penalty :
                C = C+mu*delta

            if mu < mu_max :
                mu = mu*alpha


        if not(print_all_transfo) :
            R_list.append(R_cumulative)
            T_list.append(T_cumulative)  

        #RMS
        RMS = np.sqrt(np.mean(dis**2))
        RMS_list.append(RMS)
        print(f'The error for the {ite_icp}th iteration is : {RMS}')

        ite_icp+=1

    return X, R_list, T_list, neighbors_list, RMS_list

def Sparse_ICP_point_to_plane(data, ref,normal_ref, p, mu, max_iter_icp, RMS_threshold,max_iter_in=1,max_iter_out=50,alpha=1.1,mu_max=1000,use_penalty=True,print_all_transfo=False):
    
    '''
    Sparse iterative closest point algorithm with a point to plane strategy (using L^p norm and ADMM).
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        normal_ref = (d x N_ref) matrix containing the normals at each points of the ref cloud
        p = order of the norm used for optimization
        mu = penalty weight for the Augmented Lagrangian
        max_iter_icp = stop condition on the number of iterations for the ICP
        RMS_threshold = stop condition on the distance
        max_iter_in, max_iter_out = stop condition on the number of iterations for the optimization (ADMM)
        alpha = multiplication factor of the regularization parameter mu
        mu_max = max value for the regularization parameter mu 
        use_penalty = Boolean to use penalty or not
        print_all_transfo = Boolean : True if you want the complete R_list and T_list through the inner iteration of
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''


    # Variable for aligned data
    data_aligned = np.copy(data)

    #Attribution of diferents variables
    mu_init=mu
    Z=np.zeros_like(data_aligned.shape[1])
    C=np.zeros_like(data_aligned.shape[1]) 
    X=data_aligned
    Y=ref

    # Initiate lists
    R=np.eye(data.shape[0]) 
    T=np.zeros((data.shape[0], 1)) 
    R_cumulative = R
    T_cumulative = T 
    R_list = [R_cumulative]
    T_list = [T_cumulative]


    neighbors_list = []
    RMS_list = []
    RMS=100
    ite_icp=0

    #we construct a tree on the reference cloud 
    kd_tree=KDTree(data=ref.T,leaf_size=100,metric="l2")

    while (RMS>RMS_threshold and ite_icp<max_iter_icp) :

        #STEP 1 : Coresspondances 

        #We searsh for the nearest neighboor for each point in data
        dis,ind=kd_tree.query(X.T,k=1)
        ind=np.squeeze(ind)
        neighbors_list.append(ind)

        matched_data=Y[:,ind] #nuage ref 
        matched_normal=normal_ref[:,ind]
        Q=matched_data
        N=matched_normal

        #STEP 2 : Alignment

        #Re-initialization of parameters for the ADMM loop 
        Z=np.zeros_like(data_aligned.shape[1])
        C=np.zeros_like(data_aligned.shape[1])
        mu=mu_init
        #Start of the ADMM loop
        for i in range(max_iter_out) : 
            for j in range(max_iter_in) : 
                #(1)
                H = np.sum(N * (X - Q), axis=0) + C / mu
                Z = shrink(H,mu,p,iter=3)

                #(2)
                U = Z-C/mu
                R,T = best_rigid_transform_plane_SICP(data=X, ref=Q,normal_ref=N,u=U)

                #compute cumulative for show_ICP
                R_cumulative = R @ R_cumulative
                T_cumulative = R @ T_cumulative + T
                if print_all_transfo :
                    R_list.append(R_cumulative)
                    T_list.append(T_cumulative)
            #(3)
            X=R.dot(X)+T #MAJ of X for the ADMM loop 

            delta=np.sum(N * (X - Q), axis=0) - Z
            
            if use_penalty :
                C = C+mu*delta

            if mu < mu_max :
                mu = mu*alpha



        if not(print_all_transfo) :
            R_list.append(R_cumulative)
            T_list.append(T_cumulative)  

        #RMS
        RMS = np.sqrt(np.mean(dis**2))
        RMS_list.append(RMS)
        print(f'The error for the {ite_icp}th iteration is : {RMS}')

        ite_icp+=1

    return X, R_list, T_list, neighbors_list, RMS_list
