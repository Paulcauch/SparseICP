#Utils Functions

#Imports
import numpy as np
from sklearn.neighbors import KDTree



def diagonal_box(ref,delta,data=None):
    """
    Gives the length of delta*(diagonal of the bounding box)
    if data is given it gives the average between of the bounding box for ref and data multiply by delta
    Inputs :
        ref = (d x N_ref) matrix where "N_data" is the number of points and "d" the dimension
        delta = scalar between 0 and 1
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
    Returns :
        length = length of delta*(diagonal of the bounding box)
    """

    max_bound=np.max(ref,axis=-1)
    min_bound=np.min(ref,axis=-1)

    diagonal=np.linalg.norm(max_bound-min_bound)

    if data!=None:
        max_bound_data=np.max(data,axis=-1)
        min_bound_data=np.min(data,axis=-1)

        diagonal_data=np.linalg.norm(max_bound_data-min_bound_data)

        length=0.5*delta*(diagonal+diagonal_data)

        return length
    
    length=delta*diagonal

    return length


def shrinkage_operator(n, mu, p,s,iter):
    """
    Apply the shrinkage operator to a vector h
    Inputs : 
            n = norm of the vector that we want to shrink
            mu = penalty weight
            p = order of the norm
            s = initialization for the shrinkage
            iter = number of iteration of the shrinkage operator  
    Outputs : 
            beta = the coefficient of shrinkage
    """
    beta=s
    for _ in range(iter):
        beta = 1 - (p/mu) * (n ** (p-2)) * beta ** (p-1)
    return beta


def shrink(Z,mu,p,iter):
    """
    Apply the shrinkage operator to solve the ADMM problem. 
    Inputs : 
            Z = (dxN_data) matrix which represents the residuals
            mu = penalty weight
            p = order of the norm
            iter = number of iteration of the shrinkage operator 
    Outputs : 
            Z_shrunk = Z after the the shrinkage (Step 1 of the ADMM optimization)
    
    """
    Ba = ((2 / mu) * (1 - p))**(1 / (2 - p))
    ha = Ba + (p / mu) * Ba**(p - 1)
    
    Z_shrunk = np.zeros_like(Z)
    if len(Z.shape)==1 : 
        for i in range(len(Z)):
            n=np.abs(Z[i])
            if n > ha :
                Z_shrunk[i] = Z[i]*shrinkage_operator(mu,n, p,(Ba/n + 1)/2,iter)
    else:
        for i in range(Z.shape[1]) :  # Z is of the form (3,N)
            n=np.linalg.norm(Z[:, i])
            if n > ha :
                Z_shrunk[:, i] = Z[:, i]*shrinkage_operator(mu,n, p,(Ba/n + 1)/2,iter)
    return Z_shrunk


def compute_local_PCA_knn(query_points, cloud_points, n_neighbor):
    """
    compute PCA on the neighborhoods of all query_points in cloud_points 
    Inputs : 
        query_points = (N_query x d) matrix where "N_query" is the number of points and "d" the dimension
        cloud_points = (N_cloud x d) matrix where "N_cloud" is the number of points and "d" the dimension
        n_neighbor = number of neighbor use for the knn query 
    Outputs : 
        all_eigenvalues = the eigen values of the covariance matrix of the nearest points 
        all_eigenvectors = the eigen vectors of the covariance matrix of the nearest poitns
    """
    
    all_eigenvalues=[]
    all_eigenvectors=[]

    kd_tree=KDTree(data=cloud_points,leaf_size=100,metric="l2")
    _,voisins=kd_tree.query(X=query_points,k=n_neighbor)

    for p in voisins:

        mil=np.mean(cloud_points[p],axis=0)
        Q=(cloud_points[p]-mil).T
        cov=(1/len(p))*Q@Q.T

        eigenvalues,eigenvectors = np.linalg.eigh(cov)

        all_eigenvalues.append(eigenvalues)
        all_eigenvectors.append(eigenvectors)

    all_eigenvalues = np.array(all_eigenvalues)
    all_eigenvectors = np.array(all_eigenvectors)

    return all_eigenvalues, all_eigenvectors



def RMSE_with_real(data_base,real,R_list,T_list,nbre_ite):
    """
    Compute the RMSE with the cloud that we want to align
    (not the same as the ref one if there is only partial overlapping)

    Inputs :
        data_base = (N_base x d) matrix where "N_base" is the number of points and "d" the dimension, the cloud that we want to align
        real = (N_real x d) matrix where "N_real" is the number of points and "d" the dimension, the cloud that we want to align on (not necessary the ref cloud)
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        nbre_ite = number of iteration of the algorithm used to compute R_list and T_list
    Returns :
        list = the list of the RMSE between data_base and real through the iteration
    
    
    """
    data=np.copy(data_base)

    kd_tree=KDTree(data=real.T,leaf_size=100,metric='l2')

    list=[]

    dis,_=kd_tree.query(data.T,k=1)
    list.append(np.sqrt(np.mean(dis**2)))

    for i in range(nbre_ite):

        R=R_list[i]
        T=T_list[i]

        trans=R.dot(data)+T

        dis,_=kd_tree.query(trans.T,k=1)
        list.append(np.sqrt(np.mean(dis**2)))

    return list


