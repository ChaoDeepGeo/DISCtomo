import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import sys
from pyekfmm import eikonal,ray2d2  #extract traveltime and ray tracing (new) functions from pyekfmm
from pyseistr.solvers import solver,conjgrad
from pyseistr.solvers import cgstep
from pyseistr.operators import adjnull,matrix_lop
from pyseistr.divne import trianglen_lop
from scipy.ndimage import gaussian_filter
import scipy.sparse as sp
from pyekfmm import ray3d2,extract
import gstools as gs
import math
import torch



def compute_travel_times(v_true, sources, receivers, dx, dy, nx, ny, stepa=1,ifwav=False):
    """
    Compute travel times and ray paths for a set of sources and receivers in a 2D velocity model.
    based on the Fast Marching Method (FMM) and eikonal euqation.
    
    Parameters:
    v_true : 2D array
        The true velocity model (shape: nx x ny).
    sources : list of tuples
        List of source coordinates [(sx1, sy1), (sx2, sy2), ...].
    receivers : list of tuples
        List of receiver coordinates [(rx1, ry1), (rx2, ry2), ...].
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    stepa : float
        step size for ray traing (0~1)
        when velocity is accuract stepa=1
        initially, a smaller step size help to improve accuracy
    ifwav : bool, optional
        If True, output traveltime field for the last source (default is False).

    
    Returns:
    travel_times : 1D array
        Computed travel times for each source-receiver pair.
    ray_paths : list of tuples
        List of ray paths for each source-receiver pair.
    """
    
    travel_times = []
    ray_paths = []  

    for sx, sy in sources:
    # FMM travel-time field from this source
    
        t=eikonal(v_true.transpose().flatten(order='F'),xyz=np.array([sx,0,sy]),ax=[0,dx,nx],ay=[0,dy,1],az=[0,dy,ny],order=2);
        T=t.reshape(nx,ny,order='F').T;  #first axis (vertical) is x, second is y
        for rx, ry in receivers:
                
                travel_times.append(T[round(ry/dy), round(rx/dx)])  # FMM travel time at receiver
	
                paths=ray2d2(T,np.array([sx,sy]),np.array([rx,ry]),ax=[0,dx,nx],ay=[0,dy,ny],step=stepa)
                ray_paths.append((paths.transpose()[:,0],paths.transpose()[:,1]))
    
    travel_times = np.array(travel_times)
    if ifwav:
        return travel_times, ray_paths, T
    else:
        return travel_times, ray_paths


def build_ray_matrix(ray_paths, dx, dy, nx, ny, ifraycorr=True):
    """
    Build the ray matrix A for a set of ray paths in a 2D grid.
    
    Parameters:
    ray_paths : list of tuples
        List of ray paths for each source-receiver pair.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    ifraycorr : bool, optional
        If True, conduct ray path contribution correction
    
    Returns:
    A : sparse matrix
        The ray matrix (shape: n_rays x n_cells).
    """

    n_rays = len(ray_paths)
    n_cells = nx*ny
    A = lil_matrix((n_rays, n_cells)) #A is a sparse ray-path matrix

    for i, (ray_x, ray_y) in enumerate(ray_paths):
        for x, y in zip(ray_x, ray_y):
            ix, iy = int(round(x/dx)), int(round(y/dy))
            idx = iy*nx + ix
            A[i, idx] += 1.0*dx  # Each cell contributes 1*dx to travel path

    """
    A is a sparse matrix, but some cells may be counted multiple times if the ray passes through 
    them more than once. This line ensures that the maximum contribution from any cell is capped at 1*dx.
    """
    if ifraycorr:
        A[(A>1.0*dx*1)]=1.0*dx 
        # A[(A>1.0*dx*2)]=1.0*d
    
    return A  

def first_diff_2d_F(nz: int, nx: int):
    """
    Build 2D first-order forward-difference operators for a (nz, nx) grid,
    assuming vectorization with flatten(order='F').

    Returns
    -------
    Dz : sparse matrix, shape ((nz-1)*nx, nz*nx)
         forward differences along z (row direction)
    Dx : sparse matrix, shape (nz*(nx-1), nz*nx)
         forward differences along x (col direction)
    """
    N = nz * nx

    # --- Dz: for each column j, differences between rows i and i+1
    # Each block is 1D diff of size (nz-1) x nz, repeated nx times.
    ez = np.ones(nz)
    D1z = sp.diags([-ez, ez], [0, 1], shape=(nz-1, nz), format="csr")
    Dz = sp.kron(sp.eye(nx, format="csr"), D1z, format="csr")  # (nx blocks)

    # --- Dx: for each row i, differences between columns j and j+1
    ex = np.ones(nx)
    D1x = sp.diags([-ex, ex], [0, 1], shape=(nx-1, nx), format="csr")
    Dx = sp.kron(D1x, sp.eye(nz, format="csr"), format="csr")

    return Dz, Dx

def classicloop(vini,L2,k1d,k2d,beta,niter,nx,ny,nd,obst,ifresid=False):
    """
    Inversion loop for classic Tikhonov regularization using conjugate gradient solver.

    Parameters:
    vini : initial velocity model
        Initial model parameters.
    L2 : ray path matrix
        Linear operator for the inversion.
    k1d : sparse matrix
        First-order difference operator along z.
    k2d : sparse matrix
        First-order difference operator along x.
    beta : float
        Regularization parameter.
    niter : int
        Number of iterations for the conjugate gradient solver.
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    nd : int
        Total  number of ray paths.
    obst : 1D array
        Observed travel times for each source-receiver pair.

    Returns
    -------
    v_est : estimated velocity model
        Estimated velocity model after inversion.
    """

    L2=L2.toarray();

    slowness0=1./vini.flatten(order='C');
    L2_cl=np.concatenate([L2, beta*k1d, beta*k2d], axis=0)         #########linear operator for classic inversion
    d0pad=np.matmul(L2_cl,slowness0) 	 

    dpad=np.zeros(L2_cl.shape[0]-nd)
    nm=L2_cl.shape[1]  ####number of model parameters

    ndc=nd+nx*(ny-1)+(nx-1)*ny   ##############number of neurons in the ray path matrix including regularization term

    ddpad=np.hstack([obst,dpad]) ###padded traveltime for linear inversion
    
    mm=np.zeros(nm) ###initial guess for the model parameters

    par_L_c={'matrix': L2_cl, 'nm': nm, 'nd': ndc};
    par_sol={'verb':0};

    if ifresid:
        grad,_,err=solver(matrix_lop,cgstep,nm,ndc,mm,ddpad-d0pad,niter,par_L_c,par_sol,ifres=1)
    
    else:
        grad,_=solver(matrix_lop,cgstep,nm,ndc,mm,ddpad-d0pad,niter,par_L_c,par_sol)

    alpha=stepsize(ddpad,0.1,grad,slowness0,L2_cl) ###estimated step size for each epoch

    mm2=slowness0+alpha*grad ### model update

    v_est=1/mm2.reshape(ny, nx,order="C");

    if ifresid:
        return v_est, err	
    
    else:
        return v_est	

   

def stepsize(obs,step1,grad,m0,L2):
    # step size estimation for each epoch
	t0=np.matmul(L2,m0)
	m0=m0+step1*grad
	t1=np.matmul(L2,m0)
	return step1*np.sum((t0-obs)*(t1-obs))/np.sum((t1-obs)**2)


def msecal(d1,d2):
    ####calculate the MSE between two vectors
	mse = np.mean((d1 - d2)**2)
	return mse 


def tomoloop(vini,L2,niter,nx,ny,nd,obst,par_S,ifresid=False):
    L2=L2.toarray()

    slowness0=1./vini.flatten(order='C');
    d0=np.matmul(L2,slowness0) 	#L*s0=d0
    nm=L2.shape[1]  ####number of model parameters

    par_L={'matrix': L2, 'nm': nm, 'nd': nd}
    mm=np.zeros(nm);
	
    if ifresid:
        grad,err=conjgrad(None,matrix_lop,trianglen_lop, mm, None, obst-d0, 1, 0.000001, niter,0,[],par_L,par_S,0,ifres=1)
    else:
        grad=conjgrad(None,matrix_lop,trianglen_lop, mm, None, obst-d0, 1, 0.000001, niter,0,[],par_L,par_S,0)

    alpha=stepsize(obst,0.1,grad,slowness0,L2)
    mm2=slowness0+alpha*grad
    
    v_est=1/mm2.reshape(ny, nx, order="C");
    if ifresid:
        return v_est,err
    else:
        return v_est
    

#############functions for 3D ray tracing and ray path matrix building
     
def compute_travel_times3D(v_true, sources, receivers, dx, dy, dz, nx, ny, nz, stepa=1,ifwav=False):
    """
    Compute travel times and ray paths for a set of sources and receivers in a 3D velocity model.
    based on the Fast Marching Method (FMM) and eikonal euqation.
    
    Parameters:
    v_true : 3D array
        The true velocity model (shape: nx x ny).
    sources : list of tuples
        List of source coordinates [(sx1, sy1), (sx2, sy2), ...].
    receivers : list of tuples
        List of receiver coordinates [(rx1, ry1), (rx2, ry2), ...].
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    dz : float
        Grid spacing in the z-direction.
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    nz : int
        Number of grid points in the z-direction.
    stepa : float
        step size for ray traing (0~1)
        when velocity is accuract stepa=1
        initially, a smaller step size help to improve accuracy
    ifwav : bool, optional
        If True, output traveltime field for the last source (default is False).

    
    Returns:
    travel_times : 1D array
        Computed travel times for each source-receiver pair.
    ray_paths : list of tuples
        List of ray paths for each source-receiver pair.
    """
    
    travel_times = []
    ray_paths = []  

    for sx, sy, sz in sources:
    # FMM travel-time field from this source
   
        t=eikonal(v_true.flatten(order='F'),xyz=np.array([sx,sy,sz]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
        T=t.reshape(nx,ny,nz,order='C').T;#be careful here #first axis (vertical) is x, second is z, third is y 
       
        for rx, ry, rz in receivers:

            t1=extract(T,[rx,ry,rz],ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz]) 
            #A: almost the same as option below 
            # travel_times.append(T[round(rx/dx), round(ry/dy), round(rz/dz)])  # FMM travel time at receiver
            travel_times.append(t1)
	
            paths=ray3d2(T,np.array([sx,sy,sz]),np.array([rx,ry,rz]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],step=stepa)
            ray_paths.append((paths.transpose()[:,0],paths.transpose()[:,1],paths.transpose()[:,2]))
    travel_times = np.array(travel_times)

    if ifwav:
        return travel_times, ray_paths, T
    else:
        return travel_times, ray_paths


def build_ray_matrix3D(ray_paths, dx, dy,dz, nx, ny, nz, ifraycorr=True):
    """
    Build the ray matrix A for a set of ray paths in a 3D grid.
    
    Parameters:
    ray_paths : list of tuples
        List of ray paths for each source-receiver pair.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    dz : float
        Grid spacing in the z-direction.
 
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    nz : int
        Number of grid points in the z-direction.
    ifraycorr : bool, optional
        If True, conduct ray path contribution correction
    
    Returns:
    A : sparse matrix
        The ray matrix (shape: n_rays x n_cells).
    """
    
    n_rays = len(ray_paths)
    n_cells = nx*ny*nz
    A = lil_matrix((n_rays, n_cells)) #A is a sparse matrix

    for i, (ray_x, ray_y, ray_z) in enumerate(ray_paths):
        for x, y, z in zip(ray_x, ray_y, ray_z):
            ix, iy, iz = int(round(x/dx)), int(round(y/dy)), int(round(z/dz))
            idx = iz*nx*ny + iy*nx + ix
            A[i, idx] += 1.0*dx  # Each cell contributes 1*dx to travel path
    
    if ifraycorr:
        A[(A>1.0*dx*2)]=1.0*dx #This is very important  
    
    return A  


def tomoloop3D(vini,L2,niter,nx,ny,nz,nd,obst,par_S,ifresid=False):
    
    L2=L2.toarray()

    slowness0=1./vini.flatten(order='C');
    d0=np.matmul(L2,slowness0) 	#L*s0=d0
    nm=L2.shape[1]  ####number of model parameters

    par_L={'matrix': L2, 'nm': nm, 'nd': nd}
    mm=np.zeros(nm);
	
    if ifresid:
        grad,err=conjgrad(None,matrix_lop,trianglen_lop, mm, None, obst-d0, 1, 0.000001, niter,0,[],par_L,par_S,0,ifres=1)
    else:
        grad=conjgrad(None,matrix_lop,trianglen_lop, mm, None, obst-d0, 1, 0.000001, niter,0,[],par_L,par_S,0)

    alpha=stepsize(obst,0.1,grad,slowness0,L2)
    mm2=slowness0+alpha*grad
    
    v_est=1/mm2.reshape(nx, ny,nz, order="F");
    if ifresid:
        return v_est,err
    else:
        return v_est
	
##########functions for ASVGD
def generate_random_field_perturbations(nz,nx,nump, distribution='random', min_var=1, max_var=100, seed=20170590, len_scale=5, nu=2):
    """
    Generates random field perturbations for a set of particles based on a variance of an specified distribution.
    
    Parameters:
        nz (int): model size along z axis
        nx (int): model size along x axis
        nump (int): number of particles used for inversion
    
        distribution (str): Type of distribution to sample variances ('random' or 'uniform').
        min_var (float): Minimum variance for perturbation sampling.
        max_var (float): Maximum variance for perturbation sampling.
        seed (int): Seed for random number generator.
        len_scale (float): Length scale for the Matern function.
        nu (float): Smoothness parameter for the Matern function.
        
    
    Returns:
        nmpy: perturbations to be added, on the specified device.
    """
    num_particles = nump
    # nz, nx = cfg.params.nz, cfg.params.nx
    zax = np.arange(0, nz)
    xax = np.arange(0, nx)

    # zax = torch.arange(0, nz)
    # xax = torch.arange(0, nx)

    # Sample variances based on the user-defined distribution
    if distribution == 'random':
        variances = np.random.rand(num_particles) * (max_var - min_var) + min_var
    elif distribution == 'uniform':
        variances=np.random.uniform(min_var,max_var,num_particles)
    else:
        raise ValueError("Unsupported distribution type. Use 'random' or 'uniform'.")

    variances = np.random.permutation(variances)  # Shuffle the variances

    # Setup random fields with gstools
    grf_seed = gs.random.MasterRNG(seed)
    # grf = torch.zeros(num_particles, nz * nx)
    grf = np.zeros((num_particles, nz * nx))

    for i in range(num_particles):
        rf = gs.Matern(dim=2, var=variances[i], len_scale=len_scale, nu=nu)
        srf = gs.SRF(rf, seed=grf_seed())
        srf.set_pos([zax, xax], "structured")
        # grf[i, :] = torch.from_numpy(srf().reshape(1, nz * nx)).float()
        grf[i, :] = srf().reshape(1, nz * nx).astype(np.float32)

    return grf, variances

def compute_gradient(
    model,
    sources,
    receivers,
    ny,
    nx,
    dy,
    dx,
    data_true,
    stepray,
    par_S,
    initer,
    nd
):
    """
    Compute the gradient of the loss function

    Args:
        model: velocity model
        sources: source coordinates
        receivers: receiver coordinates
        ny: number of grid points in y direction
        nx: number of grid points in x direction
        dy: spatial sampling rate in y direction
        dx: spatial sampling rate in x direction
        data_true: observed data
        stepray: step length of ray tracing
    Returns:
        gradient of the loss function
    """
    
    model = (
        model.reshape(ny, nx) if len(model.shape) == 1 else model
    )
   
    grad_loss = np.zeros_like(model)
    

    _,ray_pathsini=compute_travel_times(model, sources, receivers, dx, dy, nx, ny, stepray)   ###ray traceing based on the given model
    opsr=build_ray_matrix(ray_pathsini, dx, dy, nx, ny)  ####build ray path matrix using the initial model
	
    grad_loss=tomoloopgrad(model,opsr,initer,nx,ny,nd,data_true,par_S) ### gradient estimation and model update
    
    running_loss=msecal(data_true,np.matmul(opsr.toarray(),1./model.flatten(order='C')))

    

    return running_loss, grad_loss

def tomoloopgrad(vini,L2,niter,nx,ny,nd,obst,par_S,ifresid=False):
    L2=L2.toarray()

    slowness0=1./vini.flatten(order='C');
    d0=np.matmul(L2,slowness0) 	#L*s0=d0
    nm=L2.shape[1]  ####number of model parameters

    par_L={'matrix': L2, 'nm': nm, 'nd': nd}
    mm=np.zeros(nm);
	
    if ifresid:
        grad,err=conjgrad(None,matrix_lop,trianglen_lop, mm, None, obst-d0, 1, 0.000001, niter,0,[],par_L,par_S,0,ifres=1)
    else:
        grad=conjgrad(None,matrix_lop,trianglen_lop, mm, None, obst-d0, 1, 0.000001, niter,0,[],par_L,par_S,0)

    
    if ifresid:
        return grad,err
    else:
        return grad


class RBF(torch.nn.Module):
    """
        Initializes the RBF_FWI class.

        Args:
            sigma (float, optional): Bandwidht of the RBF kernel.
    """
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, EMA):
        d = X[:, None, :] - X[None, :, :]
        dists = (d**2).sum(axis=-1)

        # print(d.shape)
        # print(d.max(),d.min())
        # print(self.sigma)
        # print(dists.shape)
        # print(dists.max(),dists.min())
        # print((2 * math.log(X.size(0) + 1)))
        # print(torch.median(dists))

        if self.sigma is None:
            h = torch.median(dists) / (2 * math.log(X.size(0) + 1))
            sigma = math.sqrt(h)
            # print(h)
            # print(sigma)
            if EMA is not None:
                # EMA[0] is the previous sigma, EMA[1] is the smooth constant
                sigma = EMA[1] * sigma + (1 - EMA[1]) * EMA[0]
        
        else:
            sigma = self.sigma

        k = torch.exp(-dists / sigma**2 / 2)
        
        der = (d * k[:, :, None]).sum(axis=0) / sigma**2
        
    
        return k, der, sigma 
    
# class RBF:
#     """
#     RBF kernel for SVGD (NumPy version)

#     Args:
#         sigma (float, optional): Bandwidth of the RBF kernel.
#     """

#     def __init__(self, sigma=None):
#         self.sigma = sigma

#     def __call__(self, X, EMA=None):
#         """
#         Args:
#             X: (N, D) numpy array
#             EMA: None or [previous_sigma, momentum]

#         Returns:
#             k:     (N, N) RBF kernel matrix
#             der:   (N, D) kernel gradient
#             sigma: bandwidth
#         """

#         # Pairwise differences: (N, N, D)
#         d = X[:, None, :] - X[None, :, :]

#         # Pairwise squared distances: (N, N)
#         dists = np.sum(d ** 2, axis=-1)

#         # Median trick
#         if self.sigma is None:
#             h = np.median(dists) / (2.0 * math.log(X.shape[0] + 1))
#             sigma = math.sqrt(h)

#             if EMA is not None:
#                 # EMA = [previous_sigma, momentum]
#                 sigma = EMA[1] * sigma + (1.0 - EMA[1]) * EMA[0]
#         else:
#             sigma = self.sigma

#         # RBF kernel
#         k = np.exp(-dists / (2.0 * sigma ** 2))

#         # Kernel gradient
#         der = np.sum(d * k[:, :, None], axis=0) / (sigma ** 2)

#         return k, der, sigma
    
# def alpha_tanh(total_epochs, p=2):
#     """
#     Generate an alpha array using a tanh function across epochs.

#     Parameters
#     ----------
#     total_epochs : int
#         Total number of epochs.
#     p : float, optional
#         Power parameter. Default is 2.

#     Returns
#     -------
#     numpy.ndarray
#         Alpha values with shape (total_epochs,).
#     """
#     t = np.arange(1, total_epochs + 1, dtype=np.float32)
#     T = float(total_epochs)

#     alpha = np.tanh((2.0 * (t / T)) ** p)

#     return alpha

def alpha_tanh(total_epochs, p=2, device='cpu'):
    """
    Generate an alpha array using a tanh function across epochs.

    Parameters:
    total_epochs (int): Total number of epochs for the simulation.
    p (float): Power parameter for the alpha calculation. Default is 1.
    device (str): The torch device on which the alpha array will be generated. Default is 'cpu'.
    
    Using a factor of 2.0 to allow more epochs with alpha = 1.0
    Returns:
    torch.Tensor: An array of alpha values for each epoch.
    """
    t = torch.arange(1, total_epochs + 1, device=device)  # Epochs start from 1 to total_epochs
    T = total_epochs
    alpha = torch.tanh((2.0 * (t.float() / T)) ** p)
    
    return alpha

def compute_gradient_per_batch(model, grad_func):
    log_p = 0.0
    fwi_grad = np.zeros_like(model)
    for i, m in enumerate(model):
        # print(i)
        loss, grad_m = grad_func(m)
        # fwi_grad[i] = grad_m.ravel()
        fwi_grad[i] = grad_m

        log_p += loss
    log_p /= len(model)
    return log_p, fwi_grad

def compute_max_gradient_per_batch(grad):
    assert len(grad.shape) == 2
    return np.max(np.abs(grad), axis=1, keepdims=True)

import numpy as np


# class SVGD:
#     def __init__(
#         self,
#         particles,
#         K,
#         alpha,
#         learning_rate=1e-3
#     ):
#         """
#         NumPy implementation of SVGD.

#         Parameters
#         ----------
#         particles : np.ndarray
#             Initial particles with shape (num_particles, num_parameters).
#         K : callable
#             Kernel function, such as RBF. It should return:
#             K_XX, derivative, sigma.
#         alpha : np.ndarray
#             Weight schedule for the driving force.
#         learning_rate : float
#             Particle update learning rate.
#         scheduler : callable, optional
#             Learning-rate scheduler. It should accept the current epoch
#             and learning rate and return a new learning rate.
#         """
#         self.particles = np.asarray(particles, dtype=np.float32)
#         self.K = K
#         self.alpha = np.asarray(alpha)
#         self.learning_rate = learning_rate
    

#         # Variables saved for later inspection
#         self.log_grad_p = None
#         self.sigma = None
#         self.K_XX = None
#         self.gradients = None
#         self.driving_force = None
#         self.repulsive_force = None

#     def phi(self, particles, log_grad_p, epoch, EMA=None):
#         """
#         Compute the Stein-gradient update direction.

#         Parameters
#         ----------
#         particles : np.ndarray
#             Shape: (num_particles, num_parameters).
#         log_grad_p : np.ndarray
#             Gradient of the log posterior for each particle.
#         epoch : int
#             Current epoch index.
#         EMA : optional
#             EMA information passed to the kernel.

#         Returns
#         -------
#         np.ndarray
#             SVGD direction with the same shape as particles.
#         """
#         X = np.asarray(particles)
#         self.log_grad_p = np.asarray(log_grad_p)

#         self.K_XX, der, self.sigma = self.K(X, EMA)

#         # Equivalent to:
        
#         self.driving_force = self.K_XX @ self.log_grad_p

#         self.repulsive_force = der

#         alpha_iter = self.alpha[epoch]

#         phi = (
#             alpha_iter * self.driving_force
#             - self.repulsive_force
#         )

#         return phi

#     def step(
#         self,
#         X,
#         log_grad_p,
#         m_vmin,
#         m_vmax,
#         epoch,
#         gmax=1.0,
#         EMA=None
#     ):
#         """
#         Perform one SVGD update and constrain particles to model bounds.

#         Notes
#         -----
#         The original PyTorch implementation uses:

#             X.grad = phi / gmax
#             optimizer.step()

#         Since a PyTorch optimizer performs gradient descent, the equivalent
#         NumPy update is:

#             X -= learning_rate * phi / gmax
#         """
#         X = np.asarray(X)

#         if gmax == 0:
#             raise ValueError("gmax must be non-zero.")

#         self.gradients = self.phi(
#             X,
#             log_grad_p,
#             epoch,
#             EMA
#         ) / gmax

#         # Preserve the behavior of optimizer.step()
#         X -= self.learning_rate * self.gradients

#         # Equivalent to torch.clamp_
#         np.clip(
#             X,
#             m_vmin,
#             m_vmax * 1.15,
#             out=X
#         )


#         self.particles = X

#         return X

class SVGD:
    def __init__(self, particles, K, alpha, optimizer,  device='cuda'):
        """
        Simulate SVGD for n number of particles

        gmax: max FWI gradient of the initial particles (numpy.ndarray or torch.Tensor)
        K: Kernel function (e.g., RBF or IMQ) taking two arguments and returning a matrix
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler (optional)
        compute_gradient_per_batch: Function to compute FWI gradient per batches
        grad_func: Gradient of each FWI modeling
        filter_func: Function to filter the gradient
        nz: Model dimensions in z
        nx: Model dimensions in x
        device: The torch device on which computations will be performed
        """
        self.particles = particles
        self.K = K
        self.alpha = alpha
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.device = device
    

        # Variables to retrieve later
        self.log_grad_p = None
        self.sigma = None
        self.K_XX = None
        self.gradients = None
        self.driving_force = None
        self.repulsive_force = None

    def phi(self, particles, log_grad_p, epoch, EMA):
        """
        Compute the Stein Gradient. The terms inside the square bracket above.

        X: n number of models (particles)
        """
        X = particles.detach().requires_grad_(True)
        X = particles.clone()

        # print(X.min(),X.max())
        
        self.log_grad_p = log_grad_p
        self.K_XX, der, self.sigma = self.K(X, EMA)

        # print(der.min(),der.max())
        # print(log_grad_p.min(),log_grad_p.max())
    
        self.driving_force = self.K_XX.mm(self.log_grad_p)
        self.repulsive_force = der

        # print(self.driving_force.min(),self.driving_force.max())
        # print(self.repulsive_force.min(),self.repulsive_force.max())
        
        alpha_iter = self.alpha[epoch]
        
        phi = alpha_iter*self.driving_force - self.repulsive_force
        
        return phi

    def step(self, X, log_grad_p, m_vmin, m_vmax, epoch, gmax=1, EMA=None):
        """
        Bound model to the limits
        m_vmin: minimum model value
        m_vmax: maximum model value
        """
        self.optimizer.zero_grad()
        

        X.grad = self.phi(X, log_grad_p, epoch, EMA)/gmax

        
        


        self.optimizer.step()

        

        with torch.no_grad():
            X.clamp_(m_vmin, m_vmax)


def add_noise(signal, var=1e-3, seed=None):
    """
    Add white Gaussian noise to achieve the desired SNR.

    Parameters
    ----------
    signal : np.ndarray
        Original signal.
    var : float
        variance for noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    noisy_signal : np.ndarray
    noise : np.ndarray
    """
    if seed is not None:
        np.random.seed(seed)

    signal = np.asarray(signal)



    # Gaussian noise
    noise = np.random.normal(
        0,
        var,
        size=signal.shape
    )

    noisy_signal = signal + noise

    return noisy_signal, noise


def calculate_snr(signal, noise):
    """
    Calculate SNR (dB).
    """
    signal = np.asarray(signal)
    noise = np.asarray(noise)

    

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    snr = 10 * np.log10(signal_power / noise_power)

    return snr
