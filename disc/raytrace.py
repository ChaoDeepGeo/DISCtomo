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
from collections import defaultdict
import pyekfmm as fmm

def compute_travel_times(v_true, sources, receivers, dx, dy, nx, ny, stepa=1,ifwav=False):
    """
    Compute travel times and ray paths for a set of sources and receivers in a 2D velocity model.
    based on the Fast Marching Method (FMM) and eikonal euqation (crosswell geometry).
    
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
    # print(alpha)
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
        # T=t.reshape(nx,ny,nz,order='C').T;#be careful here #first axis (vertical) is x, second is z, third is y 
        T=t.reshape(nx,ny,nz,order='F')
       
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


def build_ray_matrix3D(ray_paths, dx, dy,dz, nx, ny, nz, numerr=2.0, ifraycorr=True):
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
    numerr : float
            scale of grid for sensivity correction
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
        A[(A>1.0*dx*numerr)]=1.0*dx #This is very important  
    
    return A  




def tomoloop3D(vini,L2,niter,nx,ny,nz,nd,obst,par_S,ifresid=False):
    
    L2=L2.toarray()

    slowness0=1./vini.flatten(order='F');
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
    
    v_est=1/mm2.reshape(nx,ny,nz, order="C"); ###(z,y,x)
    # v_est=np.transpose(v_est, (2,1,0));
    if ifresid:
        return v_est,err
    else:
        return v_est





def stream2d_continent(u,v, sx, sy, step=0.1, maxvert=10000):
	"""
	stream2d: draw 2D stream lines along the steepest descent direction
	
	INPUT
	u:   	derivative of traveltime in x
	v:   	derivative of traveltime in z
	
	OUTPUT  
	
	Copyright (C) 2023 The University of Texas at Austin
	Copyright (C) 2023 Yangkang Chen
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published
	by the Free Software Foundation, either version 3 of the License, or
	any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details: http://www.gnu.org/licenses/
	
	References   
	[1] Chen et al.
		
	 DEMO
	 demos/test_xxx.py
	"""
	
	xSize=u.shape[1]
	ySize=u.shape[0]
	
	[verts, numverts] = traceStreamUV_continent  (u.flatten(order='F'),v.flatten(order='F'),xSize, ySize, sx, sy, step, maxvert);
	verts=verts.reshape(2,numverts,order='F');
	
	return verts,numverts

def traceStreamUV_continent (ugrid, vgrid, xdim, ydim, sx, sy, step, maxvert):
	"""
	traceStreamUV: 2D streamline
	
	INPUT
	ugrid:   	derivative of traveltime in x
	vgrid:   	derivative of traveltime in z
	
	OUTPUT  
	
	Copyright (C) 2023 The University of Texas at Austin
	Copyright (C) 2023 Yangkang Chen
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published
	by the Free Software Foundation, either version 3 of the License, or
	any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details: http://www.gnu.org/licenses/
	
	References   
	[1] Chen et al., 2023
		
	 DEMO
	 demos/test_xxx.py
	"""
	numverts=0;
	x=sx-1;y=sy-1;
	verts=np.zeros([2*maxvert,1])
	while 1:
		if (x<0 or x>xdim-1 or y<0 or y>ydim-1 or numverts>=maxvert) : 
# 			print("First break");
			break;
		
		ix=int(np.floor(x))
		iy=int(np.floor(y))
		
		if ix == xdim-1:
			ix=ix-1;
			
		if iy == ydim-1:
			iy=iy-1;
			
		xfrac=x-ix;
		yfrac=y-iy;
		
		#weights for linear interpolation
		a=(1-xfrac)*(1-yfrac);
		b=(  xfrac)*(1-yfrac);
		c=(1-xfrac)*(  yfrac);
		d=(  xfrac)*(  yfrac);
		
		verts[2*numverts + 0] = x+1;
		verts[2*numverts + 1] = y+1;
		
		#if already been here, done
		if numverts>=2:
			if verts[2*numverts] == verts[2*(numverts-2)] and verts[2*numverts+1] == verts[2*(numverts-2)+1]:
				numverts=numverts+1;
# 				print("Second break");
				break;
		
		numverts=numverts+1;
		ui = ugrid[iy  +ydim*ix]*a + ugrid[iy  +ydim*(ix+1)]*b + ugrid[iy+1+ydim*ix]*c + ugrid[iy+1+ydim*(ix+1)]*d;
		vi = vgrid[iy  +ydim*ix]*a + vgrid[iy  +ydim*(ix+1)]*b + vgrid[iy+1+ydim*ix]*c + vgrid[iy+1+ydim*(ix+1)]*d;
		
		#calculate step size, if 0, done
		if abs(ui) > abs(vi):
			imax=abs(ui);
		else:
			imax=abs(vi);
			
		if imax==0:
			print("Third break");
			break;
		
		imax=step/imax;
		ui = ui*imax;
		vi = vi*imax;
		
		#update the current position
		x = x+ui;
		y = y+vi;
	
# 	print('numverts',numverts)
	verts=verts[0:2*numverts]
	
	return verts,numverts


def trimrays_continent(paths, start_points, T=None):
	"""
	trimrays: trim rays (remove very close ray points around the source)
	
	%paths: [2 x ngrid]
	%start_points: [2 x 1], e.g., start_points=np.array([1,1])
	%T: threshold, e.g., dx,dz
	
	"""
	ngrid=paths.shape[1]
	start_points=np.expand_dims(start_points,1);
	
	d=np.sqrt(np.sum(np.power(paths-np.repeat(start_points,ngrid,axis=1),2),0));
	
	if T is None:
		T = d.max()/300/300;
		
	I=np.where(d<T)
	
	paths=np.delete(paths,I,1)
	
	paths=np.concatenate((paths,start_points),axis=1)
	
	return paths


##############traveltime estimation using spherical axis
from scipy.interpolate import RectBivariateSpline
def compute_travel_times_sr(v_true, sources, receivers,dx, dy, nx, ny):
    
    
    travel_times = []
    ray_paths = []  

    # x=[(x0+i*dx) for i in range(nx)]
    # y=[(y0+i*dy) for i in range(ny)]

    for source, receiver in zip(sources, receivers):

        sx, sy = source
        rx, ry = receiver
        # FMM travel-time field from this source
        # t=fmm.eikonal_rtp(v_true.flatten(order='F'),rtp=np.array([6371,sy+90,sx+180]),ar=[6371,6371,1],at=[y0+90,dy,ny],ap=[x0+180,dx,nx],order=2); 
        
        t=eikonal(v_true.transpose().flatten(order='F'),xyz=np.array([sx,0,sy]),ax=[0,dx,nx],ay=[0,dy,1],az=[0,dy,ny],order=2);
        T=t.reshape(nx,ny,order='F').T;  #first axis (vertical) is x, second is y
                        
        travel_times.append(T[round(ry/dy), round(rx/dx)])  # FMM travel time at receiver
            
        paths=ray2d2(T,np.array([sx,sy]),np.array([rx,ry]),ax=[0,dx,nx],ay=[0,dy,ny],step=1)
        ray_paths.append((paths.transpose()[:,0],paths.transpose()[:,1]))
            
    travel_times = np.array(travel_times)
    
    return travel_times, ray_paths


from typing import Sequence, Tuple
def lonlat_to_cartesian_grid(
    longitudes: Sequence[float],
    latitudes: Sequence[float],
    lon0: float,
    lat0: float,
    dx: float,
    dy: float,
    radius: float = 6371.0088,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert longitude/latitude coordinates to local Cartesian coordinates
    and corresponding grid indices.

    Parameters
    ----------
    longitudes, latitudes
        Station longitude and latitude in degrees.
    lon0, lat0
        Geographic coordinates corresponding to Cartesian origin (0, 0),
        in degrees.
    dx, dy
        Cartesian grid intervals in km.

    Returns
    -------
    x_km, y_km
        Local Cartesian coordinates in km.
        x is positive eastward and y is positive northward.
    ix, iy
        Nearest Cartesian grid-point indices.
    """
    lon = np.asarray(longitudes, dtype=np.float64)
    lat = np.asarray(latitudes, dtype=np.float64)

    if lon.shape != lat.shape:
        raise ValueError(
            f"Longitude and latitude shapes differ: "
            f"{lon.shape} versus {lat.shape}"
        )

    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive.")

  

    earth_radius_km = radius ###this is the real radius of the earth, used to convert lat/lon to km


    # Keep longitude differences in [-180, 180) degrees.
    delta_lon_deg = (lon - lon0 + 180.0) % 360.0 - 180.0
    delta_lat_deg = lat - lat0

    delta_lon_rad = np.deg2rad(delta_lon_deg)
    delta_lat_rad = np.deg2rad(delta_lat_deg)
    lat0_rad = np.deg2rad(lat0)

    x_km = earth_radius_km * np.cos(lat0_rad) * delta_lon_rad
    y_km = earth_radius_km * delta_lat_rad

    # Nearest grid-point indices.
    ix = np.rint(x_km / dx).astype(np.int64)
    iy = np.rint(y_km / dy).astype(np.int64)

    return x_km, y_km, ix, iy

def cartesian_to_lonlat(
    x_km: Sequence[float],
    y_km: Sequence[float],
    lon0: float,
    lat0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert local Cartesian coordinates in km back to longitude and latitude.

    Parameters
    ----------
    x_km, y_km
        Cartesian coordinates in km.
        Positive x points east and positive y points north.

    lon0, lat0
        Longitude and latitude corresponding to Cartesian origin (0, 0),
        in degrees.

    Returns
    -------
    longitudes, latitudes
        Geographic coordinates in degrees.
    """
    x_km = np.asarray(x_km, dtype=np.float64)
    y_km = np.asarray(y_km, dtype=np.float64)

    if x_km.shape != y_km.shape:
        raise ValueError(
            f"x_km and y_km must have the same shape, "
            f"but got {x_km.shape} and {y_km.shape}."
        )

    earth_radius_km = 6371.0088
    lat0_rad = np.deg2rad(lat0)

    if abs(np.cos(lat0_rad)) < 1e-12:
        raise ValueError(
            "Longitude conversion is unstable near the poles."
        )

    delta_lat_rad = y_km / earth_radius_km
    delta_lon_rad = (
        x_km /
        (earth_radius_km * np.cos(lat0_rad))
    )

    latitudes = lat0 + np.rad2deg(delta_lat_rad)
    longitudes = lon0 + np.rad2deg(delta_lon_rad)

    # Normalize longitude into [-180, 180)
    longitudes = (
        longitudes + 180.0
    ) % 360.0 - 180.0

    return longitudes, latitudes