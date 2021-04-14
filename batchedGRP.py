#!/usr/bin/python3

import itertools
import math
import os
import pickle
import sys

import graphdot
from graphdot import Graph
from graphdot.linalg.cholesky import CholSolver
from graphdot.linalg.spectral import pinvh
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.fix import Normalization
from graphdot.kernel.marginalized.starting_probability import Uniform
from graphdot.microkernel import (
    TensorProduct,
    KroneckerDelta,
    SquareExponential,
    Constant
)
import numpy as np
from scipy.optimize import minimize

class BatchedGaussianProcessRegressor():
    def __init__(self, active=True, chunk_size=None, alpha=1e-3):
        if chunk_size == None:
            print('Must provide a chunk size')
            sys.exit()
        self.active = active
        self.chunk_size = math.ceil(chunk_size/2)
        self.alpha = alpha
        self.xg = None
        self.xm = None
        self.yg = None
        self.ym = None
        self.kernel = None
        self.theta = None
        self.success = None
    
    def make_kernel(self, h_element, l_scale, q, p):
        return Normalization(
            MarginalizedGraphKernel(
                node_kernel=TensorProduct(
                    element=KroneckerDelta(h=h_element),
                ),
                edge_kernel=TensorProduct(
                    length=SquareExponential(length_scale=l_scale)
                ),
                p=Uniform(p),
                q=q
            )
        )
    
    def batches(self):
        assert(self.xg is not None)
        assert(self.xm is not None)
        chs = self.chunk_size

        if self.yg is None:
            assert(self.ym is None)
            rows = self.xg
            cols = self.xg
            
            rmask = self.xm
            colmask = self.xm
        else:
            rows = self.yg
            cols = self.xg
            
            rmask = self.ym
            colmask = self.xm
        
        assert(len(rows) == rmask.shape[0])
        assert(len(cols) == colmask.shape[0])
        
        running_i = 0
        running_j = 0
        for i in range(0, len(rows), chs):
            gx = list()
            ix = list()
            xcs = list()
            if i+chs > len(rows):
                gx = rows[i:]
                ix = list(range(i,len(rows)))
            else:
                gx = rows[i:i+chs]
                ix = list(range(i,i+chs))
            
            for ind, gi in zip(ix, gx):
                num_nodes = len(gi.nodes['!i'])
                xcs.append(rmask[ind][:num_nodes])

            xcs = np.ma.concatenate(xcs)

            if self.active: running_i += xcs.compressed().shape[0]
            else:           running_i += xcs.shape[0]

            for j in range(0, len(cols), chs):
                if j == 0: running_j = 0

                gy = list()
                iy = list()
                ycs = list()
                if j+chs > len(cols):
                    gy = cols[j:]
                    iy = list(range(j,len(cols)))
                else:
                    gy = cols[j:j+chs]
                    iy = list(range(j,j+chs))
                
                for ind, gi in zip(iy, gy):
                    num_nodes = len(gi.nodes['!i'])
                    ycs.append(colmask[ind][:num_nodes])

                ycs = np.ma.concatenate(ycs)

                if self.active: running_j += ycs.compressed().shape[0]
                else:           running_j += ycs.shape[0]

                yield gx, gy, xcs, ycs, running_i, running_j
    
    def invert(self, m):
        try:
            return CholSolver(m), np.prod(np.linalg.slogdet(K))
        except:
            try:
                return pinvh(m, rcond=1e-8, mode='clamp', return_nlogdet=True)
            except:
                raise Exception('Matrix inverse not work')
    

    def batched_kernel_call(self):
        assert(self.xg is not None)
        assert(self.xm is not None)
        assert(self.kernel is not None)

        if self.active:
            if self.ym is None:
                size = self.xm.compressed().shape[0]
                K = np.zeros((size, size))
            else:
                sx = self.ym.compressed().shape[0]
                sy = self.xm.compressed().shape[0]
                K = np.zeros((sx, sy))
        else:
            if self.ym is None:
                size = self.xm.shape[0]
                K = np.zeros((size, size))
            else:
                sx = self.ym.shape[0]
                sy = self.xm.shape[0]
                K = np.zeros((sx, sy))
        
        start = True
        bi = 0
        bj = 0
        pi = None
        pj = None

        for x, y, xs, ys, ri, rj in self.batches():
            if self.active:
                ma = np.array([x[0] and x[1] for x in itertools.product(~xs.mask, ~ys.mask)])
                ma = ma.reshape(xs.shape[0],ys.shape[0])
                K_sub = self.kernel(x,y,nodal=True)
                K_sub = K_sub[ma].reshape(xs.compressed().shape[0],ys.compressed().shape[0])
            else:
                K_sub = self.kernel(x,y,nodal=True)
            
            if pi is None and pj is None:
                K[:ri,:rj] = K_sub
                pj = rj
                bj = pj
                pi = ri
                continue
            
            if rj <= pj: 
                bj = 0
                bi = pi
            
            K[bi:ri,bj:rj] = K_sub
            pj = rj
            pi = ri
            bj = pj
        
        return K
    
    def opt_kernel(self, pt):
        h_elm, ls, q, p = pt
        print(pt)

        self.kernel = self.make_kernel(h_elm, ls, q, p)
        
        K = self.batched_kernel_call()
        #Regularize
        d = np.diag(K)*(1+self.alpha)
        np.fill_diagonal(K, d)
        # Invert
        Kinv, logdet = self.invert(K)
        # Make Ky
        Ky = Kinv @ self.cs
        # log marginal likelihood
        yKy = self.cs @ Ky
        retval = yKy + logdet
        
        return retval
    
    def fit(self, xg, xm, tol=1e-1):
        self.xg = xg
        self.xm = xm
        self.tol=tol

        self.cs = self.xm.compressed()
        self.csmean = self.xm.mean()
        self.csstd = self.xm.std()

        self.cs = (self.cs - self.csmean) / self.csstd
        # p bounds 1 and 50
        cons = ({'type':'ineq', 'fun':lambda x: x[0] - 1e-7},
                {'type':'ineq', 'fun':lambda x: 0.90 - x[0]},
                {'type':'ineq', 'fun':lambda x: x[1] - 1e-7},
                {'type':'ineq', 'fun':lambda x: 0.90 - x[1]},
                {'type':'ineq', 'fun':lambda x: x[2] - 1e-7},
                {'type':'ineq', 'fun':lambda x: 0.90 - x[2]},
                {'type':'ineq', 'fun':lambda x: x[3] - 1.0},
                {'type':'ineq', 'fun':lambda x: 50.0 - x[3]})
                
        opt_results = minimize(self.opt_kernel,
                               x0=[0.05, 0.05, 0.05, 10.0],
                               method='COBYLA',
                               tol=self.tol,
                               constraints=cons,
                               options={
                                   'rhobeg':1e-2,
                                   'maxiter':5000,
                                   'disp':True,
                                   'catol':0.0
                               })
        
        self.success = opt_results.success
        if opt_results.success:
            print(opt_results)
            helm, lscale, q, p = opt_results.x

            self.kernel = self.make_kernel(helm, lscale, q, p)
            self.theta = (helm, lscale, q, p)
            K = self.batched_kernel_call()
            # Regularize
            d = np.diag(K)*(1+self.alpha)
            np.fill_diagonal(K, d)
            # Invert
            Kinv, logdet = self.invert(K)
            # Make Ky
            self.Ky = Kinv @ self.cs
        else:
            print(opt_results)
    
    def predict(self, yg, ym, alpha=None):
        self.yg = yg
        self.ym = ym

        Kg = self.batched_kernel_call()
        if alpha != None:
            d = np.diag(Kg)*(1+alpha)
            np.fill_diagonal(Kg, d)
        
        preds = (Kg @ self.Ky)*self.csstd + self.csmean
        return preds
    
    def save(self, path, filename='model.pickle', overwrite=True):
    	f_model = os.path.join(path, filename)
    	if os.path.isfile(f_model) and not overwrite:
    		raise RuntimeError(
    			f'Path {f_model} already exists. To overwrite, set '
    			' `overwrite=True`.'
    		)
    	assert(self.theta != None)
    	store = self.__dict__.copy()
    	store.pop('kernel', None)
    	pickle.dump(store, open(f_model, 'wb'), protocol=4)
    
    def load(self, path, filename='model.pickle'):
    	f_model = os.path.join(path, filename)
    	if not os.path.isfile(f_model):
    		raise RuntimeError(
    			f'Path {f_model} does not exist. Provide path to a pickled'
    			' file.'
    		)
    	store = pickle.load(open(f_model, 'rb'))
    	self.__dict__.update(**store)
    	
    	helm, lscale, q, p = self.theta
    	self.kernel = self.make_kernel(helm, lscale, q, p)
