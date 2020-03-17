# Implements the Cube with a spherical hole subject to uniform tension

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
import os
import scipy.io
from utils.gridPlot3D import getExactStresses
from utils.gridPlot3D import getExactTraction
from utils.gridPlot3D import refineElemVertex
from utils.gridPlot3D import energyError
from utils.gridPlot3D import scatterPlot
from utils.gridPlot3D import createFolder
from utils.gridPlot3D import energyPlot
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

from utils.Geom import Geometry3D
from utils.PINN_adaptive import Elasticity3D

class CubeWithHole(Geometry3D):
    '''
     Class for definining a quarter-annulus domain centered at the orgin
         (the domain is in the first quadrant)
     Input: rad_int, rad_ext - internal and external radii of the annulus
    '''
    def __init__(self):            
                
        geomData = dict()
        
        data = scipy.io.loadmat('cube_with_hole.mat')
        numCtrlPts = 5*5*2

        # Set degrees
        geomData['degree_u'] = data['vol1'][0][0][5][0][0] - 1
        geomData['degree_v'] = data['vol1'][0][0][5][0][1] - 1
        geomData['degree_w'] = data['vol1'][0][0][5][0][2] - 1
        
        # Set control points
        geomData['ctrlpts_size_u'] = np.int(data['vol1'][0][0][2][0][0])
        geomData['ctrlpts_size_v'] = np.int(data['vol1'][0][0][2][0][1])
        geomData['ctrlpts_size_w'] = np.int(data['vol1'][0][0][2][0][2])
        
        ctrlpts_x = data['vol1'][0][0][3][0,:].reshape(numCtrlPts, order='F')
        ctrlpts_y = data['vol1'][0][0][3][1,:].reshape(numCtrlPts, order='F')
        ctrlpts_z = data['vol1'][0][0][3][2,:].reshape(numCtrlPts, order='F')

        geomData['ctrlpts'] =  np.column_stack((ctrlpts_x, ctrlpts_y, ctrlpts_z))
        
        geomData['weights'] = data['vol1'][0][0][3][3,:].reshape(numCtrlPts, order='F')
        
        # Set knot vectors
        geomData['knotvector_u'] = [0., 0., 0., 0.5, 0.5, 1., 1., 1.]
        geomData['knotvector_v'] = [0., 0., 0., 0.5, 0.5, 1., 1., 1.]
        geomData['knotvector_w'] = [0., 0., 1., 1.]

        super().__init__(geomData)

class PINN_CWH(Elasticity3D):
    '''
    Class including (symmetry) boundary conditions for the hollow sphere problem
    '''       
    def net_uvw(self, x, y, z):

        X = tf.concat([x, y, z], 1)      

        uvw = self.neural_net(X, self.weights, self.biases)

        u = x*uvw[:, 0:1]
        v = y*uvw[:, 1:2]
        w = z*uvw[:, 2:3]

        return u, v, w

if __name__ == "__main__": 
    
    originalDir = os.getcwd()
    foldername = 'CubeWithHole'    
    createFolder('./'+ foldername + '/')    
    figHeight = 6
    figWidth = 6
    
    model_data = dict()
    model_data['E'] = 1e3
    model_data['nu'] = 0.3
    
    model = dict()
    model['E'] = model_data['E']
    model['nu'] = model_data['nu']
    model['radInt'] = 1.0
    model['lenCube'] = 4.0
    model['P'] = 1.0
    
    # Domain bounds
    model['lb'] = np.array([0., 0., 0.]) #lower bound of the plate
    model['ub'] = np.array([-model['lenCube'], model['lenCube'], model['lenCube']]) # Upper bound of the plate
    
    NN_param = dict()
    NN_param['layers'] = [3, 20, 20, 20, 3]
    NN_param['data_type'] = tf.float32
    
    #Generate interior Gauss points
    myDomain = CubeWithHole()
    numElemU = 20
    numElemV = 20
    numElemW = 20
    numGauss = 2
    
    vertex = myDomain.genElemList(numElemU, numElemV, numElemW)
    xPhys, yPhys, zPhys, wgtsPhys = myDomain.getElemIntPts(vertex, numGauss)
    X_f = np.concatenate((xPhys,yPhys,zPhys,wgtsPhys),axis=1)
    
    # Generate boundary Gauss points, normals and tractions
    numElemFace = [20, 20]
    numGaussFace = 2
    orientFace = 6
    xFace, yFace, zFace, xNorm, yNorm, zNorm, wgtsFace = myDomain.getQuadFacePts(numElemFace,
                                                                numGaussFace, orientFace)
    trac_x, trac_y, trac_z = getExactTraction(xFace, yFace, zFace, xNorm, yNorm, zNorm, model)
    X_bnd = np.concatenate((xFace, yFace, zFace, wgtsFace,  trac_x, trac_y, trac_z), axis=1)
    
    model_pts = dict()
    model_pts['X_int'] = X_f
    model_pts['X_bnd'] = X_bnd
    
    modelNN = PINN_CWH(model_data, model_pts, NN_param)
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    filename = 'Training_scatter'
    scatterPlot(X_f,X_bnd,figHeight,figWidth,filename)
    
    numSteps = 3
    num_train_its = 2000
    
    nPred = 50    
    withSides = [1, 1, 1, 1, 1, 1]
    xGrid, yGrid, zGrid = myDomain.getUnifIntPts(nPred, nPred, nPred, withSides)
    Grid = np.concatenate((xGrid, yGrid, zGrid), axis=1)
    
    for iStep in range(numSteps):
        
        start_time = time.time()
        modelNN.train(X_f, num_train_its)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        
        u_pred, v_pred, w_pred, energy_pred, sigma_x_pred, sigma_y_pred, sigma_z_pred, \
             sigma_xy_pred, sigma_yz_pred, sigma_zx_pred = modelNN.predict(Grid)
        
        filename = 'Cube_with_hole' 
        energyPlot(xGrid,yGrid,zGrid,nPred,u_pred,v_pred,w_pred,energy_pred,filename)        
        
        # Compute the L2 and energy norm errors using integration
        u_pred, v_pred, w_pred, energy_pred, sigma_x_pred, sigma_y_pred, sigma_z_pred, \
             tau_xy_pred, tau_yz_pred, tau_zx_pred = modelNN.predict(X_f)
        
        sigma_xx_exact, sigma_yy_exact, sigma_zz_exact, sigma_xy_exact, sigma_yz_exact, \
            sigma_zx_exact = getExactStresses(X_f[:,0], X_f[:,1], X_f[:,2], model)
        energy_err, energy_norm = energyError(X_f,sigma_x_pred,sigma_y_pred,model,\
                                         sigma_z_pred,tau_xy_pred,tau_yz_pred,tau_zx_pred)    
            
        print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
        
        if iStep < numSteps-1:
            # Error estimation
            f_u, f_v, f_w = modelNN.predict_f(X_f[:,0:3])
            res_err = np.sqrt(f_u**2 + f_v**2 + f_w**2)
                    
            numElem = len(vertex)
            errElem = np.zeros(numElem)
            for iElem in range(numElem):
                ptIndStart = iElem*numGauss**3
                ptIndEnd = (iElem+1)*numGauss**3
                #estimate the error in each element by integrating 
                errElem[iElem] = np.sum(res_err[ptIndStart:ptIndEnd]*X_f[ptIndStart:ptIndEnd,3])
            
            # Marking the elements for refinement
            N = 10 # N percent interior points with highest error
            ntop = np.int(np.round(numElem*N/100))
            sort_err_ind = np.argsort(-errElem, axis=0) 
            index_ref = np.squeeze(sort_err_ind[0:ntop]) # Indices of the elements that are to be refined
            
            vertex = refineElemVertex(vertex, index_ref)
            xPhys, yPhys, zPhys, wgtsPhys = myDomain.getElemIntPts(vertex, numGauss)
            X_f = np.concatenate((xPhys,yPhys, zPhys, wgtsPhys),axis=1)
            
            filename = 'Refined_scatter'+ str(iStep)
            scatterPlot(X_f,X_bnd,figHeight,figWidth,filename)
    
    os.chdir(originalDir)
