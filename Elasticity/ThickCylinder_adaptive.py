# Implements the 2D pressurized cylinder benchmark problem subjected to 
# internal pressure on the inner circular edge, under plane stress condition

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
import os
from utils.gridPlot2D import getExactDisplacements
from utils.gridPlot2D import plotDeformedDisp
from utils.gridPlot2D import energyError
from utils.gridPlot2D import scatterPlot
from utils.gridPlot2D import createFolder
from utils.gridPlot2D import energyPlot
from utils.gridPlot2D import refineElemVertex
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

from utils.Geom import Geometry2D
from utils.PINN_adaptive import Elasticity2D

class Annulus(Geometry2D):
    '''
     Class for definining a quarter-annulus domain centered at the orgin
         (the domain is in the first quadrant)
     Input: rad_int, rad_ext - internal and external radii of the annulus
    '''
    def __init__(self, radInt, radExt):            
        
        geomData = dict()
        # Set degrees
        geomData['degree_u'] = 1
        geomData['degree_v'] = 2
        
        # Set control points
        geomData['ctrlpts_size_u'] = 2
        geomData['ctrlpts_size_v'] = 3
                
        geomData['ctrlpts'] = [[radInt,0.,0.],
                    [radInt*np.sqrt(2)/2, radInt*np.sqrt(2)/2, 0.],
                    [0., radInt, 0.],
                    [radExt, 0., 0.],
                    [radExt*np.sqrt(2)/2, radExt*np.sqrt(2)/2, 0.],
                    [0., radExt, 0.]]
        
        geomData['weights'] = [1, np.sqrt(2)/2, 1, 1, np.sqrt(2)/2, 1]
        
        # Set knot vectors
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        super().__init__(geomData)

class PINN_TC(Elasticity2D):
    '''
    Class including (symmetry) boundary conditions for the thick cylinder problem
    '''
    def __init__(self, model_data, xEdgePts, NN_param):
        
        super().__init__(model_data,xEdgePts, NN_param)
        
    def net_uv(self, x, y):

        X = tf.concat([x, y], 1)      

        uv = self.neural_net(X,self.weights,self.biases)

        u = x*uv[:, 0:1]
        v = y*uv[:, 1:2]

        return u, v

if __name__ == "__main__":
    
    originalDir = os.getcwd()
    foldername = 'ThickCylinder'    
    createFolder('./'+ foldername + '/')
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    figHeight = 6
    figWidth = 8
    
    model_data = dict()
    model_data['E'] = 1e5
    model_data['nu'] = 0.3
    
    model = dict()
    model['E'] = 1e5
    model['nu'] = 0.3
    model['radInt'] = 1.0
    model['radExt'] = 4.0
    model['P'] = 10.0
    
    # Domain bounds
    model_data['lb'] = np.array([0.0,0.0]) #lower bound of the plate
    model_data['ub'] = np.array([model['radExt'],model['radExt']]) # Upper bound of the plate
    
    NN_param = dict()
    NN_param['layers'] = [2, 30, 30, 30, 2]
    NN_param['data_type'] = tf.float32
    
    # Generating points inside the domain using GeometryIGA
    myAnnulus = Annulus(model['radInt'], model['radExt'])
    
    numElemU = 20
    numElemV = 20
    numGauss = 2
    
    vertex = myAnnulus.genElemList(numElemU, numElemV)
    xPhys, yPhys, wgtsPhys = myAnnulus.getElemIntPts(vertex, numGauss)
    
    myAnnulus.plotKntSurf()
    plt.scatter(xPhys, yPhys, s=0.5)
    X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)
    
    # Generate the boundary points using Geometry class
    numElemEdge = 80
    numGaussEdge = 1
    xEdge, yEdge, xNormEdge, yNormEdge, wgtsEdge = myAnnulus.getQuadEdgePts(numElemEdge,
                                                                numGaussEdge, 4)
    trac_x = -model['P'] * xNormEdge
    trac_y = -model['P'] * yNormEdge
    xEdgePts = np.concatenate((xEdge, yEdge, wgtsEdge, trac_x, trac_y), axis=1)
    
    model_pts = dict()
    model_pts['X_int'] = X_f
    model_pts['X_bnd'] = xEdgePts
    
    modelNN = PINN_TC(model_data, xEdgePts, NN_param)
    
    filename = 'Training_scatter'
    scatterPlot(X_f,xEdgePts,figHeight,figWidth,filename)

    nPred = 40
    withEdges = [1, 1, 1, 1]
    xGrid, yGrid = myAnnulus.getUnifIntPts(nPred, nPred, withEdges)
    Grid = np.concatenate((xGrid,yGrid),axis=1)
    
    num_train_its = 1000
    numSteps = 4 # Number of refinement steps
    for i in range(numSteps):
        
        # Compute/training
        start_time = time.time()
        modelNN.train(X_f, num_train_its)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        print("Degrees of freedom ", X_f.shape[0])
        # Error estimation
        f_u, f_v = modelNN.predict_f(X_f[:,0:2])
        res_err = np.sqrt(f_u**2 + f_v**2)
                
        numElem = len(vertex)
        errElem = np.zeros(numElem)
        for iElem in range(numElem):
            ptIndStart = iElem*numGauss**2
            ptIndEnd = (iElem+1)*numGauss**2
            # Estimate the error in each element by integrating 
            errElem[iElem] = np.sum(res_err[ptIndStart:ptIndEnd]*X_f[ptIndStart:ptIndEnd,2])
        
        # Marking the elements for refinement
        N = 10 # N percent interior points with highest error
        ntop = np.int(np.round(numElem*N/100))
        sort_err_ind = np.argsort(-errElem, axis=0) 
        index_ref = np.squeeze(sort_err_ind[0:ntop]) # Indices of the elements that are to be refined
        
        # Refine element list
        vertex = refineElemVertex(vertex, index_ref)
        xPhys, yPhys, wgtsPhys = myAnnulus.getElemIntPts(vertex, numGauss)
        X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)
    
    
        filename = 'Refined_scatter'+ str(i)
        scatterPlot(X_f,xEdgePts,figHeight,figWidth,filename)
        
        u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(X_f)  
        energy_err, energy_norm = energyError(X_f,sigma_x_pred,sigma_y_pred,model,tau_xy_pred)   
        print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
        
    # Plot results
    # Magnification factors for plotting the deformed shape
    x_fac = 2
    y_fac = 2
    
    # Compute the approximate displacements at plot points
    u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(Grid)
    u_exact, v_exact = getExactDisplacements(xGrid, yGrid, model) # Computing exact displacements 
    oShapeX = np.resize(xGrid, [nPred, nPred])
    oShapeY = np.resize(yGrid, [nPred, nPred])
    surfaceUx = np.resize(u_pred, [nPred, nPred])
    surfaceUy = np.resize(v_pred, [nPred, nPred])
    surfaceExUx = np.resize(u_exact, [nPred, nPred])
    surfaceExUy = np.resize(v_exact, [nPred, nPred])
    
    defShapeX = oShapeX + surfaceUx * x_fac
    defShapeY = oShapeY + surfaceUy * y_fac
    surfaceErrUx = surfaceExUx - surfaceUx
    surfaceErrUy = surfaceExUy - surfaceUy      
           
    print("Deformation plots")
    filename = 'Deformation'
    plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY, filename)
    
    print("Exact plots")
    filename = 'Exact'
    plotDeformedDisp(surfaceExUx, surfaceExUy, defShapeX, defShapeY, filename)
    
    print("Error plots")
    filename = 'Error'
    plotDeformedDisp(surfaceErrUx, surfaceErrUy, oShapeX, oShapeY, filename)
    
    # Plotting the strain energy densities 
    filename = 'Strain_energy'       
    energyPlot(defShapeX,defShapeY,nPred,energy_pred,filename,figHeight,figWidth)

    # Compute the L2 and energy norm errors using integration
    u_pred, v_pred, energy_pred, sigma_x_pred, sigma_y_pred, tau_xy_pred = modelNN.predict(X_f)
    u_exact, v_exact = getExactDisplacements(X_f[:,0], X_f[:,1], model)
    err_l2 = np.sum(((u_exact-u_pred[:,0])**2 + (v_exact-v_pred[:,0])**2)*X_f[:,2])
    norm_l2 = np.sum((u_exact**2 + v_exact**2)*X_f[:,2])
    error_u_l2 = np.sqrt(err_l2/norm_l2)
    print("Relative L2 error (integration): ", error_u_l2)
    
    energy_err, energy_norm = energyError(X_f,sigma_x_pred,sigma_y_pred,model,tau_xy_pred)
    print("Relative energy error (integration): ", np.sqrt(energy_err/energy_norm))
    
    os.chdir(originalDir)
