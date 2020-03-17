# The script implements the elastic field and the fourth order phase field in 1D
# Uses guass points from matlab
# Uses the monolithic solver
# Use energy method

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
import os
from utils.PINN1D_PF import CalculateUPhi
from utils.gridPlot1D import plotConvergence
from utils.gridPlot1D import createFolder
from utils.gridPlot1D import plot1dPhi
from utils.gridPlot1D import plot1dU
from utils.gridPlot1D import scatterPlot
from utils.gridPlot1D import refineElemVertex
from utils.BezExtr import Geometry1D

tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
np.random.seed(1234)
tf.set_random_seed(1234)

class Bar(Geometry1D):
    '''
     Class for definining a quadrilateral domain
     Input: quadDom - array of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                containing the domain corners (control-points)
    '''
    def __init__(self, domainEnds):
      
        # Domain bounds
        self.domainEnds = domainEnds        
        self.x1 = self.domainEnds[0]
        self.x2 = self.domainEnds[1]
        
        geomData = dict()
        # Set degrees
        geomData['degree_u'] = 1
        # Set control points
        geomData['ctrlpts_size_u'] = 2
        
        geomData['ctrlpts'] = [[self.x1, 0.0, 0.0], [self.x2, 0.0, 0.0]]
        
        geomData['weights'] = [1.0, 1.0]
        
        # Set knot vectors
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        
        super().__init__(geomData)

if __name__ == "__main__":
    
    originalDir = os.getcwd()
    figHeight = 2
    figWidth = 4
    
    createFolder('./1D_4th' + '/')
    os.chdir(os.path.join(originalDir, './1D_4th' + '/'))
    
    # Generating points inside the domain using Geometry class
    domainEnds = np.array([-1.0, 1.0])
    myLine = Bar(domainEnds)
    numElemU = 200
    numGauss = 4
    xPhys, wgtsPhys, vertex = myLine.getIntPts(numElemU, numGauss)

    X_f = np.concatenate((xPhys, wgtsPhys),axis = 1)
    filename = 'Training_scatter'
    scatterPlot(X_f,figHeight,figWidth,filename)

    l = 0.0125
    # Domain bounds
    lb = np.array([-1.0])
    ub = np.array([1.0])    
    layers = [1, 50, 50, 50, 2]
    
    xLeft = np.transpose(np.array([np.linspace(-1.0, -0.25, 100)]))
    xCenter = np.transpose(np.array([np.linspace(-0.25, 0.25, 400)]))
    xRight = np.transpose(np.array([np.linspace(0.25, 1.0, 100)]))
    xSpace = np.concatenate((xLeft, xCenter, xRight),axis = 0)

    numSteps = 4 # Number of refinement steps
    num_train_its = 5000
    modelNN = CalculateUPhi(l, layers, lb, ub)
    for i in range(numSteps):

        # Compute/training
        start_time = time.time()
        modelNN.train(X_f, num_train_its)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        print("Degrees of freedom ", X_f.shape[0])
        # Error estimation
        f_u = modelNN.predict_f(X_f)
        res_err = np.sqrt(f_u**2)
                
        numElem = len(vertex)
        errElem = np.zeros(numElem)
        for iElem in range(numElem):
            ptIndStart = iElem*numGauss
            ptIndEnd = (iElem+1)*numGauss
            # Estimate the error in each element by integrating 
            errElem[iElem] = np.sum(res_err[ptIndStart:ptIndEnd]*X_f[ptIndStart:ptIndEnd,1])
            
        # Marking the elements for refinement
        N = 10 # N percent interior points with highest error
        ntop = np.int(np.round(numElem*N/100))
        sort_err_ind = np.argsort(-errElem, axis=0) 
        index_ref = np.squeeze(sort_err_ind[0:ntop]) # Indices of the elements that are to be refined
        
        # Refine element list
        vertex = refineElemVertex(vertex, index_ref)
        xPhys, wgtsPhys = myLine.getElmtIntPts(vertex, numGauss)
        X_f = np.concatenate((xPhys, wgtsPhys),axis = 1)  
    
        filename = 'Refined_scatter'+ str(i)
        scatterPlot(X_f,figHeight,figWidth,filename)
        

        u_pred, phi_pred = modelNN.predict(xSpace)
        
        u_exact = np.sin(np.pi*xSpace)/(np.pi)**2 + np.where(xSpace < 0.0, -1.0*(1+xSpace)/np.pi, (1-xSpace)/np.pi)
        phi_exact = np.exp(-np.absolute(xSpace-0.0)/l)*(1+np.absolute(xSpace-0.0)/l)
        error_u = (np.linalg.norm(u_exact-u_pred,2)/np.linalg.norm(u_exact,2))
        print('Relative error u: %e' % (error_u))   
        error_phi = (np.linalg.norm(phi_exact-phi_pred,2)/np.linalg.norm(phi_exact,2))
        print('Relative error phi: %e' % (error_phi))
    
        plot1dU(xSpace, u_pred, u_exact, figHeight,figWidth)
        plot1dPhi(xSpace, phi_pred, phi_exact, figHeight,figWidth)
        
        adam_buff = modelNN.loss_adam_buff
        lbfgs_buff = modelNN.lbfgs_buffer
        plotConvergence(num_train_its,adam_buff,lbfgs_buff,figHeight,figWidth)
    
    os.chdir(originalDir)


