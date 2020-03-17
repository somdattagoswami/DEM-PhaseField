# Implements the fourth-order phase field to study the growth of fracture in a cube
# The cube has initial crack and is under tensile loading
# Refines the domain adaptively

import tensorflow as tf
import numpy as np
import time
import os
import scipy.io
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
tf.logging.set_verbosity(tf.logging.ERROR)
from utils.gridPlot3D import genGrid
from utils.gridPlot3D import scatterPlot
from utils.gridPlot3D import plotDispStrainEnerg
from utils.gridPlot3D import plotPhiStrainEnerg
from utils.gridPlot3D import plotConvergence
from utils.gridPlot3D import createFolder
from utils.gridPlot3D import refineElemVertex
from utils.BezExtr import Geometry3D
from utils.PINN3D_PF import CalculateUPhi
from utils.gridPlot3D import plot1dPhi

np.random.seed(1234)
tf.set_random_seed(1234)

class Quadrilateral(Geometry3D):
    '''
     Class for definining a quadrilateral domain
     Input: quadDom - array of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                containing the domain corners (control-points)
    '''
    def __init__(self, quadDom):
      
         # Domain bounds
        self.quadDom = quadDom
        
        self.x1, self.y1, self.z1 = self.quadDom[0,:]
        self.x2, self.y2, self.z2 = self.quadDom[1,:]
        self.x3, self.y3, self.z3 = self.quadDom[2,:]
        self.x4, self.y4, self.z4 = self.quadDom[3,:]
        self.x5, self.y5, self.z5 = self.quadDom[4,:]
        self.x6, self.y6, self.z6 = self.quadDom[5,:]
        self.x7, self.y7, self.z7 = self.quadDom[6,:]
        self.x8, self.y8, self.z8 = self.quadDom[7,:]
        
        geomData = dict()
        
        # Set degrees
        geomData['degree_u'] = 1
        geomData['degree_v'] = 1
        geomData['degree_w'] = 1
        
        # Set control points
        geomData['ctrlpts_size_u'] = 2
        geomData['ctrlpts_size_v'] = 2
        geomData['ctrlpts_size_w'] = 2
        
        geomData['ctrlpts'] = np.array([[self.x1, self.y1, self.z1], [self.x2, self.y2, self.z2],
                        [self.x3, self.y3, self.z3], [self.x4, self.y4, self.z4], [self.x5, self.y5, self.z5],
                        [self.x6, self.y6, self.z6], [self.x7, self.y7, self.z7], [self.x8, self.y8, self.z8]])

        geomData['weights'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                
        
        # Set knot vectors
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_w'] = [0.0, 0.0, 1.0, 1.0]
        
        super().__init__(geomData)
        
if __name__ == "__main__":
    
    originalDir = os.getcwd()
    foldername = 'Cube_save'    
    createFolder('./'+ foldername + '/')
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    
    figHeight = 5
    figWidth = 5
    nSteps = 2 # Total number of steps to observe the growth of crack
    deltaU = 1e-3 #displacement increment per step 
    
    model = dict()
    model['E'] = 210.0*1e3
    model['nu'] = 0.3
    model['L'] = 1.0
    model['T'] = 0.2
    model['H'] = 1.0
    model['l'] = 0.0625/2 # length scale parameter
    
    # Domain bounds
    model['lb'] = np.array([0.0,0.0,0.0]) #lower bound of the plate
    model['ub'] = np.array([model['L'],model['T'],model['H']]) # Upper bound of the plate

    NN_param = dict()
    NN_param['layers'] = [3, 50, 50, 50, 50, 4]
    NN_param['data_type'] = tf.float32
    
    # Generating points inside the domain using Geometry class
    domainCorners = np.array([[0.0,0.0,0.0],[0.0, model['T'], 0.0],[model['L'], 0.0,0.0],[model['L'],model['T'],0.0],[0.0,0.0,model['H']],[0.0,model['T'],model['H']],[model['L'],0.0,model['H']],[model['L'],model['T'],model['H']]])
    myDomain = Quadrilateral(domainCorners)

    numElemU = 2
    numElemV = 10
    numElemW = 10
    numGauss = 2
    maxLevel = 2
    maxInnerIter = 3
    phiRefThresh = 0.1
    
    vertex = myDomain.genElemList(numElemU, numElemV, numElemW)
    xPhys, yPhys, zPhys, wgtsPhys = myDomain.getElemIntPts(vertex, numGauss)
    X_f = np.concatenate((xPhys,yPhys, zPhys, wgtsPhys),axis=1)
    hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]),dtype = np.float32)]))
    filename = 'Training_scatter'
    scatterPlot(X_f,figHeight,figWidth,filename)
    
    # Generating the prediction mesh
    nPred = np.array([111,20,111])  
    Grid, xGrid, yGrid, zGrid, hist_grid = genGrid(nPred, model['L'], model['T'], model['H'])
    phi_pred_old = hist_grid # Initializing phi_pred_old to zero

    modelNN = CalculateUPhi(model, NN_param)
    num_train_its = 10000
    num_lbfgs_its = 10000

    for iStep in range(0,nSteps):
        
        w_delta = deltaU*iStep        
        keepTraining = 1
        miter = 0        
        start_time = time.time()        
        while (keepTraining > 0) and (miter < maxInnerIter):
            
            miter = miter + 1
            print('Inner Interation: %d' %(miter))
            
            modelNN.train(X_f, w_delta, hist_f, num_train_its, num_lbfgs_its)
            
            _, _, _, phi_f, _, _, hist_f = modelNN.predict(X_f[:,0:3], hist_f, w_delta) # Computing the history function for the next step        
            
            f_u, f_v, f_w = modelNN.predict_f(X_f[:,0:3], w_delta)
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
            elem_refine_residual = np.squeeze(sort_err_ind[0:ntop]) # Indices of the elements that are to be refined         
            # Refinement of the element is allowed only up to maxLevel        
            level = vertex[elem_refine_residual,6] # Obtaining the level of refinement
            index_ref_residual = np.where(level < maxLevel)[0]
            
            elem_refine_residual = elem_refine_residual[index_ref_residual]
            
            # Marking the elements for refinement
            index = np.where(phi_f >= phiRefThresh) # Indices of the gauss points that are to be refined.
            
            elem_refine = np.floor(index[0]/numGauss**3).astype(int) # Obtains the element numbers of the gauss points that need to be refined
            elem_refine = np.unique(elem_refine)
        
            # Refinement of the element is allowed only up to maxLevel        
            level = vertex[elem_refine,6] # Obtaining the level of refinement    
            index_ref_phi = np.where(level < maxLevel)[0] # Obtaining the index numbers in the vertex matrix corresponding to the element numbers 
            
            elem_refine = elem_refine[index_ref_phi]
            
            index_ref = np.union1d(elem_refine, elem_refine_residual)
            
            if (len(elem_refine) > 0) or (iStep==0):
                print("Number of elements refining: ", len(index_ref))
                print("refining subdivisions (phi) = ", elem_refine)
                print("refining subdivisions (residual) = ", elem_refine_residual)                                      
                
                vertex = refineElemVertex(vertex, index_ref)
                xPhys, yPhys, zPhys, wgtsPhys = myDomain.getElemIntPts(vertex, numGauss)
                X_f = np.concatenate((xPhys,yPhys, zPhys, wgtsPhys),axis=1)
                hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]),dtype = np.float32)]))
                u_pred, v_pred, w_pred, phi_pred, elas_energy_pred, frac_energy_pred, hist_grid = modelNN.predict(Grid, hist_grid, w_delta)
                _, _, _, _, _, _, hist_f = modelNN.predict(X_f[:,0:3], hist_f, w_delta) # Computing the history function for the next step                
                
                filename = 'Refine_scatter_'+ str(iStep) + '_Miter_'+ str(miter)
                scatterPlot(X_f,figHeight,figWidth,filename)
                filename = str(iStep) + '_Miter_'+ str(miter)
                plotPhiStrainEnerg(nPred, xGrid, yGrid, zGrid, phi_pred, frac_energy_pred, filename)
                scipy.io.savemat('../X_f_Cube_4th.mat', {'X_f': X_f})
            else:

                keepTraining = 0
        
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        u_pred, v_pred, w_pred, phi_pred, elas_energy_pred, frac_energy_pred, hist_grid = modelNN.predict(Grid, hist_grid, w_delta)        
        
        phi_pred = np.maximum(phi_pred, phi_pred_old)
        phi_pred_old = phi_pred
        
        filename = str(iStep)
        plotPhiStrainEnerg(nPred, xGrid, yGrid, zGrid, phi_pred, frac_energy_pred, filename)
        plotDispStrainEnerg(nPred, xGrid, yGrid, zGrid, u_pred, v_pred, w_pred, elas_energy_pred, filename)

        adam_buff = modelNN.loss_adam_buff
        lbfgs_buff = modelNN.lbfgs_buffer
        plotConvergence(num_train_its,adam_buff,lbfgs_buff, iStep, figHeight, figWidth)
                
        # 1D plot of phase field    
        xVal = 0.25
        yVal = 0.25
        nPredZ = 2000
        xPred = xVal*np.ones((nPredZ,1))
        yPred = yVal*np.ones((nPredZ,1))
        zPred = np.linspace(0,model['H'],nPredZ)[np.newaxis]
        xyzPred = np.concatenate((xPred,yPred,zPred.T),axis=1)
        phi_pred_1d = modelNN.predict_phi(xyzPred)
        phi_exact = np.exp(-np.absolute(zPred-0.5)/model['l'])*(1+np.absolute(zPred-0.5)/model['l'])
         
        plot1dPhi(zPred,phi_pred_1d,phi_exact,iStep,figHeight,figWidth)
        
        error_phi = (np.linalg.norm(phi_exact.T-phi_pred_1d,2)/np.linalg.norm(phi_exact.T,2))
        print('Relative error u: %e' % (error_phi))
        
        error_phi = (np.linalg.norm(phi_exact.T-phi_pred_1d,2)/np.linalg.norm(phi_exact,2))
        print('Relative error phi: %e' % (error_phi))
        
        print('Completed '+ str(iStep+1) +' of '+str(nSteps)+'.')    
        
    os.chdir(originalDir)
    print('Quadrature points saved in X_f_Cube_4th.mat')