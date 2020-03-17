# Implements fourth order phase field + elasticity for symmetric double notched plate under tensile loading
# The plate has two initial cracks
# Refines the domain adaptively

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import time
import os
import scipy.io
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
tf.logging.set_verbosity(tf.logging.ERROR)
from utils.gridPlot2D import scatterPlot
from utils.gridPlot2D import genGrid
from utils.gridPlot2D import plotDispStrainEnerg
from utils.gridPlot2D import plotPhiStrainEnerg
from utils.gridPlot2D import plotConvergence
from utils.gridPlot2D import createFolder
from utils.gridPlot2D import plot1dPhisymmDN
from utils.gridPlot2D import refineElemVertex
from utils.BezExtr import Geometry2D
from utils.PINN2D_PF import CalculateUPhi

np.random.seed(1234)
tf.set_random_seed(1234)

class Quadrilateral(Geometry2D):
    '''
     Class for definining a quadrilateral domain
     Input: quadDom - array of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                containing the domain corners (control-points)
    '''
    def __init__(self, quadDom):
      
         # Domain bounds
        self.quadDom = quadDom
        
        self.x1, self.y1 = self.quadDom[0,:]
        self.x2, self.y2 = self.quadDom[1,:]
        self.x3, self.y3 = self.quadDom[2,:]
        self.x4, self.y4 = self.quadDom[3,:]
        
        geomData = dict()
        
        # Set degrees
        geomData['degree_u'] = 1
        geomData['degree_v'] = 1
        
        # Set control points
        geomData['ctrlpts_size_u'] = 2
        geomData['ctrlpts_size_v'] = 2
        
        geomData['ctrlpts'] = np.array([[self.x1, self.y1, 0], [self.x2, self.y2, 0],
                        [self.x3, self.y3, 0], [self.x4, self.y4, 0]])

        geomData['weights'] = np.array([[1.0], [1.0], [1.0], [1.0]])
        
        # Set knot vectors
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 1.0, 1.0]

        super().__init__(geomData)

class PINN_PF(CalculateUPhi):
    '''
    Class including (symmetry) boundary conditions for the tension plate
    '''
    def __init__(self, model, NN_param):
        
        super().__init__(model, NN_param)
        
    def net_uv(self,x,y,vdelta):

        X = tf.concat([x,y],1)

        uvphi = self.neural_net(X,self.weights,self.biases)
        uNN = uvphi[:,0:1]
        vNN = uvphi[:,1:2]
        
        u = y*uNN
        v = y*(y-1)*vNN + y*vdelta

        return u, v
    
    def net_hist(self,x,y):
        
        shape = tf.shape(x)
        init_hist = tf.zeros((shape[0],shape[1]), dtype = np.float32)
        
        self.crackTip1 = 0.1
        self.crackTip2 = 0.4
        # Considering the crack on the left side
        dist1 = tf.where(x > self.crackTip1, tf.sqrt((x-self.crackTip1)**2 + (y-0.5)**2), tf.abs(y-0.5))
        init_hist = tf.where(dist1 < 0.5*self.l, self.B*self.cEnerg*0.5*(1-(2*dist1/self.l))/self.l, init_hist)    
        
        # Cosidering the crack on the right side
        dist2 = tf.where(x < self.crackTip2, tf.sqrt((x-self.crackTip2)**2 + (y-0.5)**2), tf.abs(y-0.5))
        init_hist = tf.where(dist2 < 0.5*self.l, self.B*self.cEnerg*0.5*(1-(2*dist2/self.l))/self.l, init_hist)
        
        return init_hist
    
    def net_energy(self,x,y,hist,vdelta):

        u, v = self.net_uv(x,y,vdelta)
        phi = self.net_phi(x,y)
        
        g = (1-phi)**2
        phi_x = tf.gradients(phi,x)[0]
        phi_y = tf.gradients(phi,y)[0]
        phi_xx = tf.gradients(phi_x,x)[0]
        phi_yy = tf.gradients(phi_y,y)[0] 
        
        nabla = phi_x**2 + phi_y**2     
        laplacian = (phi_xx + phi_yy)**2
        u_x = tf.gradients(u,x)[0]
        v_y = tf.gradients(v,y)[0]
        u_y = tf.gradients(u,y)[0]
        v_x = tf.gradients(v,x)[0]
        u_xy = (u_y + v_x)
        
        hist = self.net_update_hist(x, y, u_x, v_y, u_xy, hist) 
        
        sigmaX = self.c11*u_x + self.c12*v_y
        sigmaY = self.c21*u_x + self.c22*v_y
        tauXY = self.c33*u_xy
        
        energy_u = 0.5*g*(sigmaX*u_x + sigmaY*v_y + tauXY*u_xy)
        energy_phi = 0.5*self.cEnerg * (phi**2/self.l + self.l*nabla + \
                                        0.5*self.l**3*laplacian) + g* hist
        
        return energy_u, energy_phi, hist
        
if __name__ == "__main__":
    
    originalDir = os.getcwd()
    foldername = 'SymmDoubNot_save'    
    createFolder('./'+ foldername + '/')
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    
    figHeight = 7
    figWidth = 5
    nSteps = 11 # Total number of steps to observe the growth of crack 
    deltaV = 5e-4 # Displacement increment per step 
    
    model = dict()
    model['E'] = 210.0*1e3
    model['nu'] = 0.3
    model['L'] = 0.5
    model['W'] = 1.0
    model['l'] = 0.01 # length scale parameter
    
    # Domain bounds
    model['lb'] = np.array([0.0,0.0]) # Lower bound of the plate
    model['ub'] = np.array([model['L'],model['W']]) # Upper bound of the plate

    NN_param = dict()
    NN_param['layers'] = [2, 50, 50, 50, 50, 3]
    NN_param['data_type'] = tf.float32
    
    # Generating points inside the domain using Geometry class
    domainCorners = np.array([[0.0,0.0],[model['L'],0.0],[0.0,model['W']],[model['L'],model['W']]])
    myQuad = Quadrilateral(domainCorners)

    numElemU = 10
    numElemV = 20
    numGauss = 4
    maxLevel = 3
    
    phiRefThresh = 0.1
    
    vertex = myQuad.genElemList(numElemU, numElemV)
    xPhys, yPhys, wgtsPhys = myQuad.getElemIntPts(vertex, numGauss)
    X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)
    hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]),dtype = np.float32)]))
    filename = 'Training_scatter'
    scatterPlot(X_f,figHeight,figWidth,filename)
   
    # Generating the prediction mesh
    nPred = np.array([[135,45],[135,45],[135,45]])
    offset = 2*model['l']    
    secBound = np.array([[0.0, 0.5*model['W']-offset],[0.5*model['W']-offset, 
                          0.5*model['W']+offset],[0.5*model['W']+offset, model['W']]], dtype = np.float32)
    Grid, xGrid, yGrid, hist_grid = genGrid(nPred,model['L'],secBound)
    filename = 'Prediction_scatter'
    scatterPlot(Grid,figHeight,figWidth,filename)

    phi_pred_old = hist_grid #Initializing phi_pred_old to zero
    
    modelNN = PINN_PF(model, NN_param)
    num_train_its = 10000
    
    for iStep in range(0,nSteps):
        
        v_delta = deltaV*iStep        
        keepTraining = 1
        miter = 0       
        start_time = time.time()
        
        if iStep<2:
            maxInnerIter = 2
            num_lbfgs_its = 10000
        else:
            maxInnerIter = 1
            num_lbfgs_its = 1000
        
        while (keepTraining > 0) and (miter < maxInnerIter):
            
            miter = miter + 1
            print('Inner Interation: %d' %(miter))
            
            modelNN.train(X_f, v_delta, hist_f, num_train_its, num_lbfgs_its)
            
            _, _, phi_f, _, _, hist_f = modelNN.predict(X_f[:,0:2], hist_f, v_delta) # Computing the history function for the next step        
            
            f_u, f_v = modelNN.predict_f(X_f[:,0:2],v_delta)
            res_err = np.sqrt(f_u**2 + f_v**2)
            numElem = len(vertex)
            errElem = np.zeros(numElem)
            for iElem in range(numElem):
                ptIndStart = iElem*numGauss**2
                ptIndEnd = (iElem+1)*numGauss**2
                # Estimate the error in each element by integrating 
                errElem[iElem] = np.sum(res_err[ptIndStart:ptIndEnd]*X_f[ptIndStart:ptIndEnd,2])
                
            # Marking the elements for refinement
            N = 7 # N percent interior points with highest residual
            ntop = np.int(np.round(numElem*N/100))
            sort_err_ind = np.argsort(-errElem, axis=0) 
            elem_refine_residual = np.squeeze(sort_err_ind[0:ntop]) # Indices of the elements that are to be refined            
            # Refinement of the element is allowed only up to maxLevel        
            level = vertex[elem_refine_residual,4] # Obtaining the level of refinement
            index_ref_residual = np.where(level < maxLevel)[0]
            
            elem_refine_residual = elem_refine_residual[index_ref_residual]
            
            # Marking the elements for refinement
            index = np.where(phi_f >= phiRefThresh) # Indices of the gauss points that are to be refined.
            
            elem_refine = np.floor(index[0]/numGauss**2).astype(int) # Obtains the element numbers of the gauss points that need to be refined
            elem_refine = np.unique(elem_refine)
        
            # Refinement of the element is allowed only up to maxLevel        
            level = vertex[elem_refine,4] # Obtaining the level of refinement    
            index_ref_phi = np.where(level < maxLevel)[0] # Obtaining the index numbers in the vertex matrix corresponding to the element numbers 
            
            elem_refine = elem_refine[index_ref_phi]
            
            index_ref = np.union1d(elem_refine, elem_refine_residual)
            
            if (len(elem_refine) > 0) or (iStep==0):
                print("Number of elements refining: ", len(index_ref))
                print("refining subdivisions (phi) = ", elem_refine)
                print("refining subdivisions (residual) = ", elem_refine_residual)        
                
                #tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
                #modelNN.weights, modelNN.biases = modelNN.initialize_NN(NN_param['layers'])
                
                vertex = refineElemVertex(vertex, index_ref)
                xPhys, yPhys, wgtsPhys = myQuad.getElemIntPts(vertex, numGauss)
                X_f = np.concatenate((xPhys,yPhys, wgtsPhys),axis=1)
                hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]),dtype = np.float32)]))
                u_pred, v_pred, phi_pred, elas_energy_pred, frac_energy_pred, hist_grid = modelNN.predict(Grid, hist_grid, v_delta)
                _, _, _, _, _, hist_f = modelNN.predict(X_f[:,0:2], hist_f, v_delta) # Computing the history function for the next step
                
                filename = 'Refine_scatter_'+ str(iStep) + '_Miter_'+ str(miter)
                scatterPlot(X_f,figHeight,figWidth,filename)
                filename = str(iStep) + '_Miter_'+ str(miter)
                plotPhiStrainEnerg(nPred, xGrid, yGrid, phi_pred, frac_energy_pred, filename,figHeight,figWidth)
                scipy.io.savemat('../X_f_SymmDN.mat', {'X_f': X_f})
            else:

                keepTraining = 0
        
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        u_pred, v_pred, phi_pred, elas_energy_pred, frac_energy_pred, hist_grid = modelNN.predict(Grid, hist_grid, v_delta)        
              
        phi_pred = np.maximum(phi_pred, phi_pred_old)
        phi_pred_old = phi_pred
        filename = str(iStep)
        plotPhiStrainEnerg(nPred,xGrid,yGrid,phi_pred,frac_energy_pred,filename,figHeight,figWidth)
        plotDispStrainEnerg(nPred,xGrid,yGrid,u_pred,v_pred,elas_energy_pred,filename,figHeight,figWidth)

        adam_buff = modelNN.loss_adam_buff
        lbfgs_buff = modelNN.lbfgs_buffer
        plotConvergence(num_train_its,adam_buff,lbfgs_buff,iStep,figHeight,figWidth)
                
        # 1D plot of phase field
        xValSet = [0.05, 0.45]
        for xVal in xValSet:
            nPredY = 2000
            xPred = xVal*np.ones((nPredY,1))
            yPred = np.linspace(0,model['W'],nPredY)[np.newaxis]
            xyPred = np.concatenate((xPred,yPred.T),axis=1)
            phi_pred_1d = modelNN.predict_phi(xyPred)
            phi_exact = np.exp(-np.absolute(yPred-0.5)/model['l'])*(1+np.absolute(yPred-0.5)/model['l'])
            
            plot1dPhisymmDN(yPred,phi_pred_1d,phi_exact,iStep,xVal,figHeight,figWidth)
        
            error_phi = (np.linalg.norm(phi_exact.T-phi_pred_1d,2)/np.linalg.norm(phi_exact,2))
            print('Relative error phi: %e' % (error_phi))
        
        print('Completed '+ str(iStep+1) +' of '+str(nSteps)+'.')    
        
    os.chdir(originalDir)
    print('Quadrature points saved in X_f_SymmDN.mat')