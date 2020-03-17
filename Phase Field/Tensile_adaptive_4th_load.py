# Implements the fourth-order phase field to study the growth of fracture in a two dimensional plate
# The plate has initial crack and is under tensile loading
# Load the data file for the final refined domain to obtain the crack path

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
from utils.gridPlot2D import plot1dPhi
from utils.gridPlot2D import plotForceDisp
from utils.PINN2D_PF import CalculateUPhi

np.random.seed(1234)
tf.set_random_seed(1234)

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
        
        u = (1-x)*x*uNN
        v = y*(y-1)*vNN + y*vdelta

        return u, v
    
    def net_hist(self,x,y):
        
        shape = tf.shape(x)
        self.crackTip = 0.5
        init_hist = tf.zeros((shape[0],shape[1]), dtype = np.float32)
        dist = tf.where(x > self.crackTip, tf.sqrt((x-0.5)**2 + (y-0.5)**2), tf.abs(y-0.5))
        init_hist = tf.where(dist < 0.5*self.l, self.B*self.cEnerg*0.5*(1-(2*dist/self.l))/self.l, init_hist)
        
        return init_hist
    
    def net_energy(self,x,y,hist,vdelta):

        u, v = self.net_uv(x,y,vdelta)
        phi = self.net_phi(x,y)
        
        g = (1-phi)**2
        phi_x = tf.gradients(phi, x)[0]
        phi_y = tf.gradients(phi, y)[0]
        phi_xx = tf.gradients(phi_x, x)[0]
        phi_yy = tf.gradients(phi_y, y)[0] 
        
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
    foldername = 'TensionPlate_results_4thOrder'    
    createFolder('./'+ foldername + '/')

    figHeight = 5
    figWidth = 5
    nSteps = 8 # Total number of steps to observe the growth of crack 
    deltaV = 1e-3 # Displacement increment per step     
    
    model = dict()
    model['E'] = 210.0*1e3
    model['nu'] = 0.3
    model['L'] = 1.0
    model['W'] = 1.0
    model['l'] = 0.0125 # length scale parameter
    
    # Domain bounds
    model['lb'] = np.array([0.0,0.0]) # Lower bound of the plate
    model['ub'] = np.array([model['L'],model['W']]) # Upper bound of the plate

    NN_param = dict()
    NN_param['layers'] = [2, 50, 50, 50, 3]
    NN_param['data_type'] = tf.float32
    
    data = scipy.io.loadmat('X_f_Tensile_4th.mat')
    X_f = data['X_f']
    hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]),dtype = np.float32)]))
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    filename = 'Training_scatter'
    scatterPlot(X_f,figHeight,figWidth,filename)

    # Boundary data
    N_b = 800 # Number of boundary points for the fracture analysis (Left, Right, Top, Bottom)
    x_bottomEdge = np.array([np.linspace(0.0,model['L'],int(N_b/4), dtype = np.float32)])
    y_bottomEdge = np.zeros((int(N_b/4),1),dtype = np.float32)
    xBottomEdge = np.concatenate((x_bottomEdge.T,y_bottomEdge), axis=1)
    
    # Generate the prediction mesh
    nPred = np.array([[135,45],[135,45],[135,45]])
    offset = 2*model['l']    
    secBound = np.array([[0.0, 0.5*model['L']-offset],[0.5*model['L']-offset, 
                          0.5*model['L']+offset],[0.5*model['L']+offset, model['L']]], dtype = np.float32)
    Grid, xGrid, yGrid, hist_grid = genGrid(nPred,model['L'],secBound)
    filename = 'Prediction_scatter'
    scatterPlot(Grid,figHeight,figWidth,filename)

    fdGraph = np.zeros((nSteps,2),dtype = np.float32)
    phi_pred_old = hist_grid # Initializing phi_pred_old to zero
    
    modelNN = PINN_PF(model, NN_param)
    num_train_its = 12000
    
    for iStep in range(0,nSteps):
        
        v_delta = deltaV*iStep
        
        if iStep==0:
            num_lbfgs_its = 10000
        else:
            num_lbfgs_its = 1000
            
        start_time = time.time()
                            
        modelNN.train(X_f, v_delta, hist_f, num_train_its, num_lbfgs_its)
        
        _, _, phi_f, _, _, hist_f = modelNN.predict(X_f[:,0:2], hist_f, v_delta) # Computing the history function for the next step        
                              
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
        
        traction_pred = modelNN.predict_traction(xBottomEdge, v_delta)
        fdGraph[iStep,0] = v_delta
        fdGraph[iStep,1] = 4*np.sum(traction_pred, axis=0)/N_b
        
        # 1D plot of phase field
        xVal = 0.25
        nPredY = 2000
        xPred = xVal*np.ones((nPredY,1))
        yPred = np.linspace(0,model['W'],nPredY)[np.newaxis]
        xyPred = np.concatenate((xPred,yPred.T),axis=1)
        phi_pred_1d = modelNN.predict_phi(xyPred)
        phi_exact = np.exp(-np.absolute(yPred-0.5)/model['l'])*(1+np.absolute(yPred-0.5)/model['l'])
        plot1dPhi(yPred,phi_pred_1d,phi_exact,iStep,figHeight,figWidth)
        
        error_phi = (np.linalg.norm(phi_exact-phi_pred_1d,2)/np.linalg.norm(phi_exact,2))
        print('Relative error phi: %e' % (error_phi))
        
        print('Completed '+ str(iStep+1) +' of '+str(nSteps)+'.')    
        
    os.chdir(originalDir)
    plotForceDisp(fdGraph,figHeight,figWidth) # Plotting the force vs displacement graph