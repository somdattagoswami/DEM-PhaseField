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
from utils.PINN3D_PF import CalculateUPhi
from utils.gridPlot3D import plot1dPhi

np.random.seed(1234)
tf.set_random_seed(1234)
       
if __name__ == "__main__":
    
    originalDir = os.getcwd()
    foldername = 'Cube_results'    
    createFolder('./'+ foldername + '/')
    os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
    
    figHeight = 5
    figWidth = 5
    nSteps = 10 # Total number of steps to observe the growth of crack
    deltaU = 5e-4 #displacement increment per step 
    
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
    
    data = scipy.io.loadmat('../X_f_Cube_4th.mat')
    X_f = data['X_f']
    hist_f = np.transpose(np.array([np.zeros((X_f.shape[0]),dtype = np.float32)]))
    filename = 'Training_scatter'
    scatterPlot(X_f,figHeight,figWidth,filename)
    
    # Generating the prediction mesh
    nPred = np.array([111,20,111])  
    Grid, xGrid, yGrid, zGrid, hist_grid = genGrid(nPred, model['L'], model['T'], model['H'])
    phi_pred_old = hist_grid #Initializing phi_pred_old to zero

    modelNN = CalculateUPhi(model, NN_param)
    num_train_its = 10000

    for iStep in range(0,nSteps):
        
        if iStep==0:
            num_lbfgs_its = 10000
        else:
            num_lbfgs_its = 1000
        
        w_delta = deltaU*iStep            
        start_time = time.time()
            
        modelNN.train(X_f, w_delta, hist_f, num_train_its, num_lbfgs_its)
        
        _, _, _, phi_f, _, _, hist_f = modelNN.predict(X_f[:,0:3], hist_f, w_delta) # Computing the history function for the next step
        
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