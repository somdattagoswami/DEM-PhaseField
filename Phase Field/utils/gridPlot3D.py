import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from mpl_toolkits.mplot3d import Axes3D
from pyevtk.hl import gridToVTK 

def scatterPlot(X_f,figHeight,figWidth,filename):
    
    fig = plt.figure(figsize=(figWidth,figHeight))
    ax = fig.gca(projection='3d')
    ax.scatter(X_f[:,0], X_f[:,1], X_f[:,2], s = 0.75)
    ax.set_xlabel('$x$',fontweight='bold',fontsize = 12)
    ax.set_ylabel('$y$',fontweight='bold',fontsize = 12)
    ax.set_zlabel('$z$',fontweight='bold',fontsize = 12)
    ax.tick_params(axis='both', which='major', labelsize = 6)
    ax.tick_params(axis='both', which='minor', labelsize = 6)
    plt.savefig(filename + ".png", dpi=300, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close() 

def genGrid(nPred, L, T, H):
    
    x = np.array([np.linspace(0.0,L,nPred[0], dtype = np.float32)])
    y = np.array([np.linspace(0.0,T,nPred[1], dtype = np.float32)])
    z = np.array([np.linspace(0.0,H,nPred[2], dtype = np.float32)])
    x, y, z = np.meshgrid(x, y, z, indexing = 'ij')
    x1 = np.transpose(np.array([x.flatten()]))
    y1 = np.transpose(np.array([y.flatten()]))
    z1 = np.transpose(np.array([z.flatten()]))
    Grid = np.concatenate((x1, y1, z1), axis=1)      

    totalPts = np.sum(nPred[0]*nPred[1]*nPred[2])
    hist = np.zeros((totalPts,1), dtype = np.float32)
    
    return Grid, x, y, z, hist

def plotPhiStrainEnerg(nPred, xGrid, yGrid, zGrid, phi_pred, frac_energy_pred, iStep):   
    
    nx, ny, nz = nPred[0], nPred[1], nPred[2]    
    phi = phi_pred.reshape(nx, ny, nz)
    fracEnergy = frac_energy_pred.reshape(nx, ny, nz)
    filename = 'Phase_Field_Step_' + iStep
    gridToVTK("./"+filename, xGrid, yGrid, zGrid, pointData = {"Phi" : phi, "Fracture Energy" : fracEnergy})
    
def plotDispStrainEnerg(nPred, xGrid, yGrid, zGrid, u_pred, v_pred, w_pred, elas_energy_pred, iStep):
    
    nx, ny, nz = nPred[0], nPred[1], nPred[2]
    u = u_pred.reshape(nx, ny, nz)
    v = v_pred.reshape(nx, ny, nz)
    w = w_pred.reshape(nx, ny, nz) 
    disp = (u, v, w)
    elasEnergy = elas_energy_pred.reshape(nx, ny, nz)
    filename = 'Elastic_Step_' + iStep
    gridToVTK("./"+filename, xGrid, yGrid, zGrid, pointData = 
              {"Displacement" : disp, "Elastic Energy" : elasEnergy})				
    
def plotConvergence(iter, adam_buff, lbfgs_buff, iStep,figHeight,figWidth):
            
    filename = "Loss_convergence_"
    plt.figure(figsize=(figWidth,figHeight))        
    range_adam = np.arange(1,iter+1)
    range_lbfgs = np.arange(iter+2, iter+2+len(lbfgs_buff))
    ax0, = plt.semilogy(range_adam, adam_buff, c='b', label='Adam',linewidth=2.0)
    ax1, = plt.semilogy(range_lbfgs, lbfgs_buff, c='r', label='L-BFGS',linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Iteration',fontweight='bold',fontsize=14)
    plt.ylabel('Loss value',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename +str(iStep)+".pdf", dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)
        
def plot1dPhi(zPred,phi_pred_1d,phi_exact,iStep,figHeight,figWidth):

    filename = '1dPhi_'
    plt.figure(figsize=(figWidth,figHeight))
    ax0, = plt.plot(zPred.T, phi_pred_1d, label='$\phi_{comp}$', c='b', linewidth=2.0)
    ax1, = plt.plot(zPred.T, phi_exact.T, label='$\phi_{exact}$', c='r',linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('$\phi(x)$',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename + str(iStep) +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close() 

def refineElemVertex(vertex, refList):
    #refines the elements in vertex with indices given by refList by splitting 
    #each element into 8 subdivisions
    #Input: vertex - array of vertices in format [umin, vmin, wmin, umax, vmax, wmax]
    #       refList - list of element indices to be refined
    #Output: newVertex - refined list of vertices
    
    numRef = len(refList)
    newVertex = np.zeros((8*numRef,7))

    for i in range(numRef):        
        elemIndex = refList[i]
        uMin = vertex[elemIndex, 0]
        vMin = vertex[elemIndex, 1]
        wMin = vertex[elemIndex, 2]
        uMax = vertex[elemIndex, 3]
        vMax = vertex[elemIndex, 4]
        wMax = vertex[elemIndex, 5]
        level = vertex[elemIndex, 6]
        uMid = (uMin+uMax)/2
        vMid = (vMin+vMax)/2
        wMid = (wMin+wMax)/2
        newVertex[8*i, :] = [uMin, vMin, wMin, uMid, vMid, wMid, level+1]
        newVertex[8*i+1, :] = [uMid, vMin, wMin, uMax, vMid, wMid, level+1]
        newVertex[8*i+2, :] = [uMin, vMid, wMin, uMid, vMax, wMid, level+1]
        newVertex[8*i+3, :] = [uMid, vMid, wMin, uMax, vMax, wMid, level+1]
        newVertex[8*i+4, :] = [uMin, vMin, wMid, uMid, vMid, wMax, level+1]
        newVertex[8*i+5, :] = [uMid, vMin, wMid, uMax, vMid, wMax, level+1]
        newVertex[8*i+6, :] = [uMin, vMid, wMid, uMid, vMax, wMax, level+1]
        newVertex[8*i+7, :] = [uMid, vMid, wMid, uMax, vMax, wMax, level+1]
    vertex = np.delete(vertex, refList, axis=0)
    newVertex = np.concatenate((vertex, newVertex))

    return newVertex

