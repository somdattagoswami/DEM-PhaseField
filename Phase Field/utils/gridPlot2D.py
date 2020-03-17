import os
import numpy as np
import matplotlib.pyplot as plt

def scatterPlot(X_f,figHeight,figWidth,filename):
    
    plt.figure(figsize=(figWidth,figHeight))
    plt.scatter(X_f[:,0], X_f[:,1],s=0.5)
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename +'.pdf',dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()

def genGrid(nPred,L,secBound):
    
    xCrackDown = np.linspace(0.0, L, nPred[0,0], dtype = np.float32)
    yCrackDown = np.linspace(secBound[0,0], secBound[0,1], nPred[0,1], dtype = np.float32)
    xCD, yCD = np.meshgrid(xCrackDown, yCrackDown)
    xCD = np.array([xCD.flatten()])
    yCD = np.array([yCD.flatten()])
    X_CD = np.concatenate((xCD.T, yCD.T), axis=1)
    
    xCrack = np.linspace(0.0, L, nPred[1,0], dtype = np.float32)
    yCrack = np.linspace(secBound[1,0], secBound[1,1], nPred[1,1], dtype = np.float32)
    xC, yC = np.meshgrid(xCrack, yCrack)
    xC = np.array([xC.flatten()])
    yC = np.array([yC.flatten()])
    X_C = np.concatenate((xC.T, yC.T), axis=1)
    
    xCrackUp = np.linspace(0.0, L, nPred[2,0], dtype = np.float32)
    yCrackUp = np.linspace(secBound[2,0], secBound[2,1], nPred[2,1], dtype = np.float32)
    xCU, yCU = np.meshgrid(xCrackUp, yCrackUp)
    xCU = np.array([xCU.flatten()])
    yCU = np.array([yCU.flatten()])
    X_CU = np.concatenate((xCU.T, yCU.T), axis=1)      

    Grid = np.concatenate((X_CD,X_C),axis=0)
    Grid = np.concatenate((Grid,X_CU),axis=0)
    xGrid = np.transpose(np.array([Grid[:,0]]))
    yGrid = np.transpose(np.array([Grid[:,1]]))
    totalPts = np.sum(nPred[:,0]*nPred[:,1])
    hist = np.zeros((totalPts,1), dtype = np.float32)
    
    return Grid, xGrid, yGrid, hist

def plotPhiStrainEnerg(nPred,xGrid,yGrid,phi_pred,frac_energy_pred,iStep,figHeight,figWidth):
    
    # Removing the negative values of phi
    index = np.where(phi_pred[:,0] < 0.0)
    np.put(phi_pred, index[0], [0.0])
    index = np.where(phi_pred[:,0] > 1.0)
    np.put(phi_pred, index[0], [1.0])
    
    phi_min = min(phi_pred)
    phi_max = max(phi_pred)
    frac_energy_min = min(frac_energy_pred)
    frac_energy_max = max(frac_energy_pred)
    
    # Plot results
    oShapeX_CD = np.resize(xGrid[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])
    oShapeY_CD = np.resize(yGrid[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])
    phi_CD = np.resize(phi_pred[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])
    frac_energy_CD = np.resize(frac_energy_pred[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])
    
    oShapeX_C = np.resize(xGrid[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])
    oShapeY_C = np.resize(yGrid[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])
    phi_C = np.resize(phi_pred[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])
    frac_energy_C = np.resize(frac_energy_pred[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])

    oShapeX_CU = np.resize(xGrid[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])
    oShapeY_CU = np.resize(yGrid[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])
    phi_CU = np.resize(phi_pred[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])
    frac_energy_CU = np.resize(frac_energy_pred[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])

    # Plotting phi
    filename = "Phi_"+ iStep
    plt.figure(figsize=(figWidth, figHeight))
    cbarlabels = np.linspace(0.0, 1.0, 255, endpoint=True)
    cbarticks = np.linspace(0.0, 1.0, 15, endpoint=True)       
    plt.contourf(oShapeX_CD, oShapeY_CD, phi_CD, cbarlabels, vmin = phi_min, vmax = phi_max, cmap=plt.cm.jet)
    plt.contourf(oShapeX_C, oShapeY_C, phi_C, cbarlabels, vmin = phi_min, vmax = phi_max, cmap=plt.cm.jet)
    plt.contourf(oShapeX_CU, oShapeY_CU, phi_CU, cbarlabels, vmin = phi_min, vmax = phi_max, cmap=plt.cm.jet)
    cbar = plt.colorbar(ticks = cbarticks)
    cbar.ax.tick_params(labelsize=14) 
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename +".png",dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
   
    # Plotting the strain energy densities
    filename = "Fracture_energy_" + iStep
    plt.figure(figsize=(figWidth, figHeight))
    plt.contourf(oShapeX_CD, oShapeY_CD, frac_energy_CD, 255, vmin = frac_energy_min, vmax = frac_energy_max, cmap=plt.cm.jet)
    plt.contourf(oShapeX_C, oShapeY_C, frac_energy_C, 255, vmin = frac_energy_min, vmax = frac_energy_max, cmap=plt.cm.jet)
    plt.contourf(oShapeX_CU, oShapeY_CU, frac_energy_CU, 255, vmin = frac_energy_min, vmax = frac_energy_max, cmap=plt.cm.jet)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    #plt.title("Fracture Energy Density for "+str(iStep)+" and with convergernce step " +str(nIter))
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename + ".png",dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def plotDispStrainEnerg(nPred,xGrid,yGrid,u_pred,v_pred,elas_energy_pred,iStep,figHeight,figWidth):
    
    # Magnification factors for plotting the deformed shape
    x_fac = 50
    y_fac = 50
    
    v_min = min(v_pred)
    v_max = max(v_pred)
    elas_energy_min = min(elas_energy_pred)
    elas_energy_max = max(elas_energy_pred)
    
    # Compute the approximate displacements at plot points     
    oShapeX_CD = np.resize(xGrid[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])
    oShapeY_CD = np.resize(yGrid[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])
    surfaceUx_CD = np.resize(u_pred[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])
    surfaceUy_CD = np.resize(v_pred[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])
    defShapeX_CD = oShapeX_CD + surfaceUx_CD * x_fac
    defShapeY_CD = oShapeY_CD + surfaceUy_CD * y_fac
    elas_energy_CD = np.resize(elas_energy_pred[0 : nPred[0,0] * nPred[0,1], 0], [nPred[0,1], nPred[0,0]])

    oShapeX_C = np.resize(xGrid[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])
    oShapeY_C = np.resize(yGrid[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])
    surfaceUx_C = np.resize(u_pred[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])
    surfaceUy_C = np.resize(v_pred[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])
    defShapeX_C = oShapeX_C + surfaceUx_C * x_fac
    defShapeY_C = oShapeY_C + surfaceUy_C * y_fac
    elas_energy_C = np.resize(elas_energy_pred[nPred[0,0] * nPred[0,1] : (nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]), 0], [nPred[1,1], nPred[1,0]])
    
    oShapeX_CU = np.resize(xGrid[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])
    oShapeY_CU = np.resize(yGrid[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])
    surfaceUx_CU = np.resize(u_pred[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])
    surfaceUy_CU = np.resize(v_pred[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])
    defShapeX_CU = oShapeX_CU + surfaceUx_CU * x_fac
    defShapeY_CU = oShapeY_CU + surfaceUy_CU * y_fac
    elas_energy_CU = np.resize(elas_energy_pred[(nPred[0,0]*nPred[0,1]) + (nPred[1,0]*nPred[1,1]):, 0], [nPred[2,1], nPred[2,0]])
    
    # Plotting the y-displacement
    filename = "Displacement_Y_"+ iStep
    plt.figure(figsize=(figWidth, figHeight))
    plt.contourf(defShapeX_CD, defShapeY_CD, surfaceUy_CD, 255, vmin = v_min, vmax = v_max, cmap=plt.cm.jet)
    plt.contourf(defShapeX_C, defShapeY_C, surfaceUy_C, 255, vmin = v_min, vmax = v_max, cmap=plt.cm.jet)
    plt.contourf(defShapeX_CU, defShapeY_CU, surfaceUy_CU, 255, vmin = v_min, vmax = v_max, cmap=plt.cm.jet)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14) 
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename + ".png",dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    # Plotting the strain energy densities
    filename = "Elastic_energy_"+ iStep
    plt.figure(figsize=(figWidth, figHeight))
    plt.contourf(defShapeX_CD, defShapeY_CD, elas_energy_CD, 255, vmin = elas_energy_min, vmax = elas_energy_max, cmap=plt.cm.jet)
    plt.contourf(defShapeX_C, defShapeY_C, elas_energy_C, 255, vmin = elas_energy_min, vmax = elas_energy_max, cmap=plt.cm.jet)
    plt.contourf(defShapeX_CU, defShapeY_CU, elas_energy_CU, 255, vmin = elas_energy_min, vmax = elas_energy_max, cmap=plt.cm.jet)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14) 
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename + ".png",dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    filename = "Scatter_"+iStep
    plt.figure(figsize=(figWidth, figHeight))
    plt.scatter(defShapeX_C, defShapeY_C, s=0.5)
    plt.scatter(defShapeX_CU, defShapeY_CU, s=0.5)
    plt.scatter(defShapeX_CD, defShapeY_CD, s=0.5)
    plt.xlabel('$x$',fontweight='bold',fontsize=14)
    plt.ylabel('$y$',fontweight='bold',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename +".pdf",dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()					
    
def plotConvergence(iter,adam_buff,lbfgs_buff,iStep,figHeight,figWidth):
    
    filename = "Loss_convergence_"+ str(iStep)
    plt.figure(figsize=(figWidth, figHeight))        
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
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def plotForceDisp(fdGraph,figHeight,figWidth):           
    
    filename = "Force-Displacement"
    plt.figure(figsize=(figWidth, figHeight))
    plt.plot(fdGraph[:,0], fdGraph[:,1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Displacement',fontweight='bold',fontsize=14)
    plt.ylabel('Force',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename + ".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def plot1dPhi(yPred,phi_pred_1d,phi_exact,iStep,figHeight,figWidth):
    
    filename = '1dPhi_'+ str(iStep)
    plt.figure(figsize=(figWidth, figHeight))
    ax0, = plt.plot(yPred.T, phi_pred_1d, label='$\phi_{comp}$', c='b', linewidth=2.0)
    ax1, = plt.plot(yPred.T, phi_exact.T, label='$\phi_{exact}$', c='r',linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('$\phi(x)$',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def plot1dPhisymmDN(yPred,phi_pred_1d,phi_exact,iStep,xVal,figHeight,figWidth):
    
    filename = '1dPhi_'
    plt.figure(figsize=(figWidth, figHeight))
    ax0, = plt.plot(yPred.T, phi_pred_1d, label='$\phi_{comp}$', c='b', linewidth=2.0)
    ax1, = plt.plot(yPred.T, phi_exact.T, label='$\phi_{exact}$', c='r',linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('$\phi(x)$',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename + str(iStep) + "xVal" +str(xVal) +".png", 
                        dpi=700, facecolor='w', edgecolor='w', 
                        transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)

def refineElemVertex(vertex, refList):
    #refines the elements in vertex with indices given by refList by splitting 
    #each element into 4 subdivisions
    #Input: vertex - array of vertices in format [umin, vmin, umax, vmax]
    #       refList - list of element indices to be refined
    #Output: newVertex - refined list of vertices
    
    numRef = len(refList)
    newVertex = np.zeros((4*numRef,5))
    for i in range(numRef):        
        elemIndex = refList[i]
        uMin = vertex[elemIndex, 0]
        vMin = vertex[elemIndex, 1]
        uMax = vertex[elemIndex, 2]
        vMax = vertex[elemIndex, 3]
        level = vertex[elemIndex, 4]
        uMid = (uMin+uMax)/2
        vMid = (vMin+vMax)/2
        newVertex[4*i, :] = [uMin, vMin, uMid, vMid, level+1]
        newVertex[4*i+1, :] = [uMid, vMin, uMax, vMid, level+1]
        newVertex[4*i+2, :] = [uMin, vMid, uMid, vMax, level+1]
        newVertex[4*i+3, :] = [uMid, vMid, uMax, vMax, level+1]
    vertex = np.delete(vertex, refList, axis=0)
    newVertex = np.concatenate((vertex, newVertex),axis=0)
    
    return newVertex

def refineElemRegion(vertex, refYmin, refYmax, refXmax, X_f, ref_elem):
    #refines the region bounded by refYmin < y < refYmax
    #Input: vertex - array of vertices in format [umin, vmin, umax, vmax]
    #       refYmin - lower bound of the refinement region
    #       refYmax - upper bound of the refinement region
    #Output: newVertex - new list of vertices
    tol = 1e-4 #tolerance for equality
    refYmax = refYmax+tol
    refYmin = refYmin-tol
    refXmax = refXmax+tol
    
    index = []
    for iVertex in range(0,X_f.shape[0]):
        if (X_f[iVertex,1] >= refYmin) and (X_f[iVertex,1] <= refYmax) and (X_f[iVertex,0] <= refXmax):
            index.append(iVertex)
       
    elem_refine = np.squeeze(ref_elem[index]) # Obtains the element numbers of the gauss points that need to be refined
    elem_refine = np.unique(elem_refine)
    elmt_nos = np.transpose(np.array([vertex[:,0]],dtype = np.int32)) # Extracting the 1st column of the vertex matrix to convert to int32
    index_ref = np.where(elmt_nos == elem_refine)[0] # Obtaining the index numbers in the vertex matrix corresponding to the element numbers            
    newVertex = refineElemVertex(vertex, index_ref)
    
    return newVertex