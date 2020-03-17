import os
import numpy as np
import matplotlib.pyplot as plt

def scatterPlot(X_f,figHeight,figWidth,filename):

    Yaxis =  np.zeros((X_f.shape[0],1),dtype = np.float64)
    plt.figure(figsize=(figWidth,figHeight))
    plt.scatter(X_f[:,0:1], Yaxis[:,0:1],  marker='o', color='black', s= 0.15)
    plt.tight_layout()
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
   
def plotConvergence(iter,adam_buff,lbfgs_buff,figHeight,figWidth):
    
    filename = "Loss_convergence_"
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
    
def plot1dPhi(xSpace, phi_pred, phi_exact, figHeight,figWidth):  

    filename = '1D_phi_4th'
    plt.figure(figsize=(figWidth,figHeight)) 
    ax0, = plt.plot(xSpace, phi_pred, label='$\phi_{comp}$', c='b', linewidth=2.0)
    ax1, = plt.plot(xSpace, phi_exact, label='$\phi_{exact}$', c='r',linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    #plt.title('$\phi_{comp}$ and $\phi_{exact}$')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('$\phi(x)$',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def plot1dU(xSpace, u_pred, u_exact, figHeight,figWidth):  
    
    filename = '1D_u_4th'
    plt.figure(figsize=(figWidth,figHeight)) 
    axisLine =  np.zeros((xSpace.shape[0],1),dtype = np.float64)
    ax0, = plt.plot(xSpace, u_pred, label='$u_{comp}$', c='b', linewidth=2.0)
    ax1, = plt.plot(xSpace, u_exact, label='$u_{exact}$', c='r', linewidth=2.0)
    plt.legend(handles=[ax0,ax1],fontsize=14)
    #plt.title('$u_{comp}$ and $u_{exact}$')
    plt.plot(xSpace, axisLine, c='black', linewidth=2.0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('x',fontweight='bold',fontsize=14)
    plt.ylabel('$u(x)$',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename +".pdf", dpi=700, facecolor='w', edgecolor='w', 
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
    #each element into 2 subdivisions
    #Input: vertex - array of vertices in format [umin,umax]
    #       refList - list of element indices to be refined
    #Output: newVertex - refined list of vertices
    
    numRef = len(refList)
    newVertex = np.zeros((2*numRef,2))
    for i in range(numRef):        
        elemIndex = refList[i]
        uMin = vertex[elemIndex, 0]
        uMax = vertex[elemIndex, 1]
        uMid = (uMin+uMax)/2
        newVertex[2*i, :] = [uMin, uMid]
        newVertex[2*i+1, :] = [uMid, uMax]
    vertex = np.delete(vertex, refList, axis=0)
    newVertex = np.concatenate((vertex, newVertex),axis=0)
    
    return newVertex