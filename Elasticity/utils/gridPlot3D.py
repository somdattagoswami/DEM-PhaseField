import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyevtk.hl import gridToVTK 

def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)

def cart2sph(x, y, z):
    # From https://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r    

def sph2cart(azimuth, elevation, r):
    # From https://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z    

def getExactStresses(x, y, z, model):
    numPts = len(x)
    S = model['P']
    a = model['radInt']
    nu = model['nu']
    
    sigma_xx = np.zeros_like(x)
    sigma_yy = np.zeros_like(x)
    sigma_zz = np.zeros_like(x)
    sigma_xy = np.zeros_like(x)
    sigma_yz = np.zeros_like(x)
    sigma_zx = np.zeros_like(x)
    
    for i in range(numPts):
    
        azimuth, elevation, r  = cart2sph(x[i], y[i], z[i])    
        
        phi = azimuth
        theta = np.pi/2-elevation
        
        sigma_rr = S*np.cos(theta)*np.cos(theta)+S/(7-5*nu)*(a**3/r**3*(6-5*(5-nu)* \
                    np.cos(theta)*np.cos(theta))+(6*a**5)/r**5*(3*np.cos(theta)*np.cos(theta)-1))
        sigma_phiphi = 3*S/(2*(7-5*nu))*(a**3/r**3*(5*nu-2+5*(1-2*nu)*np.cos(theta)* \
                    np.cos(theta))+(a**5)/r**5*(1-5*np.cos(theta)*np.cos(theta)));
        sigma_thth =  S*np.sin(theta)*np.sin(theta)+S/(2*(7-5*nu))*(a**3/r**3* \
                    (4-5*nu+5*(1-2*nu)*np.cos(theta)*np.cos(theta))+(3*a**5)/r**5*\
                    (3-7*np.cos(theta)*np.cos(theta)))
        sigma_rth =  S*(-1+1/(7-5*nu)*(-5*a**3*(1+nu)/(r**3)+(12*a**5)/r**5))*np.sin(theta)*np.cos(theta)

        
        rot_mat = np.array( \
             [[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
              [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)],
              [np.cos(theta), -np.sin(theta), 0.]])
        A = np.array( [[sigma_rr, sigma_rth, 0.], [sigma_rth, sigma_thth, 0.], [0., 0., sigma_phiphi]] )
        stress_cart = rot_mat@A@rot_mat.T
        
        sigma_xx[i] = stress_cart[0,0]
        sigma_yy[i] = stress_cart[1,1]
        sigma_zz[i] = stress_cart[2,2]
        sigma_xy[i] = stress_cart[0,1]
        sigma_zx[i] = stress_cart[0,2]
        sigma_yz[i] = stress_cart[1,2]    

    return sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx

def getExactTraction(x, y, z, xNorm, yNorm, zNorm, model):    
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx = getExactStresses(x[:,0], 
                                                        y[:,0], z[:,0], model)
        
    sigma_xx = np.expand_dims(sigma_xx, axis=1)
    sigma_yy = np.expand_dims(sigma_yy, axis=1)
    sigma_zz = np.expand_dims(sigma_zz, axis=1)
    sigma_xy = np.expand_dims(sigma_xy, axis=1)
    sigma_yz = np.expand_dims(sigma_yz, axis=1)
    sigma_zx = np.expand_dims(sigma_zx, axis=1)
    
    trac_x = xNorm[:,0:1]*sigma_xx + yNorm[:,0:1]*sigma_xy + zNorm[:,0:1]*sigma_zx
    trac_y = xNorm[:,0:1]*sigma_xy + yNorm[:,0:1]*sigma_yy + zNorm[:,0:1]*sigma_yz
    trac_z = xNorm[:,0:1]*sigma_zx + yNorm[:,0:1]*sigma_yz + zNorm[:,0:1]*sigma_zz
    
    return trac_x, trac_y, trac_z

def scatterPlot(X_f,X_bnd,figHeight,figWidth,filename):

    fig = plt.figure(figsize=(figWidth,figHeight))
    ax = fig.gca(projection='3d')
    ax.scatter(X_f[:,0], X_f[:,1], X_f[:,2], s = 0.75)
    ax.scatter(X_bnd[:,0], X_bnd[:,1], X_bnd[:,2], s = 0.75, c='red')
    ax.set_xlabel('$x$',fontweight='bold',fontsize = 12)
    ax.set_ylabel('$y$',fontweight='bold',fontsize = 12)
    ax.set_zlabel('$z$',fontweight='bold',fontsize = 12)
    ax.tick_params(axis='both', which='major', labelsize = 6)
    ax.tick_params(axis='both', which='minor', labelsize = 6)
    plt.savefig(filename+'.png', dpi=300, facecolor='w', edgecolor='w', 
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
    newVertex = np.zeros((8*numRef,6))
    for i in range(numRef):        
        elemIndex = refList[i]
        uMin = vertex[elemIndex, 0]
        vMin = vertex[elemIndex, 1]
        wMin = vertex[elemIndex, 2]
        uMax = vertex[elemIndex, 3]
        vMax = vertex[elemIndex, 4]
        wMax = vertex[elemIndex, 5]
        uMid = (uMin+uMax)/2
        vMid = (vMin+vMax)/2
        wMid = (wMin+wMax)/2
        newVertex[8*i, :] = [uMin, vMin, wMin, uMid, vMid, wMid]
        newVertex[8*i+1, :] = [uMid, vMin, wMin, uMax, vMid, wMid]
        newVertex[8*i+2, :] = [uMin, vMid, wMin, uMid, vMax, wMid]
        newVertex[8*i+3, :] = [uMid, vMid, wMin, uMax, vMax, wMid]
        newVertex[8*i+4, :] = [uMin, vMin, wMid, uMid, vMid, wMax]
        newVertex[8*i+5, :] = [uMid, vMin, wMid, uMax, vMid, wMax]
        newVertex[8*i+6, :] = [uMin, vMid, wMid, uMid, vMax, wMax]
        newVertex[8*i+7, :] = [uMid, vMid, wMid, uMax, vMax, wMax]
    vertex = np.delete(vertex, refList, axis=0)
    newVertex = np.concatenate((vertex, newVertex))
    return newVertex

def energyPlot(xGrid,yGrid,zGrid,nPred,u_pred,v_pred,w_pred,energy_pred,filename):    
    # Plot results        
    oShapeX = np.resize(xGrid, [nPred, nPred, nPred])
    oShapeY = np.resize(yGrid, [nPred, nPred, nPred])
    oShapeZ = np.resize(zGrid, [nPred, nPred, nPred])
    
    u = np.resize(u_pred, [nPred, nPred, nPred])
    v = np.resize(v_pred, [nPred, nPred, nPred])
    w = np.resize(w_pred, [nPred, nPred, nPred])
    displacement = (u, v, w)
    
    elas_energy = np.resize(energy_pred, [nPred, nPred, nPred])

    gridToVTK("./"+ filename, oShapeX, oShapeY, oShapeZ, pointData = 
                  {"Displacement": displacement, "Elastic Energy": elas_energy})  
    
def energyError(X_f,sigma_x_pred,sigma_y_pred,model,sigma_z_pred,tau_xy_pred,tau_yz_pred,tau_zx_pred):
    
    sigma_xx_exact, sigma_yy_exact, sigma_zz_exact, sigma_xy_exact, sigma_yz_exact, \
        sigma_zx_exact = getExactStresses(X_f[:,0], X_f[:,1], X_f[:,2], model)
    sigma_xx_err = sigma_xx_exact - sigma_x_pred[:,0]
    sigma_yy_err = sigma_yy_exact - sigma_y_pred[:,0]
    sigma_zz_err = sigma_zz_exact - sigma_z_pred[:,0]
    sigma_xy_err = sigma_xy_exact - tau_xy_pred[:,0]
    sigma_yz_err = sigma_yz_exact - tau_yz_pred[:,0]
    sigma_zx_err = sigma_zx_exact - tau_zx_pred[:,0]
    
    energy_err = 0
    energy_norm = 0
    numPts = X_f.shape[0]
    
    C_mat = np.zeros((6,6))
    C_mat[0,0] = model['E']*(1-model['nu'])/((1+model['nu'])*(1-2*model['nu']))
    C_mat[1,1] = model['E']*(1-model['nu'])/((1+model['nu'])*(1-2*model['nu']))
    C_mat[2,2] = model['E']*(1-model['nu'])/((1+model['nu'])*(1-2*model['nu']))
    C_mat[0,1] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
    C_mat[0,2] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
    C_mat[1,0] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
    C_mat[1,2] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
    C_mat[2,0] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
    C_mat[2,0] = model['E']*model['nu']/((1+model['nu'])*(1-2*model['nu']))
    C_mat[3,3] = model['E']/(2*(1+model['nu']))
    C_mat[4,4] = model['E']/(2*(1+model['nu']))
    C_mat[5,5] = model['E']/(2*(1+model['nu']))
    
    C_inv = np.linalg.inv(C_mat)
    for i in range(numPts):
        err_pt = np.array([sigma_xx_err[i], sigma_yy_err[i], sigma_zz_err[i], 
                           sigma_xy_err[i], sigma_yz_err[i], sigma_zx_err[i]])
        norm_pt = np.array([sigma_xx_exact[i], sigma_yy_exact[i], sigma_zz_exact[i], \
                            sigma_xy_exact[i], sigma_yz_exact[i], sigma_zx_exact[i]])
        energy_err = energy_err + err_pt@C_inv@err_pt.T*X_f[i,3]
        energy_norm = energy_norm + norm_pt@C_inv@norm_pt.T*X_f[i,3]
        
    return energy_err, energy_norm