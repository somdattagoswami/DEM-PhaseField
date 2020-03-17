# -*- coding: utf-8 -*-
"""
File for base geometry class built using the Geomdl class
"""

import numpy as np
from geomdl import NURBS
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
#import sys
#from geomdl.visualization import VisMPL
class Geometry1D:
    '''
     Base class for 1D domains
     Input: geomData - dictionary containing the geomety information
     Keys: degree_u: polynomial degree in the u direction
       ctrlpts_size_u: number of control points in u direction
       ctrlpts: weighted control points (in a list with 
            ctrlpts_size_u rows and 3 columns for x,y,z coordinates)
       weights: correspond weights (list with ctrlpts_size_u*ctrlpts_size_v entries)
       knotvector_u: knot vectors in the u direction
    '''
    def __init__(self, geomData):
        self.curv = NURBS.Curve()
        self.curv.degree = geomData['degree_u']
#        self.curv.ctrlpts_size = geomData['ctrlpts_size_u']
        self.curv.ctrlpts = self.getUnweightedCpts(geomData['ctrlpts'], 
                                             geomData['weights'])
        self.curv.weights = geomData['weights']
        self.curv.knotvector = geomData['knotvector_u']

    def getUnweightedCpts(self, ctrlpts, weights):
        numCtrlPts = np.shape(ctrlpts)[0]
        PctrlPts = np.zeros_like(ctrlpts)
        for i in range(3):
            for j in range(numCtrlPts):
                PctrlPts[j,i]=ctrlpts[j][i]/weights[j]
        PctrlPts = PctrlPts.tolist()
        return PctrlPts
        
    def mapPoints(self, uPar):
        '''
        Map points from the parameter domain [0,1]x[0,1] to the quadrilater domain
        Input:  uPar - array containing the u-coordinates in the parameter space
        Output: xPhys - array containing the x-coordinates in the physical space
        '''        
        gpParamU = np.array([uPar])
        evalList = tuple(map(tuple, gpParamU.transpose()))
        res = np.array(self.curv.evaluate_list(evalList))
                
        return res
    
    def getUnifIntPts(self, numPtsU, withEdges):
        '''
        Generate uniformly spaced points inside the domain
        Input: numPtsU - number of points (including edges) in the u direction in the parameter space
               withEdges - 1x2 array of zeros or ones specifying whether the boundary points
                           should be included. The boundary order is [bottom, right,
                           top, left] for the unit square.
        Output: xM, yM - flattened array containing the x and y coordinates of the points
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(0, 1, numPtsU)
        
        #remove endpoints depending on values of withEdges
        if withEdges[0]==0:
            uEdge = uEdge[:-1]
        if withEdges[1]==0:
            uEdge = uEdge[1:]

        #map points
        res = self.mapPoints(uEdge.T)
        
        xPhys = res[:, 0:1]
        
        return xPhys
    
    def getIntPts(self, numElemU, numGauss):
        '''
        Generate quadrature points inside the domain
        Input: numElemU - number of subdivisions in the u 
                   direction in the parameter space
               numGauss - number of Gauss quadrature points for each subdivision
        Output: xPhys, wgtPhy - arrays containing the x coordinate
                                    of the points and the corresponding weights
        '''
        # Allocate quadPts array
        quadPts = np.zeros((numElemU*numGauss, 2))
        vertex = np.zeros((numElemU, 2))
        # Get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)
        
        # Generate the knots on the interval [0,1]
        uEdge = np.linspace(0, 1, numElemU+1)
        uPar = uEdge              
                        
        # Generate points for each element
        indexPt = 0
        for iU in range(numElemU):
            uMin = uPar[iU]
            uMax = uPar[iU+1]
            vertex[iU,0] = uMin
            vertex[iU,1] = uMax
            gpParamU = (uMax-uMin)/2*gp+(uMax+uMin)/2
            # Jacobian of the transformation from the reference element [-1,1]
            scaleFac = (uMax-uMin)/2
            # Map the points to the physical space
            for iPt in range(numGauss):
                curPtU = gpParamU[iPt]
                derivMat = self.curv.derivatives(curPtU, order=1)
                physPtX = derivMat[0][0]
                derivU = derivMat[1][0:1]
                JacobMat = np.array([derivU])
                detJac = np.linalg.det(JacobMat)
                quadPts[indexPt, 0] = physPtX
                quadPts[indexPt, 1] = scaleFac * detJac * gw[iPt]
                indexPt = indexPt + 1
                            
        xPhys = quadPts[:, 0:1]
        wgtPhys = quadPts[:, 1:2]

        return xPhys, wgtPhys, vertex

class Geometry2D:
    '''
     Base class for 2D domains
     Input: geomData - dictionary containing the geomety information
     Keys: degree_u, degree_v: polynomial degree in the u and v directions
       ctrlpts_size_u, ctrlpts_size_v: number of control points in u,v directions
       ctrlpts: weighted control points (in a list with 
            ctrlpts_size_u*ctrlpts_size_v rows and 3 columns for x,y,z coordinates)
       weights: correspond weights (list with ctrlpts_size_u*ctrlpts_size_v entries)
       knotvector_u, knotvector_v: knot vectors in the u and v directions
    '''
    def __init__(self, geomData):
        self.surf = NURBS.Surface()
        self.surf.degree_u = geomData['degree_u']
        self.surf.degree_v = geomData['degree_v']
        self.surf.ctrlpts_size_u = geomData['ctrlpts_size_u']
        self.surf.ctrlpts_size_v = geomData['ctrlpts_size_v']
        self.surf.ctrlpts = self.getUnweightedCpts(geomData['ctrlpts'], 
                                             geomData['weights'])
        self.surf.weights = geomData['weights']
        self.surf.knotvector_u = geomData['knotvector_u']
        self.surf.knotvector_v = geomData['knotvector_v']                
        
    def getUnweightedCpts(self, ctrlpts, weights):
        numCtrlPts = np.shape(ctrlpts)[0]
        PctrlPts = np.zeros_like(ctrlpts)
        for i in range(3):
            for j in range(numCtrlPts):
                PctrlPts[j,i]=ctrlpts[j][i]/weights[j]
        PctrlPts = PctrlPts.tolist()
        return PctrlPts
        
    def mapPoints(self, uPar, vPar):
        '''
        Map points from the parameter domain [0,1]x[0,1] to the quadrilater domain
        Input:  uPar - array containing the u-coordinates in the parameter space
                vPar - array containing the v-coordinates in the parameter space
                Note: the arrays uPar and vPar must be of the same size
        Output: xPhys - array containing the x-coordinates in the physical space
                yPhys - array containing the y-coordinates in the physical space
        '''        
        gpParamUV = np.array([uPar, vPar])
        evalList = tuple(map(tuple, gpParamUV.transpose()))
        res = np.array(self.surf.evaluate_list(evalList))
                
        return res
    
    def getUnifIntPts(self, numPtsU, numPtsV, withEdges):
        '''
        Generate uniformly spaced points inside the domain
        Input: numPtsU, numPtsV - number of points (including edges) in the u and v
                   directions in the parameter space
               withEdges - 1x4 array of zeros or ones specifying whether the boundary points
                           should be included. The boundary order is [bottom, right,
                           top, left] for the unit square.
        Output: xM, yM - flattened array containing the x and y coordinates of the points
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(0, 1, numPtsU)
        vEdge = np.linspace(0, 1, numPtsV)
        
        #remove endpoints depending on values of withEdges
        if withEdges[0]==0:
            vEdge = vEdge[1:]
        if withEdges[1]==0:
            uEdge = uEdge[:-1]
        if withEdges[2]==0:
            vEdge = vEdge[:-1]
        if withEdges[3]==0:
            uEdge = uEdge[1:]
            
        #create meshgrid
        uPar, vPar = np.meshgrid(uEdge, vEdge)        
                        
        uPar = uPar.flatten()
        vPar = vPar.flatten()     
        #map points
        res = self.mapPoints(uPar.T, vPar.T)
        
        xPhys = res[:, 0:1]
        yPhys = res[:, 1:2]
        
        return xPhys, yPhys
    
    def genElemList(self, numElemU, numElemV):
        '''
        Generate quadrature points inside the domain for an initial (uniform)
        subdivision mesh
        Input: numElemU, numElemV - number of subdivisions in the u and v
                   directions in the parameter space               
        Output: vertex - arrays containing the element vertices
        '''
        vertex = np.zeros((numElemU*numElemV, 4))
                        
        #generate the knots on the interval [0,1]
        uEdge = np.linspace(0, 1, numElemU+1)
        vEdge = np.linspace(0, 1, numElemV+1)            
#        totalGaussPts = numGauss**2
        #create meshgrid
        uPar, vPar = np.meshgrid(uEdge, vEdge)              
        counterElem = 0                
        #generate points for each element

        for iV in range(numElemV):
            for iU in range(numElemU):
                uMin = uPar[iV, iU]
                uMax = uPar[iV, iU+1]
                vMin = vPar[iV, iU]
                vMax = vPar[iV+1, iU]                
                vertex[counterElem, 0] = uMin
                vertex[counterElem, 1] = vMin
                vertex[counterElem, 2] = uMax
                vertex[counterElem, 3] = vMax
                counterElem = counterElem + 1
                                        
        return vertex
    
    def getElemIntPts(self, vertex, numGauss):
        '''
        Generate quadrature points inside the domain
        Input: vertex - contains the vertices of the elements the refined elements
               numGauss - number of Gauss quadrature points for each subdivision
        Output: xPhys, yPhys, wgtPhy - arrays containing the x and y coordinates
                                    of the points and the corresponding weights
        '''
        #allocate quadPts array
        numElems = vertex.shape[0]
        quadPts = np.zeros((numElems*numGauss**2, 3))
        
        # Get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)
        
        #get the Gauss weights on the reference element [-1, 1]x[-1,1]
        gpWeightU, gpWeightV = np.meshgrid(gw, gw)
        gpWeightUV = np.array(gpWeightU.flatten()*gpWeightV.flatten())
               
        #generate points for each element
        indexPt = 0
        for iElem in range(numElems):
            
            uMin = vertex[iElem,0]
            uMax = vertex[iElem,2]
            vMin = vertex[iElem,1]
            vMax = vertex[iElem,3]
            gpParamU = (uMax-uMin)/2*gp+(uMax+uMin)/2
            gpParamV = (vMax-vMin)/2*gp+(vMax+vMin)/2
            gpParamUg, gpParamVg = np.meshgrid(gpParamU, gpParamV)
            gpParamUV = np.array([gpParamUg.flatten(), gpParamVg.flatten()])
            #Jacobian of the transformation from the reference element [-1,1]
            scaleFac = (uMax-uMin)*(vMax-vMin)/4
                
                #map the points to the physical space
            for iPt in range(numGauss**2):
                curPtU = gpParamUV[0, iPt]
                curPtV = gpParamUV[1, iPt]
                derivMat = self.surf.derivatives(curPtU, curPtV, order=1)
                physPtX = derivMat[0][0][0]
                physPtY = derivMat[0][0][1]
                derivU = derivMat[1][0][0:2]
                derivV = derivMat[0][1][0:2]
                JacobMat = np.array([derivU,derivV])
                detJac = np.linalg.det(JacobMat)
                quadPts[indexPt, 0] = physPtX
                quadPts[indexPt, 1] = physPtY
                quadPts[indexPt, 2] = scaleFac * detJac * gpWeightUV[iPt]
                indexPt = indexPt + 1
        
        xPhys = quadPts[:, 0:1]
        yPhys = quadPts[:, 1:2]
        wgtPhys = quadPts[:, 2:3]
        
        return xPhys, yPhys, wgtPhys
    
    def getUnweightedCpts2d(self, ctrlpts2d, weights):
        numCtrlPtsU = np.shape(ctrlpts2d)[0]
        numCtrlPtsV = np.shape(ctrlpts2d)[1]
        PctrlPts = np.zeros([numCtrlPtsU,numCtrlPtsV,3])
        counter = 0    
        for j in range(numCtrlPtsU):
            for k in range(numCtrlPtsV):
                for i in range(3):
                    PctrlPts[j,k,i]=ctrlpts2d[j][k][i]/weights[counter]
                counter = counter + 1
        PctrlPts = PctrlPts.tolist()
        return PctrlPts
    
    
    def plotSurf(self):
        #plots the NURBS/B-Spline surface and the control points in 2D
        fig, ax = plt.subplots()
        patches = []
            
        #get the number of points in the u and v directions
        numPtsU = np.int(1/self.surf.delta[0])
        numPtsV = np.int(1/self.surf.delta[1])
        
        for j in range(numPtsV):
            for i in range(numPtsU):
                #get the index of point in the lower left corner of the visualization element
                indexPtSW = j*(numPtsU+1) + i
                indexPtSE = indexPtSW + 1
                indexPtNE = indexPtSW + numPtsU + 2
                indexPtNW = indexPtSW + numPtsU + 1
                XYPts = np.array(self.surf.evalpts)[[indexPtSW, indexPtSE, 
                                indexPtNE, indexPtNW],0:2]
                poly = mpatches.Polygon(XYPts)#, ec="none")
                patches.append(poly)
                
                
        collection = PatchCollection(patches, color="lightgreen", cmap=plt.cm.hsv, alpha=1)
        ax.add_collection(collection)
        
        numCtrlPtsU = self.surf._control_points_size[0]
        numCtrlPtsV = self.surf._control_points_size[1]
        ctrlpts = self.getUnweightedCpts2d(self.surf.ctrlpts2d, self.surf.weights)
        #plot the horizontal lines
        for j in range(numCtrlPtsU):
            plt.plot(np.array(ctrlpts)[j,:,0],np.array(ctrlpts)[j,:,1],ls='--',color='black')
        #plot the vertical lines
        for i in range(numCtrlPtsV):
            plt.plot(np.array(ctrlpts)[:,i,0],np.array(ctrlpts)[:,i,1],ls='--',color='black')
        #plot the control points
        plt.scatter(np.array(self.surf.ctrlpts)[:,0],np.array(self.surf.ctrlpts)[:,1],color='red',zorder=10)
        plt.axis('equal')
        
    def plotKntSurf(self):
        #plots the NURBS/B-Spline surface and the knot lines in 2D
        fig, ax = plt.subplots()
        patches = []
        
        #get the number of points in the u and v directions
        self.surf.delta = 0.02
        self.surf.evaluate()
        numPtsU = np.int(1/self.surf.delta[0])
        numPtsV = np.int(1/self.surf.delta[1])
        
        for j in range(numPtsV):
            for i in range(numPtsU):
                #get the index of point in the lower left corner of the visualization element
                indexPtSW = j*(numPtsU+1) + i
                indexPtSE = indexPtSW + 1
                indexPtNE = indexPtSW + numPtsU + 2
                indexPtNW = indexPtSW + numPtsU + 1
                XYPts = np.array(self.surf.evalpts)[[indexPtSW, indexPtSE, indexPtNE, indexPtNW],0:2]
                poly = mpatches.Polygon(XYPts)
                patches.append(poly)
                
        collection = PatchCollection(patches, color="lightgreen", cmap=plt.cm.hsv, alpha=1)
        ax.add_collection(collection)
        
        #plot the horizontal knot lines
        for j in np.unique(self.surf.knotvector_u):
            vVal = np.linspace(0, 1, numPtsV)
            uVal = np.ones(numPtsV)*j    
            uvVal = np.array([uVal, vVal])
            
            evalList=tuple(map(tuple, uvVal.transpose()))
            res=np.array(self.surf.evaluate_list(evalList))        
            plt.plot(res[:,0],res[:,1], ls='-', linewidth=1, color='black')
            
        #plot the vertical lines
        for i in np.unique(self.surf.knotvector_v):
            uVal = np.linspace(0, 1, numPtsU)
            vVal = np.ones(numPtsU)*i    
            uvVal = np.array([uVal, vVal])
            
            evalList=tuple(map(tuple, uvVal.transpose()))
            res=np.array(self.surf.evaluate_list(evalList))        
            plt.plot(res[:,0],res[:,1], ls='-', linewidth=1, color='black')
       
        plt.axis('equal')        
    
    def getQuadEdgePts(self, numElem, numGauss, orient):
        '''
        Generate points on the boundary edge given by orient
        Input: numElem - number of number of subdivisions (in the v direction)
               numGauss - number of Gauss points per subdivision
               orient - edge orientation in parameter space: 1 is down (v=0), 
                        2 is left (u=1), 3 is top (v=1), 4 is right (u=0)
        Output: xBnd, yBnd, wgtBnd - coordinates of the boundary in the physical
                                     space and the corresponding weights
                xNorm, yNorm  - x and y component of the outer normal vector
        '''
                #allocate quadPts array
        quadPts = np.zeros((numElem*numGauss, 5))
        
        #get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)        
        
        #generate the knots on the interval [0,1]
        edgePar = np.linspace(0, 1, numElem+1)            
                        
        #generate points for each element
        indexPt = 0
        for iE in range(numElem):                
                edgeMin = edgePar[iE]
                edgeMax = edgePar[iE+1]
                if orient==1:
                    gpParamU = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                    gpParamV = np.zeros_like(gp)                    
                elif orient==2:
                    gpParamU = np.ones_like(gp)
                    gpParamV = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                elif orient==3:
                    gpParamU = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                    gpParamV = np.ones_like(gp)   
                elif orient==4:
                    gpParamU = np.zeros_like(gp)
                    gpParamV = (edgeMax-edgeMin)/2*gp+(edgeMax+edgeMin)/2
                else:
                    raise Exception('Wrong orientation given')
                        
                gpParamUV = np.array([gpParamU.flatten(), gpParamV.flatten()])
                
                #Jacobian of the transformation from the reference element [-1,1]
                scaleFac = (edgeMax-edgeMin)/2
                
                #map the points to the physical space
                for iPt in range(numGauss):
                    curPtU = gpParamUV[0, iPt]
                    curPtV = gpParamUV[1, iPt]
                    derivMat = self.surf.derivatives(curPtU, curPtV, order=1)
                    physPtX = derivMat[0][0][0]
                    physPtY = derivMat[0][0][1]
                    derivU = derivMat[1][0][0:2]
                    derivV = derivMat[0][1][0:2]
                    JacobMat = np.array([derivU,derivV])
                    if orient==1:                                                
                        normX = JacobMat[0,1]
                        normY = -JacobMat[0,0]
                    elif orient==2:
                        normX = JacobMat[1,1]
                        normY = -JacobMat[1,0]
                    elif orient==3:
                        normX = -JacobMat[0,1]
                        normY = JacobMat[0,0]
                    elif orient==4:
                        normX = -JacobMat[1,1]
                        normY = JacobMat[1,0]
                    else:
                        raise Exception('Wrong orientation given')
                        
                    JacobEdge = np.sqrt(normX**2+normY**2)
                    normX = normX/JacobEdge
                    normY = normY/JacobEdge
        
                    quadPts[indexPt, 0] = physPtX
                    quadPts[indexPt, 1] = physPtY
                    quadPts[indexPt, 2] = normX
                    quadPts[indexPt, 3] = normY
                    quadPts[indexPt, 4] = scaleFac * JacobEdge * gw[iPt]
                    indexPt = indexPt + 1
                            
        xPhys = quadPts[:, 0:1]
        yPhys = quadPts[:, 1:2]
        xNorm = quadPts[:, 2:3]
        yNorm = quadPts[:, 3:4]
        wgtPhys = quadPts[:, 4:5]        
        
        return xPhys, yPhys, xNorm, yNorm, wgtPhys
    
class Geometry3D:
    '''
     Base class for 3D domains
     Input: geomData - dictionary containing the geomety information
     Keys: degree_u, degree_v, degree_w: polynomial degree in the u, v, w directions
       ctrlpts_size_u, ctrlpts_size_v, ctrlpts_size_w: number of control points in u,v,w directions
       ctrlpts: weighted control points (in a list with 
            ctrlpts_size_u*ctrlpts_size_v*ctrlpts_size_w rows and 3 columns for x,y,z coordinates)
       weights: correspond weights (list with ctrlpts_size_u*ctrlpts_size_v*ctrlpts_size_w entries)
       knotvector_u, knotvector_v, knotvector_w: knot vectors in the u, v, w directions
    '''
    def __init__(self, geomData):
        self.vol = NURBS.Volume()
        self.vol.degree_u = geomData['degree_u']
        self.vol.degree_v = geomData['degree_v']
        self.vol.degree_w = geomData['degree_w']
        self.vol.ctrlpts_size_u = geomData['ctrlpts_size_u']
        self.vol.ctrlpts_size_v = geomData['ctrlpts_size_v']
        self.vol.ctrlpts_size_w = geomData['ctrlpts_size_w']
        self.vol.ctrlpts = self.getUnweightedCpts(geomData['ctrlpts'], 
                                             geomData['weights'])
        self.vol.weights = geomData['weights']
        self.vol.knotvector_u = geomData['knotvector_u']
        self.vol.knotvector_v = geomData['knotvector_v']
        self.vol.knotvector_w = geomData['knotvector_w']
        
    def getUnweightedCpts(self, ctrlpts, weights):
        numCtrlPts = np.shape(ctrlpts)[0]
        PctrlPts = np.zeros_like(ctrlpts)
        for i in range(3):
            for j in range(numCtrlPts):
                PctrlPts[j,i]=ctrlpts[j][i]/weights[j]
        PctrlPts = PctrlPts.tolist()
        return PctrlPts        
        
    def mapPoints(self, uPar, vPar, wPar):
        '''
        Map points from the parameter domain [0,1]x[0,1]x[0,1] to the hexahedral domain
        Input:  uPar - array containing the u-coordinates in the parameter space
                vPar - array containing the v-coordinates in the parameter space
                wPar - array containing the w-coordinates in the parameter space
                Note: the arrays uPar, vPar and wPar must be of the same size
        Output: xPhys - array containing the x-coordinates in the physical space
                yPhys - array containing the y-coordinates in the physical space
                zPhys - array containing the z-coordinates in the physical space
        '''        
        gpParamUVW = np.array([uPar, vPar, wPar])
        evalList = tuple(map(tuple, gpParamUVW.transpose()))
        res = np.array(self.vol.evaluate_list(evalList))
                
        return res
    
    def bezierExtraction(self, knot, deg):
        '''
        Bezier extraction
        Based on Algorithm 1, from Borden - Isogeometric finite element data
        structures based on Bezier extraction
        '''
        m = len(knot)-deg-1
        a = deg + 1
        b = a + 1
        #initialize C with the number of non-zero knotspans in the 3rd dimension
        nb_final = len(np.unique(knot))-1
        C = np.zeros((deg+1,deg+1,nb_final))
        nb = 1
        C[:,:,0] = np.eye(deg + 1)
        while b <= m:        
            C[:,:,nb] = np.eye(deg + 1)
            i = b        
            while (b <= m) and (knot[b] == knot[b-1]):
                b = b+1            
            multiplicity = b-i+1    
            alphas = np.zeros(deg-multiplicity)        
            if (multiplicity < deg):    
                numerator = knot[b-1] - knot[a-1]            
                for j in range(deg,multiplicity,-1):
                    alphas[j-multiplicity-1] = numerator/(knot[a+j-1]-knot[a-1])            
                r = deg - multiplicity
                for j in range(1,r+1):
                    save = r-j+1
                    s = multiplicity + j                          
                    for k in range(deg+1,s,-1):                                
                        alpha = alphas[k-s-1]
                        C[:,k-1,nb-1] = alpha*C[:,k-1,nb-1] + (1-alpha)*C[:,k-2,nb-1]  
                    if b <= m:                
                        C[save-1:save+j,save-1,nb] = C[deg-j:deg+1,deg,nb-1]  
                nb=nb+1
                if b <= m:
                    a=b
                    b=b+1    
            elif multiplicity==deg:
                if b <= m:
                    nb = nb + 1
                    a = b
                    b = b + 1                
        assert(nb==nb_final)
        
        return C, nb

    def computeC(self):
        
        knotU = self.vol.knotvector_u
        knotV = self.vol.knotvector_v
        knotW = self.vol.knotvector_w
        degU = self.vol.degree_u
        degV = self.vol.degree_v
        degW = self.vol.degree_w
        C_u, nb = self.bezierExtraction(knotU, degU)
        C_v, nb = self.bezierExtraction(knotV, degV)
        C_w, nb = self.bezierExtraction(knotW, degW)
        
        numElemU = len(np.unique(knotU)) - 1
        numElemV = len(np.unique(knotV)) - 1
        numElemW = len(np.unique(knotW)) - 1
        
        basisU = len(knotU) - degU - 1
        basisV = len(knotV) - degV - 1
        nument = (degU+1)*(degV+1)*(degW+1)
        elemInfo = dict()
        elemInfo['vertex'] = []
        elemInfo['nodes'] = []
        elemInfo['C'] = []

        for k in range (0, len(knotW)-1):
            for j in range (0, len(knotV)-1):
                for i in range (0, len(knotU)-1):
                    if ((knotU[i+1] > knotU[i]) and (knotV[j+1] > knotV[j]) and (knotW[k+1] > knotW[k])):
                        vertices = np.array([knotU[i], knotV[j], knotW[k], knotU[i+1], knotV[j+1], knotW[k+1]])
                        elemInfo['vertex'].append(vertices)
                        currow = np.zeros(nument)
                        tcount = 0
                        for t3 in range(k-degW+1,k+2):
                            for t2 in range(j+1-degV,j+2):
                                for t1 in range(i+1-degU,i+2):
                                    currow[tcount] = t1 + (t2-1)*basisU + (t3-1)*basisU*basisV
                                    tcount = tcount + 1
                        elemInfo['nodes'].append(currow)

        for k in range (0, numElemW):
            for j in range (0, numElemV):
                for i in range (0, numElemU):
                    cElem = np.kron(np.kron(C_w[:,:,k],C_v[:,:,j]),C_u[:,:,j])
                    elemInfo['C'].append(cElem)
                    
        return elemInfo
    
    def bernsteinBasis(self, xi, deg):
        '''
        Algorithm A1.3 in Piegl & Tiller
        xi is a 1D array        '''
        
        B = np.zeros((len(xi),deg+1))
        B[:,0] = 1.0
        u1 = 1-xi
        u2 = 1+xi    
        
        for j in range(1,deg+1):
            saved = 0.0
            for k in range(0,j):
                temp = B[:,k].copy()
                B[:,k] = saved + u1*temp        
                saved = u2*temp
            B[:,j] = saved
        B = B/np.power(2,deg)
        
        dB = np.zeros((len(xi),deg))
        dB[:,0] = 1.0
        for j in range(1,deg):
            saved = 0.0
            for k in range(0,j):
                temp = dB[:,k].copy()
                dB[:,k] = saved + u1*temp
                saved = u2*temp
            dB[:,j] = saved
        dB = dB/np.power(2,deg)
        dB0 = np.transpose(np.array([np.zeros(len(xi))]))
        dB = np.concatenate((dB0, dB, dB0), axis=1)
        dB = (dB[:,0:-1] - dB[:,1:])*deg
    
        return B, dB

    def genElemList(self, numElemU, numElemV, numElemW):
        '''
        Generate quadrature points inside the domain for an initial (uniform)
        subdivision mesh
        Input: numElemU, numElemV, numElemW - number of subdivisions in the u, v, and w
                   directions in the parameter space               
        Output: vertex - arrays containing the element vertices
                        Format: [uMin, vMin, wMin, uMax, vMax, wMax]
        '''
        vertex = np.zeros((numElemU*numElemV*numElemW, 6))
                        
        #generate the knots on the interval [0,1]
        uEdge = np.linspace(0, 1, numElemU+1)
        vEdge = np.linspace(0, 1, numElemV+1)
        wEdge = np.linspace(0, 1, numElemW+1)

        #create meshgrid
        uPar, vPar, wPar = np.meshgrid(uEdge, vEdge, wEdge, indexing='ij')              
        counterElem = 0                
        #generate points for each element

        for iW in range(numElemW):
            for iV in range(numElemV):
                for iU in range(numElemU):
                    uMin = uPar[iU, iV, iW]
                    uMax = uPar[iU+1, iV, iW]
                    vMin = vPar[iU, iV, iW]
                    vMax = vPar[iU, iV+1, iW]
                    wMin = wPar[iU, iV, iW]
                    wMax = wPar[iU, iV, iW+1]
                    vertex[counterElem, 0] = uMin
                    vertex[counterElem, 1] = vMin
                    vertex[counterElem, 2] = wMin
                    vertex[counterElem, 3] = uMax
                    vertex[counterElem, 4] = vMax
                    vertex[counterElem, 5] = wMax
                    counterElem = counterElem + 1                                        
        return vertex
    
    def findspan(self, uCoord, vCoord, wCoord):
        '''
        Generates the element number on which the co-ordinate is located'''
        knotU = self.vol.knotvector_u
        knotV = self.vol.knotvector_v
        knotW = self.vol.knotvector_w        
        
        counter = 0
        for k in range (0, len(knotW)-1):
            for j in range (0, len(knotV)-1):
                for i in range (0, len(knotU)-1):
                    if ((knotU[i+1] > knotU[i]) and (knotV[j+1] > knotV[j]) and (knotW[k+1] > knotW[k])):
                        if ((uCoord >= knotU[i]) and (uCoord <= knotU[i+1]) and (vCoord >= knotV[j]) and (vCoord <= knotV[j+1]) and (wCoord >= knotW[k]) and (wCoord <= knotW[k+1])):
                            elmtNum = counter
                            break
                        counter = counter + 1
        
        return elmtNum
    
    def getDerivatives(self, uCoord, vCoord, wCoord, elmtNo):
        '''
        Generate physical points and jacobians for parameter points inside the domain
        Input: uCoord, vCoord, wCoord: Inputs the co-odinates of the Gauss points in the parameter space.
                elmtNo: element index
        Output: xPhys, yPhys, jacMat - Generates the co-ordinates in the physical space and the jacobian matrix
        '''
        curVertex = self.vertex[elmtNo]
        cElem = self.C[elmtNo]
        curNodes = np.int32(self.nodes[elmtNo])-1 # Python indexing starts from 0        
        ctrlpts = np.array(self.vol.ctrlpts)
        weights = np.array(self.vol.weights)
        curPts = np.squeeze(ctrlpts[curNodes,0:3])
        wgts = np.transpose(weights[curNodes][np.newaxis])
        #assert 0
        # Get the Gauss points on the reference interval [-1,1]
        uMax = curVertex[3]
        uMin = curVertex[0]
        vMax = curVertex[4]
        vMin = curVertex[1]
        wMax = curVertex[5]
        wMin = curVertex[2]
                
        uHatCoord = (2*uCoord - (uMax+uMin))/(uMax-uMin)
        vHatCoord = (2*vCoord - (vMax+vMin))/(vMax-vMin)
        wHatCoord = (2*wCoord - (wMax+wMin))/(wMax-wMin)
        
        degU = self.vol.degree_u
        degV = self.vol.degree_v
        degW = self.vol.degree_w
        
        B_u, dB_u = self.bernsteinBasis(uHatCoord,degU)
        B_v, dB_v = self.bernsteinBasis(vHatCoord,degV)
        B_w, dB_w = self.bernsteinBasis(wHatCoord,degW)
        numGauss = len(uCoord)

        # Computing the Bernstein polynomials in 3D
        dBdu = np.zeros((numGauss, numGauss, numGauss, (degU+1)*(degV+1)*(degW+1)))
        dBdv = np.zeros((numGauss, numGauss, numGauss, (degU+1)*(degV+1)*(degW+1)))
        dBdw = np.zeros((numGauss, numGauss, numGauss, (degU+1)*(degV+1)*(degW+1)))
        R = np.zeros((numGauss, numGauss, numGauss, (degU+1)*(degV+1)*(degW+1)))

        counter = 0
        for k in range(0,degW+1):
            for j in range(0,degV+1):
                for i in range(0,degU+1):
                    for kk in range(numGauss):
                        for jj in range(numGauss):
                            for ii in range(numGauss):
                                R[ii,jj,kk,counter] = B_u[ii,i]* B_v[jj,j]*B_w[kk,k]
                                dBdu[ii,jj,kk,counter] = dB_u[ii,i]*B_v[jj,j]*B_w[kk,k]
                                dBdv[ii,jj,kk,counter] = B_u[ii,i]*dB_v[jj,j]*B_w[kk,k]
                                dBdw[ii,jj,kk,counter] = B_u[ii,i]*B_v[jj,j]*dB_w[kk,k]
                    counter = counter + 1              
        
        # Map the points to the physical space
        for kPt in range(0,numGauss):
            for jPt in range(0,numGauss):
                for iPt in range(0,numGauss):
                    dRdx = np.matmul(cElem,np.transpose(np.array([dBdu[iPt,jPt,kPt,:]])))*2/(uMax-uMin)
                    
                    dRdy = np.matmul(cElem,np.transpose(np.array([dBdv[iPt,jPt,kPt,:]])))*2/(vMax-vMin)
                    dRdz = np.matmul(cElem,np.transpose(np.array([dBdw[iPt,jPt,kPt,:]])))*2/(wMax-wMin)
                    RR = np.matmul(cElem,np.transpose(np.array([R[iPt,jPt,kPt,:]])))
                    RR = RR*wgts
                    dRdx = dRdx*wgts
                    dRdy = dRdy*wgts
                    dRdz = dRdz*wgts
                    
                    w_sum = np.sum(RR, axis=0)
                    dw_xi = np.sum(dRdx, axis=0)
                    dw_eta = np.sum(dRdy, axis=0)
                    dw_zeta = np.sum(dRdz, axis=0)
                    
                    dRdx = dRdx/w_sum  - RR*dw_xi/np.power(w_sum,2)
                    dRdy = dRdy/w_sum - RR*dw_eta/np.power(w_sum,2)
                    dRdz = dRdz/w_sum - RR*dw_zeta/np.power(w_sum,2)
                    RR = RR/w_sum                    
                    dR  = np.concatenate((dRdx.T,dRdy.T,dRdz.T),axis=0)
                    jacMat = np.matmul(dR,curPts)
                    coord = np.matmul(np.transpose(RR),curPts)
                    
                    xPhys = coord[0,0]
                    yPhys = coord[0,1]
                    zPhys = coord[0,2]
                 
        return xPhys, yPhys, zPhys, jacMat
    
    def getElemIntPts(self, vertex, numGauss):
        '''
        Generate quadrature points inside the domain
        Input: vertex - contains the vertices of the elements the refined elements
               numGauss - number of Gauss quadrature points for each subdivision
        Output: xPhys, yPhys, zPhys, wgtPhy - arrays containing the x, y and z 
                        coordinates of the points and the corresponding weights
        '''
        #allocate quad pts arrays (xPhys, yPhys, zPhys, wgtPhys)
        numElems = vertex.shape[0]
        xPhys = np.zeros((numElems*numGauss**3, 1))
        yPhys = np.zeros((numElems*numGauss**3, 1))
        zPhys = np.zeros((numElems*numGauss**3, 1))
        wgtPhys = np.zeros((numElems*numGauss**3, 1))
        
        # Get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)
        
        #get the Gauss weights on the reference element [-1, 1]x[-1,1]x[-1,1]
        gpWeightU, gpWeightV, gpWeightW = np.meshgrid(gw, gw, gw, indexing='ij')
        gpWeightUVW = np.array(gpWeightU.flatten()*gpWeightV.flatten()*gpWeightW.flatten())
               
        elemInfo = self.computeC()
        self.C = elemInfo['C']
        self.nodes = elemInfo['nodes']
        self.vertex = elemInfo['vertex']
                
        #generate points for each element
        indexPt = 0
        for iElem in range(numElems):            
            uMin = vertex[iElem, 0]
            uMax = vertex[iElem, 3]
            vMin = vertex[iElem, 1]
            vMax = vertex[iElem, 4]
            wMin = vertex[iElem, 2]
            wMax = vertex[iElem, 5]
            gpParamU = (uMax-uMin)/2*gp+(uMax+uMin)/2
            gpParamV = (vMax-vMin)/2*gp+(vMax+vMin)/2
            gpParamW = (wMax-wMin)/2*gp+(wMax+wMin)/2
            gpParamUg, gpParamVg, gpParamWg = np.meshgrid(gpParamU, gpParamV, gpParamW, indexing='ij')
            gpParamUVW = np.array([gpParamUg.flatten(), gpParamVg.flatten(), gpParamWg.flatten()])
            #Jacobian of the transformation from the reference element [-1,1]x[-1,1]x[-1,1]
            scaleFac = (uMax-uMin)*(vMax-vMin)*(wMax-wMin)/8
                
            #map the points to the physical space
            for iPt in range(numGauss**3):
                curPtU = np.array([gpParamUVW[0, iPt]])
                curPtV = np.array([gpParamUVW[1, iPt]])
                curPtW = np.array([gpParamUVW[2, iPt]])
                
                elmtNo = self.findspan(curPtU, curPtV, curPtW)                        
                physPtX, physPtY, physPtZ, jacMat = self.getDerivatives(curPtU, curPtV, curPtW, elmtNo)
                ptJac = np.absolute(np.linalg.det(jacMat))
                xPhys[indexPt] = physPtX
                yPhys[indexPt] = physPtY
                zPhys[indexPt] = physPtZ
                wgtPhys[indexPt] = scaleFac * ptJac * gpWeightUVW[iPt]
                indexPt = indexPt + 1            
        
        return xPhys, yPhys, zPhys, wgtPhys
    
    def getQuadFacePts(self, numElem, numGauss, orient):
        '''
        Generate points on the boundary face given by orient
        Input: numElem - 1x2 array with the number of number of subdivisions
               numGauss - number of Gauss points per subdivision (in each direction)
               orient - edge orientation in parameter space: 1 is front (v=0), 
                        2 is right (u=1), 3 is back (v=1), 4 is left (u=0), 
                        5 is down (w=0), 6 is up (w=1)
        Output: xBnd, yBnd, zBnd - coordinates of the boundary in the physical space                                     
                xNorm, yNorm, zNorm  - x,y,z components of the outer normal vector
                wgtBnd - Gauss weights of the boundary points
        '''
        #allocate quad pts arrays (xBnd, yBnd, zBnd, wgtBnd, xNorm, yNorm, zNorm)
        xBnd = np.zeros((numElem[0]*numElem[1]*numGauss**2, 1))
        yBnd = np.zeros((numElem[0]*numElem[1]*numGauss**2, 1))
        zBnd = np.zeros((numElem[0]*numElem[1]*numGauss**2, 1))
        xNorm = np.zeros((numElem[0]*numElem[1]*numGauss**2, 1))
        yNorm = np.zeros((numElem[0]*numElem[1]*numGauss**2, 1))
        zNorm = np.zeros((numElem[0]*numElem[1]*numGauss**2, 1))
        wgtBnd = np.zeros((numElem[0]*numElem[1]*numGauss**2, 1))
        
        #get the Gauss points on the reference interval [-1,1]
        gp, gw = np.polynomial.legendre.leggauss(numGauss)  
        
        #get the Gauss weights on the reference element [-1, 1]x[-1,1]
        gpWeight0, gpWeight1 = np.meshgrid(gw, gw)
        gpWeight01 = np.array(gpWeight0.flatten()*gpWeight1.flatten())
                
        #generate the knots on the interval [0,1]
        edge0Par = np.linspace(0, 1, numElem[0]+1)
        edge1Par = np.linspace(0, 1, numElem[1]+1)
                        
        #generate points for each element
        indexPt = 0
        for i1E in range(numElem[1]):                
            for i0E in range(numElem[0]):
                edge0Min = edge0Par[i0E]
                edge0Max = edge0Par[i0E+1]
                edge1Min = edge1Par[i1E]
                edge1Max = edge1Par[i1E+1]
                if orient==1:
                    gpParamU = (edge0Max-edge0Min)/2*gp+(edge0Max+edge0Min)/2                    
                    gpParamW = (edge1Max-edge1Min)/2*gp+(edge1Max+edge1Min)/2
                    gpParamUg, gpParamWg = np.meshgrid(gpParamU, gpParamW)
                    gpParamVg = np.zeros_like(gpParamUg)
                elif orient==2:
                    gpParamV = (edge0Max-edge0Min)/2*gp+(edge0Max+edge0Min)/2
                    gpParamW = (edge1Max-edge1Min)/2*gp+(edge1Max+edge1Min)/2
                    gpParamVg, gpParamWg = np.meshgrid(gpParamV, gpParamW)
                    gpParamUg = np.ones_like(gpParamVg)
                elif orient==3:
                    gpParamU = (edge0Max-edge0Min)/2*gp+(edge0Max+edge0Min)/2
                    gpParamW = (edge1Max-edge1Min)/2*gp+(edge1Max+edge1Min)/2
                    gpParamUg, gpParamWg = np.meshgrid(gpParamU, gpParamW)
                    gpParamVg = np.ones_like(gpParamUg)
                elif orient==4:                    
                    gpParamV = (edge0Max-edge0Min)/2*gp+(edge0Max+edge0Min)/2
                    gpParamW = (edge1Max-edge1Min)/2*gp+(edge1Max+edge1Min)/2
                    gpParamVg, gpParamWg = np.meshgrid(gpParamV, gpParamW)
                    gpParamUg = np.zeros_like(gpParamVg)
                elif orient==5:
                    gpParamU = (edge0Max-edge0Min)/2*gp+(edge0Max+edge0Min)/2
                    gpParamV = (edge1Max-edge1Min)/2*gp+(edge1Max+edge1Min)/2
                    gpParamUg, gpParamVg = np.meshgrid(gpParamU, gpParamV)
                    gpParamWg = np.zeros_like(gpParamUg)
                elif orient==6:
                    gpParamU = (edge0Max-edge0Min)/2*gp+(edge0Max+edge0Min)/2
                    gpParamV = (edge1Max-edge1Min)/2*gp+(edge1Max+edge1Min)/2
                    gpParamUg, gpParamVg = np.meshgrid(gpParamU, gpParamV)
                    gpParamWg = np.ones_like(gpParamUg)
                else:
                    raise Exception('Wrong orientation given')
                        
                gpParamUVW = np.array([gpParamUg.flatten(), gpParamVg.flatten(), gpParamWg.flatten()])
                
                #Jacobian of the transformation from the reference element [-1,1]            
                scaleFac = (edge0Max-edge0Min)*(edge1Max-edge1Min)/4

                #map the points to the physical space                
                for iPt in range(numGauss**2):
                    curPtU = np.array([gpParamUVW[0, iPt]])
                    curPtV = np.array([gpParamUVW[1, iPt]])
                    curPtW = np.array([gpParamUVW[2, iPt]])
                    
                    elmtNo = self.findspan(curPtU, curPtV, curPtW)                        
                    physPtX, physPtY, physPtZ, jacMat = self.getDerivatives(curPtU, curPtV, curPtW, elmtNo)
                
                    if orient==1:                                                
                        normX = jacMat[0,1]*jacMat[2,2] - jacMat[0,2]*jacMat[2,1]
                        normY = jacMat[0,2]*jacMat[2,0] - jacMat[0,0]*jacMat[2,2]
                        normZ = jacMat[0,0]*jacMat[2,1] - jacMat[0,1]*jacMat[2,0] 
                    elif orient==2:
                        normX = jacMat[1,1]*jacMat[2,2] - jacMat[1,2]*jacMat[2,1]
                        normY = jacMat[1,2]*jacMat[2,0] - jacMat[1,0]*jacMat[2,2]
                        normZ = jacMat[1,0]*jacMat[2,1] - jacMat[1,1]*jacMat[2,0]
                    elif orient==3:
                        normX = -jacMat[0,1]*jacMat[2,2] + jacMat[0,2]*jacMat[2,1]
                        normY = -jacMat[0,2]*jacMat[2,0] + jacMat[0,0]*jacMat[2,2]
                        normZ = -jacMat[0,0]*jacMat[2,1] + jacMat[0,1]*jacMat[2,0]
                    elif orient==4:
                        normX = -jacMat[1,1]*jacMat[2,2] + jacMat[1,2]*jacMat[2,1]
                        normY = -jacMat[1,2]*jacMat[2,0] + jacMat[1,0]*jacMat[2,2]
                        normZ = -jacMat[1,0]*jacMat[2,1] + jacMat[1,1]*jacMat[2,0]
                    elif orient==5:
                        normX = jacMat[1,1]*jacMat[0,2] - jacMat[1,2]*jacMat[0,1]
                        normY = jacMat[1,2]*jacMat[0,0] - jacMat[1,0]*jacMat[0,2]
                        normZ = jacMat[1,0]*jacMat[0,1] - jacMat[1,1]*jacMat[0,0]
                    elif orient==6:
                        normX = -jacMat[1,1]*jacMat[0,2] + jacMat[1,2]*jacMat[0,1]
                        normY = -jacMat[1,2]*jacMat[0,0] + jacMat[1,0]*jacMat[0,2]
                        normZ = -jacMat[1,0]*jacMat[0,1] + jacMat[1,1]*jacMat[0,0]
                    else:
                        raise Exception('Wrong orientation given')
                        
                    JacobFace = np.sqrt(normX**2+normY**2+normZ**2)
                    normX = normX/JacobFace
                    normY = normY/JacobFace
                    normZ = normZ/JacobFace
        
                    xBnd[indexPt] = physPtX
                    yBnd[indexPt] = physPtY
                    zBnd[indexPt] = physPtZ
                    xNorm[indexPt] = normX
                    yNorm[indexPt] = normY
                    zNorm[indexPt] = normZ                    
                    wgtBnd[indexPt] = scaleFac * JacobFace * gpWeight01[iPt]
                    indexPt = indexPt + 1                                    
        
        return xBnd, yBnd, zBnd, xNorm, yNorm, zNorm, wgtBnd
    
    def getUnifIntPts(self, numPtsU, numPtsV, numPtsW, withSides):
        '''
        Generate uniformly spaced points inside the domain
        Input: numPtsU, numPtsV, numPtsW - number of points (including edges) in the u, v, w
                   directions in the parameter space
               withSides - 1x6 array of zeros or ones specifying whether the boundary points
                           should be included. The boundary order is [front, right,
                           back, left, bottom, top] for the unit square.
        Output: xM, yM, zM - flattened array containing the x and y coordinates of the points
        '''
        #generate points in the x direction on the interval [0,1]
        uEdge = np.linspace(0, 1, numPtsU)
        vEdge = np.linspace(0, 1, numPtsV)
        wEdge = np.linspace(0, 1, numPtsW)
        
        #remove endpoints depending on values of withEdges
        if withSides[0]==0:
            vEdge = vEdge[1:]
        if withSides[1]==0:
            uEdge = uEdge[:-1]
        if withSides[2]==0:
            vEdge = vEdge[:-1]
        if withSides[3]==0:
            uEdge = uEdge[1:]
        if withSides[4]==0:
            wEdge = wEdge[1:]
        if withSides[5]==0:
            wEdge = wEdge[:-1]
            
        #create meshgrid
        uPar, vPar, wPar = np.meshgrid(uEdge, vEdge, wEdge, indexing='ij')
                        
        uPar = uPar.flatten()
        vPar = vPar.flatten()
        wPar = wPar.flatten()
        
        #map points
        res = self.mapPoints(uPar.T, vPar.T, wPar.T)
        
        xPhys = res[:, 0:1]
        yPhys = res[:, 1:2]
        zPhys = res[:, 2:3]
        
        return xPhys, yPhys, zPhys    
