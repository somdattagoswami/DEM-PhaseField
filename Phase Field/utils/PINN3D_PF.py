# This is the class for Physics informed neural network for phase field modeling in 3D

import tensorflow as tf
import numpy as np
import time

class CalculateUPhi:
    # Initialize the class
    def __init__(self, model, NN_param):
        
        # Elasticity parameters
        self.E = model['E']
        self.nu = model['nu']
        
        self.c11 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c22 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c33 = self.E *(1-self.nu)/((1+self.nu)*(1-2*self.nu))
        self.c12 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c13 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c21 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c23 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c31 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c32 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c44 = self.E/(2*(1+self.nu))
        self.c55 = self.E/(2*(1+self.nu))
        self.c66 = self.E/(2*(1+self.nu))  
        
        self.lamda = self.E*self.nu/((1-2*self.nu)*(1+self.nu))
        self.mu = 0.5*self.E/(1+self.nu)
        
        # Phase field parameters
        self.cEnerg = 0.5 # Critical energy release rate of the material
        self.B = 100
        self.l = model['l']
        self.crackTip = 0.5
        
        self.lb = model['lb']
        self.ub = model['ub']    
        
        self.layers = NN_param['layers']
        self.data_type = NN_param['data_type']
        self.weights, self.biases = self.initialize_NN(self.layers)
        
        # tf Placeholders        
        self.x_f_tf = tf.placeholder(self.data_type)
        self.y_f_tf = tf.placeholder(self.data_type)
        self.z_f_tf = tf.placeholder(self.data_type)
        self.wt_f_tf = tf.placeholder(self.data_type)
        self.hist_tf = tf.placeholder(self.data_type)
        self.wdelta_tf = tf.placeholder(self.data_type)
        
        # tf Graphs        
        self.energy_u_pred, self.energy_phi_pred, self.hist_pred = \
            self.net_energy(self.x_f_tf,self.y_f_tf, self.z_f_tf, self.hist_tf,self.wdelta_tf)
        self.u_pred, self.v_pred, self.w_pred = self.net_uvw(self.x_f_tf,self.y_f_tf, self.z_f_tf, self.wdelta_tf)
        self.phi_pred = self.net_phi(self.x_f_tf, self.y_f_tf, self.z_f_tf)
        self.f_u_pred, self.f_v_pred, self.f_w_pred = self.net_f_uvw(self.x_f_tf,self.y_f_tf, self.z_f_tf, self.wdelta_tf)
        
        # Loss
        self.loss_energy_u = tf.reduce_sum(self.energy_u_pred*self.wt_f_tf) 
        self.loss_energy_phi = tf.reduce_sum(self.energy_phi_pred*self.wt_f_tf) 
        
        self.loss = self.loss_energy_u + self.loss_energy_phi 
              
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = [self.weights, self.biases])
        
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self,layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)
    
    def neural_net(self,X,weights,biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uvw(self,x,y,z,wdelta):

        X = tf.concat([x,y,z],1)

        uvwphi = self.neural_net(X,self.weights,self.biases)
        uNN = uvwphi[:,0:1]
        vNN = uvwphi[:,1:2]
        wNN = uvwphi[:,2:3]
        
        u = z*uNN
        v = z*vNN 
        w = z*(z-1)*wNN + z*wdelta
        
        return u, v, w
    
    def net_phi(self,x,y,z):

        X = tf.concat([x,y,z],1)
        
        uvphi = self.neural_net(X,self.weights,self.biases)
        phi = uvphi[:,3:4]        

        return phi
    
    def net_hist(self,x,y,z):
        
        shapeX = tf.shape(x)
        init_hist = tf.zeros((shapeX[0],shapeX[1]), dtype = np.float32)
        dist = tf.where(x > self.crackTip, tf.sqrt((x-0.5)**2 + (z-0.5)**2), tf.abs(z-0.5))
        init_hist = tf.where(dist < 0.5*self.l, self.B*self.cEnerg*0.5*(1-(2*dist/self.l))/self.l, init_hist)
        
        return init_hist
    
    def net_update_hist(self,x, y, z, u_x, v_y, w_z, u_xy, u_yz, u_zx, hist):
        
        init_hist = self.net_hist(x,y,z)
        
        # Computing the tensile strain energy     
        eigSum = (u_x + v_y + w_z)
        sEnergy_pos = 0.125*self.lamda * (eigSum + tf.abs(eigSum))**2 + \
                      0.25*self.mu*((u_x + tf.abs(u_x))**2 + (v_y + tf.abs(v_y))**2 + \
                     (w_z + tf.abs(w_z))**2)        
       
        hist_temp = tf.maximum(init_hist, sEnergy_pos)
        hist= tf.maximum(hist, hist_temp)
        
        return hist
    
    def net_energy(self, x, y, z, hist, wdelta):

        u, v, w = self.net_uvw(x, y, z, wdelta)
        phi = self.net_phi(x, y, z)
        
        g = (1-phi)**2
        
        phi_x = tf.gradients(phi, x)[0]
        phi_y = tf.gradients(phi, y)[0]
        phi_z = tf.gradients(phi, z)[0] 
        nabla = phi_x**2 + phi_y**2 + phi_z**2
        
        phi_xx = tf.gradients(phi_x, x)[0]
        phi_yy = tf.gradients(phi_y, y)[0]
        phi_zz = tf.gradients(phi_z, z)[0]
        
        laplacian = (phi_xx + phi_yy + phi_zz)**2       
        u_x = tf.gradients(u,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_z = tf.gradients(u,z)[0]
        v_x = tf.gradients(v,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_z = tf.gradients(v,z)[0]
        w_x = tf.gradients(w,x)[0]
        w_y = tf.gradients(w,y)[0]
        w_z = tf.gradients(w,z)[0]
        u_xy = (u_y + v_x)
        u_yz = (v_z + w_y)
        u_zx = (u_z + w_x)
        
        hist = self.net_update_hist(x, y, z, u_x, v_y, w_z, u_xy, u_yz, u_zx, hist) 
        
        sigmaX = self.c11*u_x + self.c12*v_y + self.c13*w_z
        sigmaY = self.c21*u_x + self.c22*v_y + self.c23*w_z
        sigmaZ = self.c31*u_x + self.c32*v_y + self.c33*w_z
        tauYZ = self.c44*u_yz
        tauZX = self.c55*u_zx
        tauXY = self.c66*u_xy
        
        energy_u = 0.5*g*(sigmaX*u_x + sigmaY*v_y + sigmaZ*w_z + tauYZ*u_yz + \
                          tauZX*u_zx + tauXY*u_xy)
        energy_phi = 0.5*self.cEnerg * (phi**2/self.l + self.l*nabla + \
                                        0.5*self.l**3*laplacian) + g* hist
        
        return energy_u, energy_phi, hist
    
    def net_f_uvw(self,x,y,z,wdelta):

        u, v, w = self.net_uvw(x, y, z, wdelta)
        phi = self.net_phi(x, y, z)
        g = (1-phi)**2
        
        u_x = tf.gradients(u,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_z = tf.gradients(u,z)[0]
        v_x = tf.gradients(v,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_z = tf.gradients(v,z)[0]
        w_x = tf.gradients(w,x)[0]
        w_y = tf.gradients(w,y)[0]
        w_z = tf.gradients(w,z)[0]
        gamma_xy = u_y + v_x
        gamma_yz = v_z + w_y
        gamma_zx = u_z + w_x
        
        sigmaX = self.c11*u_x + self.c12*v_y + self.c13*w_z
        sigmaY = self.c21*u_x + self.c22*v_y + self.c23*w_z
        sigmaZ = self.c31*u_x + self.c32*v_y + self.c33*w_z
        tauXY = self.c44*gamma_xy
        tauYZ = self.c55*gamma_yz
        tauZX = self.c66*gamma_zx
        
        f_u = -g*(tf.gradients(sigmaX, x)[0] + tf.gradients(tauXY, y)[0] + \
                tf.gradients(tauZX, z)[0])
        f_v = -g*(tf.gradients(tauXY, x)[0] + tf.gradients(sigmaY, y)[0] + \
                tf.gradients(tauYZ, z)[0])
        f_w = -g*(tf.gradients(tauZX, x)[0] + tf.gradients(tauYZ, y)[0] + \
                tf.gradients(sigmaZ, z)[0])
                
        return f_u, f_v, f_w
    
    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        
    def train(self, X_f, w_delta, hist_f, nIter,  nIterLBFGS):

        tf_dict = {self.x_f_tf: X_f[:,0:1], self.y_f_tf: X_f[:,1:2], 
                   self.z_f_tf: X_f[:,2:3], self.wt_f_tf: X_f[:,3:4],
                   self.hist_tf: hist_f, self.wdelta_tf: w_delta}

        start_time = time.time()
        self.loss_adam_buff = np.zeros(nIter)
        
        for it in range(nIter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_adam_buff[it] = loss_value
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                energy_u_val = self.sess.run(self.loss_energy_u, tf_dict)
                energy_phi_val = self.sess.run(self.loss_energy_phi, tf_dict)

                print('It: %d, Total Loss: %.3e, Energy U: %.3e, Energy Phi: %.3e, Time: %.2f' %
                      (it, loss_value, energy_u_val, energy_phi_val, elapsed))
                start_time = time.time()
                
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': nIterLBFGS,
                                                                         'maxfun': nIterLBFGS,
                                                                         'maxcor': 100,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
                
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)         
        
    def predict(self, X_star, Hist_star, w_delta):

        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2], 
                   self.z_f_tf: X_star[:,2:3], self.hist_tf: Hist_star[:,0:1],
                   self.wdelta_tf: w_delta}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        energy_u_star = self.sess.run(self.energy_u_pred, tf_dict)
        energy_phi_star = self.sess.run(self.energy_phi_pred, tf_dict)
        hist_star = self.sess.run(self.hist_pred, tf_dict) 
        
        return u_star, v_star, w_star, phi_star, energy_u_star, energy_phi_star, hist_star
    
    def predict_phi(self, X_star):
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2], 
                   self.z_f_tf: X_star[:,2:3]}                       
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        return phi_star
    
    def predict_f(self, X_star, w_delta):
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2], 
                   self.z_f_tf: X_star[:,2:3], self.wdelta_tf: w_delta}
                      
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        f_w_star = self.sess.run(self.f_w_pred, tf_dict)
        
        return f_u_star, f_v_star, f_w_star
    
    def getWeightsBiases(self):
        weights =  self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        return weights, biases