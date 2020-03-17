# This is the class for Physics informed neural network for phase field modeling in 2D

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
        self.c12 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c21 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.c31 = 0.0
        self.c32 = 0.0
        self.c13 = 0.0
        self.c23 = 0.0
        self.c33 = self.E/(2*(1+self.nu))
        
        self.lamda = self.E*self.nu/((1-2*self.nu)*(1+self.nu))
        self.mu = 0.5*self.E/(1+self.nu)
        
        # Phase field parameters
        self.cEnerg = 2.7 # Critical energy release rate of the material
        self.B = 92
        self.l = model['l']
        
        self.lb = model['lb']
        self.ub = model['ub']    
        
        self.layers = NN_param['layers']
        self.data_type = NN_param['data_type']
        self.weights, self.biases = self.initialize_NN(self.layers)
        
        # tf Placeholders        
        self.x_f_tf = tf.placeholder(self.data_type)
        self.y_f_tf = tf.placeholder(self.data_type)
        self.wt_f_tf = tf.placeholder(self.data_type)
        self.hist_tf = tf.placeholder(self.data_type)
        self.vdelta_tf = tf.placeholder(self.data_type)
        
        # tf Graphs        
        self.energy_u_pred, self.energy_phi_pred, self.hist_pred = \
            self.net_energy(self.x_f_tf,self.y_f_tf, self.hist_tf,self.vdelta_tf)
        self.u_pred, self.v_pred = self.net_uv(self.x_f_tf,self.y_f_tf,self.vdelta_tf)
        self.phi_pred = self.net_phi(self.x_f_tf,self.y_f_tf)
        self.traction_pred = self.net_traction(self.x_f_tf,self.y_f_tf,self.vdelta_tf)
        self.f_u_pred, self.f_v_pred = self.net_f(self.x_f_tf,self.y_f_tf,self.vdelta_tf)
        
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
    
    def net_phi(self,x,y):

        X = tf.concat([x,y],1)
        
        uvphi = self.neural_net(X,self.weights,self.biases)
        phi = uvphi[:,2:3]        

        return phi
    
    def net_hist(self,x,y):
        
        shape = tf.shape(x)
        init_hist = tf.zeros((shape[0],shape[1]), dtype = np.float32)
        dist = tf.where(x > self.crackTip, tf.sqrt((x-0.5)**2 + (y-0.5)**2), tf.abs(y-0.5))
        init_hist = tf.where(dist < 0.5*self.l, self.B*self.cEnerg*0.5*(1-(2*dist/self.l))/self.l, init_hist)
        
        return init_hist
    
    def net_update_hist(self,x,y,u_x,v_y,u_xy,hist):
        
        init_hist = self.net_hist(x,y)
        
        # Computing the tensile strain energy
        u_xy = 0.5*u_xy
        M = tf.sqrt((u_x-v_y)**2 + 4*(u_xy**2))
        lambda1 = 0.5*(u_x + v_y) + 0.5*M
        lambda2 = 0.5*(u_x + v_y) - 0.5*M
        
        eigSum = (lambda1 + lambda2)        
        sEnergy_pos = 0.125*self.lamda * (eigSum + tf.abs(eigSum))**2 + \
        0.25*self.mu*((lambda1 + tf.abs(lambda1))**2 + (lambda2 + tf.abs(lambda2))**2)        
        
        hist_temp = tf.maximum(init_hist, sEnergy_pos)
        hist = tf.maximum(hist, hist_temp)
        
        return hist
    
    def net_f(self,x,y,vdelta):

        u, v = self.net_uv(x,y,vdelta)
        phi = self.net_phi(x,y)
        
        g = (1-phi)**2        

        u_x = tf.gradients(u,x)[0]
        u_xx = tf.gradients(u_x,x)[0]
        v_y = tf.gradients(v,y)[0]
        v_yx = tf.gradients(v_y,x)[0]
        u_y = tf.gradients(u,y)[0]
        u_yy = tf.gradients(u_y,y)[0]
        v_x = tf.gradients(v,x)[0]
        v_xy = tf.gradients(v_x,y)[0]

        f_u = -g*(self.c11*u_xx + self.c12*v_yx + self.c33*u_yy + self.c33*v_xy)

        u_xy = tf.gradients(u_x,y)[0]
        v_yy = tf.gradients(v_y,y)[0]
        u_yx = tf.gradients(u_y,x)[0]
        v_xx = tf.gradients(v_x,x)[0]

        f_v = -g*(self.c21*u_xy + self.c22*v_yy + self.c33*u_yx + self.c33*v_xx)
        
        return f_u, f_v 
    
    def net_traction(self,x,y,vdelta):
        
        u, v = self.net_uv(x,y,vdelta)
        u_x = tf.gradients(u,x)[0]
        v_y = tf.gradients(v,y)[0]
        
        traction = self.c21*u_x + self.c22*v_y
        
        return traction
    
    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
#        print('Loss:', loss)
        
    def train(self, X_f, v_delta, hist_f, nIter, nIterLBFGS):

        tf_dict = {self.x_f_tf: X_f[:,0:1], self.y_f_tf: X_f[:,1:2], self.wt_f_tf: X_f[:,2:3],
                    self.hist_tf: hist_f, self.vdelta_tf: v_delta}

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
        
    def predict(self, X_star, Hist_star, v_delta):

        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2], 
                   self.hist_tf: Hist_star[:,0:1],self.vdelta_tf: v_delta}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        energy_u_star = self.sess.run(self.energy_u_pred, tf_dict)
        energy_phi_star = self.sess.run(self.energy_phi_pred, tf_dict)
        hist_star = self.sess.run(self.hist_pred, tf_dict)
        
        return u_star, v_star, phi_star, energy_u_star, energy_phi_star, hist_star
    
    def predict_traction(self, X_star, v_delta):
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2], 
                   self.vdelta_tf: v_delta}                       
        trac_star = self.sess.run(self.traction_pred, tf_dict)        
        
        return trac_star
    
    def predict_phi(self, X_star):
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2]}                       
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        return phi_star
    
    def predict_f(self, X_star, v_delta):
        
        tf_dict = {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2], self.vdelta_tf: v_delta}                       
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        
        return f_u_star, f_v_star
    
    def getWeightsBiases(self):
        weights =  self.sess.run(self.weights)
        biases = self.sess.run(self.biases)
        return weights, biases