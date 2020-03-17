# This is the class for Physics informed neural network for phase field modeling in 1D

import tensorflow as tf
import numpy as np
import time
import math as m

class CalculateUPhi:
    # Initialize the class
    def __init__(self, l, layers, lb, ub):
        
        self.l = l
        self.lb = lb
        self.ub = ub
               
        self.B = 1000.0
        self.cEnerg = 2.7
        self.E = 1.0
        self.crackLoc = 0.0
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.x_f_tf = tf.placeholder(tf.float64)
        self.wt_f_tf = tf.placeholder(tf.float64)
        # tf Graphs
        self.energy_u_pred, self.energy_phi_pred, self.ext_work_pred = \
            self.net_energy(self.x_f_tf)
        self.u_pred = self.net_u(self.x_f_tf)
        self.phi_pred = self.net_phi(self.x_f_tf)
        self.f_u_pred = self.net_f(self.x_f_tf)
        
        self.loss_energy_u = tf.reduce_sum(self.energy_u_pred*self.wt_f_tf) - tf.reduce_sum(self.ext_work_pred*self.wt_f_tf) 
        self.loss_energy_phi = tf.reduce_sum(self.energy_phi_pred*self.wt_f_tf) 
        
        self.loss = self.loss_energy_u + self.loss_energy_phi                 

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10000,
                                                                         'maxfun': 10000,
                                                                         'maxcor': 100,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        self.lbfgs_buffer = []
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

    def neural_net(self, X, weights, biases):
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

    def net_u(self,x):
        
        X = x[:, 0:1]
        uphi = self.neural_net(X, self.weights, self.biases)
        uNN = uphi[:,0:1]
        
        u = (x+1)*(x-1)*uNN
        return u
    
    def net_phi(self,x):
        
        X = x[:, 0:1]
        uphi = self.neural_net(X, self.weights, self.biases)
        phi = uphi[:,1:2]
               
        return phi
    
    def net_hist(self,x):
        
        shape = tf.shape(x)
        init_hist = tf.zeros((shape[0],shape[1]), dtype = np.float64)
        dist = tf.abs(x-self.crackLoc)
        init_hist = tf.where(dist < 0.5*self.l, self.B*self.cEnerg*0.5*(1-(2*dist/self.l))/self.l, init_hist)
        
        return init_hist
    
    def net_energy(self,x):

        u = self.net_u(x)
        phi = self.net_phi(x)        
        hist = self.net_hist(x)
        
        g = (1-phi)**2
        phi_x = tf.gradients(phi,x)[0]
        nabla = phi_x**2
        phi_xx = tf.gradients(phi_x,x)[0]
        laplacian = phi_xx**2
        u_x = tf.gradients(u,x)[0]  
        sigmaX = self.E*u_x 
        
        energy_u = 0.5*g*sigmaX*u_x
        ext_work = tf.math.sin(m.pi*x)*u
        energy_phi = 0.5*self.cEnerg * (phi**2/self.l + self.l*nabla + 0.5*self.l**3*laplacian) + g* hist
        
        return energy_u, energy_phi, ext_work

    def net_f(self,x):

        u = self.net_u(x)
        phi = self.net_phi(x)
        
        g = (1-phi)**2        

        u_x = tf.gradients(u,x)[0]
        u_xx = tf.gradients(u_x,x)[0]

        f_u = -g*self.E*u_xx
        
        return f_u     
    
    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        print('Loss:', loss)

    def train(self, X_f, nIter):

        tf_dict = {self.x_f_tf: X_f[:,0:1],self.wt_f_tf: X_f[:,1:2]}
        self.loss_adam_buff = np.zeros(nIter)
        start_time = time.time()
        
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_adam_buff[it] = loss_value

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        tf_dict = {self.x_f_tf: X_star[:,0:1]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        
        return u_star, phi_star
    
    def predict_f(self, X_star):
        
        tf_dict = {self.x_f_tf: X_star[:,0:1]}                       
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        
        return f_u_star