#!/usr/bin/env python


import tensorflow as tf
import numpy as np
import warnings

from .Models import Model, def_W_init, def_b_init


class SriniNet(Model):

    __default_params = {
      "n_fmaps": [10,10,10,10,10],
      "n_hlayers": 5,
      "n_out"  : 3,
      "W_init"  : def_W_init,
      "b_init"  : def_b_init
    }

    def concretize(self, params, pl_in):

        #Fetching parameters from the args
        n_fmaps = params["n_fmaps"]
        n_hidden_layers = params["n_hlayers"]
        n_out = params["n_out"]

        W_init = params["W_init"]
        b_init = params["b_init"]

        assert( n_hidden_layers == len(n_fmaps) )
        assert( n_hidden_layers >= 1 )


        #First hidden layer
        with tf.name_scope("h1"):
          W = tf.Variable(W_init([1,7,7,1,n_fmaps[0]]), name="W")
          b = tf.Variable(b_init([n_fmaps[0]]), name="b")
          h = tf.nn.relu(tf.nn.conv3d(pl_in,W,strides=[1,1,1,1,1],padding="VALID") + b)


        #Defining hidden layers between the and the output layers
        for i in range(1,n_hidden_layers):

          with tf.name_scope("h{}".format(i+1)):
            W = tf.Variable(W_init([1,7,7,n_fmaps[i-1],n_fmaps[i]]), name="W")
            b = tf.Variable(b_init([n_fmaps[i]]), name="b")
            h = tf.nn.relu(tf.nn.conv3d(h,W,strides=[1,1,1,1,1],padding="VALID") + b)
    

        #Output layer
        with tf.name_scope("output"):
          Wo = tf.Variable(W_init([1,7,7,n_fmaps[-1],n_out]), name="W")
          bo = tf.Variable(b_init([n_out]), name="b")
          o  = tf.nn.conv3d(h,Wo,strides=[1,1,1,1,1],padding="VALID") + bo


        #Defining the fov now that the model is finished
        xy_fov = (n_hidden_layers+1) * 6 + 1
        self.fov = np.array([1,xy_fov,xy_fov])

        return o

    def __init__(self, *input_vars, **new_params):
        """
        Initializes a model given an input variables/placeholders
        and any modified parameters (specified by keyword arguments).
        The result is created within the model attribute of the class
        after instantiation
        """
        
        params = self.__default_params
        
        valid_params = params.keys()
        for k,v in new_params.items(): 
          if k in valid_params:
            params[k] = v
          else:
            warnings.warn("Invalid parameter \"{}\" passed".format(k))
        
        self.fov = None #default if not concretized properly
        self.params = params
        self.model = self.concretize(params, *input_vars)

