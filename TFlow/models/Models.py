#!/usr/bin/env python3


import numpy as np
import tensorflow as tf
import warnings


class Model:
    """
    Model abstraction which (hopefully) allows for
    easy experimentation by changing hyperparameters
    by keyword arguments. Input variables are passed by
    plain arguments to the constructor, and hyperparameters
    are specified by keyword argument
    """


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


    def concretize(self, params, *input_vars):
        """Creates an explicit model using the filled in parameters, and
        any input variables passed to the constructor"""

        warnings.warn("Invoking base model - passing back input variables")
        return input_vars


    __default_params = {}


#Default initialization functions for weight and bias variables
def_W_init = lambda shape : tf.truncated_normal(shape, stddev=0.1)
def_b_init = lambda shape : tf.constant(0.1, shape=shape)


