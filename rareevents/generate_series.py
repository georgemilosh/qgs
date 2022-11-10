## Systems definition
#General parameters. I am selecting standard parameters for the runs below, not sure if this is what Robin is using
# Time parameters
import sys, os
sys.path.extend([os.path.abspath('../')])
import numpy as np
import matplotlib.pyplot as plt
import time
start = time.time()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
rc('font',**{'family':'serif','sans-serif':['Times'],'size':14})
import xarray as xr
np.random.seed(1234) # Initializing the random number generator (for reproducibility). -- Disable if needed.

from qgs.params.params import QgParams #Importing the model's modules
from qgs.integrators.integrator import RungeKuttaIntegrator, RungeKuttaTglsIntegrator
from qgs.functions.tendencies import create_tendencies
from qgs.plotting.util import std_plot

from qgs.diagnostics.streamfunctions import MiddleAtmosphericStreamfunctionDiagnostic # and diagnostics
from qgs.diagnostics.variables import VariablesDiagnostic
from qgs.diagnostics.multi import MultiDiagnostic

from qgs.diagnostics.temperatures import GroundTemperatureDiagnostic
from qgs.diagnostics.temperatures import MiddleAtmosphericTemperatureDiagnostic

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
#### Custom Metrics ######

class UnbiasedMetric(keras.metrics.Metric):
    def __init__(self, name, undersampling_factor=1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.undersampling_factor = undersampling_factor
        self.r = tf.cast([0.5*np.log(undersampling_factor), -0.5*np.log(undersampling_factor)], tf.float32)

    
class MCCMetric(UnbiasedMetric): # This function is designed to produce confusion matrix during training each epoch
    def __init__(self, num_classes=2, threshold=None, undersampling_factor=1, name='MCC', **kwargs):
        '''
        Mathews correlation coefficient metric

        Parameters
        ----------
        num_classes : int, optional
            number of classes, by default 2
        threshold : float, optional
            If num_classes == 2 allows to choose a threshold over which to consider an event positive. If None the event is positive if it has probability higher than 0.5. By default None
        '''
        super().__init__(name=name, undersampling_factor=undersampling_factor, **kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.threshold = threshold
        if self.num_classes != 2:
            raise NotImplementedError('MCC works only with 2 classes')
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
    
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
            
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
    
    @tf.autograph.experimental.do_not_convert
    def result(self):
        #return self.process_confusion_matrix()
        cm=self.total_cm
        
        #return cm
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        TP = cm[1,1]
        MCC_den = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        MCC = (TP * TN - FP *FN)/tf.sqrt(MCC_den)
        return tf.cond(MCC_den == 0, lambda: tf.constant(0, dtype=tf.float32), lambda: MCC)
    
    def confusion_matrix(self,y_true, y_pred): # Make a confusion matrix
        if self.undersampling_factor > 1 or self.threshold is not None:
            y_pred = keras.layers.Softmax()(y_pred + self.r) # apply shift of logits and softmax to convert to balanced probabilities
        if self.threshold is None:
            y_pred=tf.argmax(y_pred,1)
        else:
            y_pred = tf.cast(y_pred[:,1] > self.threshold, tf.int8)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def fill_output(self,output):
        results=self.result()
        
class ConfusionMatrixMetric(UnbiasedMetric): # This function is designed to produce confusion matrix during training each epoch
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes, undersampling_factor=1, name='confusion_matrix', **kwargs):
        super().__init__(name=name, undersampling_factor=undersampling_factor, **kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
            
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
        
    def result(self):
        #return self.process_confusion_matrix()
        cm=self.total_cm
        return cm
    
    def confusion_matrix(self,y_true, y_pred): # Make a confusion matrix
        if self.undersampling_factor > 1:
            y_pred = keras.layers.Softmax()(y_pred + self.r) # apply shift of logits and softmax to convert to balanced probabilities
        
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def fill_output(self,output):
        results=self.result()

class BrierScoreMetric(UnbiasedMetric):
    def __init__(self, undersampling_factor=1, name='BrierScore', **kwargs):
        super().__init__(name=name, undersampling_factor=undersampling_factor, **kwargs)
        self.mse = keras.metrics.MeanSquaredError()
        self.my_metric = self.add_weight(name='BScore', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        _ = self.mse.update_state(y_true, keras.layers.Softmax()(y_pred + self.r))
        self.my_metric.assign(self.mse.result())

    def result(self):
        return self.my_metric

class UnbiasedCrossentropyMetric(UnbiasedMetric):

  def __init__(self, undersampling_factor=1, name='UnbiasedCrossentropy', **kwargs):
    super().__init__(name=name, undersampling_factor=undersampling_factor, **kwargs)
    self.my_metric = self.add_weight(name='CLoss', initializer='zeros')
    self.m = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)

  def update_state(self, y_true, y_pred, sample_weight=None):
    #_ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    _ = self.m.update_state(y_true, y_pred+self.r) # the idea is to add the weight factor inside the logit so that we effectively change the probabilities
    self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric


# same as above but less elegant
class CustomLoss(tf.keras.metrics.Metric):

  def __init__(self, r, name='CustomLoss', **kwargs):
    super().__init__(name=name, **kwargs)
    self.my_metric = self.add_weight(name='CLoss', initializer='zeros')
    self.r = r # undersampling_factor array (we expect the input as tf.cast(-0.5*np.log(undersampling_factor), 0.5*np.log(undersampling_factor))
    self.m = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)


  def update_state(self, y_true, y_pred, sample_weight=None):
    #_ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    _ = self.m.update_state(y_true, y_pred+self.r) # the idea is to add the weight factor inside the logit so that we effectively change the probabilities
    self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric

  """ 
  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.my_metric.assign(0.) 
  """

class UnbiasedCrossentropyLoss(keras.losses.SparseCategoricalCrossentropy):
    '''
    This is the same as the UnbiasedCrossentropyMetric but can be used as a loss
    '''
    def __init__(self, undersampling_factor=1, name='unbiased_crossentropy_loss'):
        super().__init__(from_logits=True, name=name)
        self.r = tf.cast([0.5*np.log(undersampling_factor), -0.5*np.log(undersampling_factor)], tf.float32)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred + self.r)

class MyMetrics_layer(tf.keras.metrics.Metric):

  def __init__(self, name='MyMetrics_layer', **kwargs):
    super(MyMetrics_layer, self).__init__(name=name, **kwargs)
    self.my_metric = self.add_weight(name='my_metric1', initializer='zeros')
    self.m = tf.keras.metrics.SparseCategoricalAccuracy()


  def update_state(self, y_true, y_pred, sample_weight=None):
    _ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.my_metric.assign(0.)
"""
class CustomLoss(tf.keras.metrics.Metric):

  def __init__(self, name='custom_loss', **kwargs):
    super(CustomLoss, self).__init__(name=name, **kwargs)
    #self.custom_loss = self.add_weight(name='closs', initializer='zeros')
    self.scce=tf.keras.losses.SparseCategoricalCrossentropy()#from_logits=True), #If the predicted labels are not converted to a probability distribution by the last layer of the model (using sigmoid or softmax activation functions), we need to inform these t

  def update_state(self, y_true, y_pred, sample_weight=None):
    update_state_out = self.scce(y_true,y_pred)
    update_state_out = y_true.shape
    self.custom_loss.assign_add(update_state_out)

  def result(self):
    #return self.custom_loss
    return self.scce(y_true,y_pred)
"""

def create_Logistic(model_input_dim, regularizer='l2', regularizer_coef=1e-9):
    inputs = keras.Input(shape=model_input_dim) # Create a standard model (alternatively we can use our function my_create_logistic_model which builds a custom sequential model
    x = tf.keras.layers.Flatten(input_shape=model_input_dim)(inputs)
    if regularizer == 'none':
        outputs = tf.keras.layers.Dense(2)(x)
    elif regularizer == 'l2':
        outputs = tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l2(regularizer_coef))(x)
    else:
        outputs = tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l1(regularizer_coef))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)   # alternatively we can write:     model = MyModel(inputs=inputs, outputs=outputs)
    return model


def autocorrelation(myseries, maxlag):
    # this pads each year with padsize sample time of 0s so that when the array is permuted to be multiplied by itself we don't end up using the previous part of the year
    series_pad = np.pad(myseries,((0, 0), (0, maxlag)), 'constant')  
    autocorr = []
    for k in range(maxlag):
        autocorr.append(np.sum(series_pad*np.roll(series_pad, -k))/(series_pad.shape[0]*(series_pad.shape[1]-k-maxlag)))
    return autocorr

def PltAutocorrelation(series_in,bootstrapnumber,timelenx,ax):
    autocorr = []
    series = series_in[:bootstrapnumber*(len(series_in)//bootstrapnumber)] # remove the extra stuff
    series_bootstrapped = series.reshape(bootstrapnumber,-1)
    print(f'{series_bootstrapped.shape =}')
    for i in range(bootstrapnumber):
        autocorr.append(autocorrelation(series_bootstrapped[i:(i+1),:],timelenx))
    autocorr = np.array(autocorr)#/np.std(t2m.abs_area_int[i:(i+1),:])**2 # uncomment this is autocorrelation is seeked for (so that at time = 0 it is normalized to 1)
    print(f'{autocorr.shape = }')
    autocorr_mean = np.mean(autocorr,0)
    autocorr_std = np.std(autocorr,0)
    print(f'{autocorr.shape = }, {autocorr_mean.shape = },{autocorr_std.shape = }')
    ax.fill_between((groundT._time[:len(autocorr_mean)]*groundT._model_params.dimensional_time), autocorr_mean-autocorr_std, autocorr_mean+autocorr_std,color='yellow')
    print(f'{series.reshape(1,-1).shape = }')
    full_auto = autocorrelation(series.reshape(1,-1),timelenx)
    ax.plot((groundT._time[:len(full_auto)]*groundT._model_params.dimensional_time),full_auto,label='Gaussian')
    return autocorr_mean, autocorr_std

def ComputeTimeAverage(series,T=14,tau=0, percent=5, threshold=None): 
        '''
        Computes running mean from time series
        '''
        print(f'{T = }')
        convseq = np.ones(T)/T
        A=np.convolve(series,  convseq, mode='valid')
        if threshold is None:
            threshold = np.sort(A)[np.ceil(A.shape[0]*(1-percent/100)).astype('int')]
        list_extremes = list(A >= threshold)
        print(f'{len(list_extremes) = }')
        return A, threshold, list_extremes, convseq

#The other parameters can come from model1.pickle that Robin prepared:
import pickle
# loading the model
with open('model1.pickle', "rb") as file:
    model = pickle.load(file)

f = model['f']
Df = model['Df']
model_parameters = model['parameters']
# Printing the model's parameters
model_parameters.print_params()

dt =  0.01/model_parameters.dimensional_time # we made dt a bit smaller so that we get daily output
# Saving the model state n steps
write_steps = 100 # It seems to me that time = 18 is interpreted as 1 day by some routines. This way we will force the output to be daily (write every `write_steps` iterations)

number_of_trajectories = 1
number_of_perturbed_trajectories = 10
#Why did Robin choose n = 0.353? Otherwise the setup is similar to ground_heat.ipynb but with more spectral components. It is not clear to me if we need many spectral components. The typical setup with fewer components is usually enough to provide some realism. 

#Parameters that are different:
#* Atmospheric Temperature parameters
#    * 'C[1]' : 122 [W][m^-2] instead of 99 [W][m^-2]: (spectral component 1 of the short-wave radiation of the atmosphere), other C's are zero
#    * 'hlambda': 20  [W][m^-2][K^-1] instead of 10 [W][m^-2][K^-1] (sensible+turbulent heat exchange between ocean and atmosphere),
#* Ground Parameters:
#    * 'hk[2]': 0.4 instead of 0.2    (spectral components 2 of the orography), other h'k are zero
#* Ground Temperature Parameters:
#    * 'gamma': 200000000  [J][m^-2][K^-1] instead of 16000000  (specific heat capacity of the ground),
#    * 'C[1]': 280.0  [W][m^-2]  (spectral component 1 of the short-wave radiation of the ground), other C parameters are zero
#Below we can also investigate what fields that are actually used 
# model_parameters.latex_var_string

# Now we create the test and train trajectories.
ndays=1000000 # how many days to generate


f, Df, ic, times, reference_traj, reference_time, MiddleT, psi, groundT, timedimensional, A, threshold, list_extremes = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
for sample_set in ['tr','va']:
    trajfilename = f'data/traj_{sample_set}_{ndays}.nc' # to create new trajectories rename the file or remove it form the directory
    suffix = 0
    while os.path.exists(trajfilename): 
        suffix = suffix + 1
        trajfilename = f'data/traj_{sample_set}_{ndays}_{suffix}.nc'

    trajfilename = f'{trajfilename}'
    f[sample_set], Df[sample_set] = create_tendencies(model_parameters)
    print('tendencies created')
    ## Time integration
    #Defining an integrator
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f[sample_set])
    #Start on a random initial condition and integrate over a transient time to obtain an initial condition on the attractors. I kept the standard time that is typically used in other runs, so not sure if it perfectly adapted...
    print('integrator set up')
    ic[sample_set] = np.random.rand(model_parameters.ndim)*0.1
    integrator.integrate(0., 30000/model_parameters.dimensional_time, dt, ic=ic[sample_set], write_steps=0) # actually 200000 should be enought according to rare_statistics.ipynb
    times[sample_set], ic[sample_set] = integrator.get_trajectories()
    print(f"done generating initial coordinates with {times[sample_set] = } and {times[sample_set]*model_parameters.dimensional_time = }")

    #Now integrate to obtain a trajectory on the attractor
    integrator.integrate(0., ndays/model_parameters.dimensional_time, dt, ic=ic[sample_set], write_steps=write_steps)
    reference_time[sample_set], reference_traj[sample_set] = integrator.get_trajectories()
    ds = xr.Dataset(data_vars=dict(traj=(["comp", "time"], reference_traj[sample_set])), coords=dict(time=reference_time[sample_set]), attrs=dict(description="Components of the trajectory."))
    print(f'saving {trajfilename}')
    ds.to_netcdf(trajfilename)

    end = time.time()
    print(f"Total elapsed time since the initialization: {end - start = }")
