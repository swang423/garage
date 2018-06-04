## A splice layer that could add context to speech frames
## sc, 06/04/2018 @Alibaba

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MySplice(Layer):
  def __init__(self, splice, **kwargs):
    assert splice >= 0
    self.splice = splice
    super(MySplice, self).__init__(**kwargs)

  def build(self, input_shape):
    super(MySplice, self).build(input_shape)  # Be sure to call this at the end

  def call(self, x):
    if self.splice == 0:
      return x
    else:
      n = self.splice
      y = K.expand_dims(x,axis=0)
      y = K.concatenate((K.tile(y[:,0,:],(n,1)),
                        x,
                        K.tile(y[:,-1,:],(n,1))),axis=0)
      nc = int(x.shape[1])
      z = x[:,0:0]
      for nn in range(-n,n+1):
        row_offset = nn + n
        col_offset = row_offset * nc
        row_end = nn - n
        if row_end == 0:
          row_end = None
        z = K.concatenate((z,y[row_offset:row_end,:] ),axis=1)
    #   z[:,col_offset:col_offset+nc] = y[row_offset:row_end,:] #indexing into tensor assignment of z is not allowed
      return z
  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) >= 2
    assert input_shape[-1]
    output_shape = list(input_shape)
    output_shape[-1] = input_shape[-1]*(2*self.splice+1)
    return tuple(output_shape)

  def get_config(self):
    config = {
          'splice':self.splice
    }
    base_config = super(MySplice,self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

"""
#Alternatively, implemnted with Lambda and some test
n = 2
params = {'n':n}
setattr(K, 'params', params)

def splice_shape(input_shape):
  shape = list(input_shape)
  shape[-1]*=2
  return tuple(shape)

def slice(x):
  y = K.expand_dims(x,axis=0)
  xh = K.tile(y[:,0,:],(K.params['n'],1))
  xt = K.tile(y[:,-1,:],(K.params['n'],1))
  y = K.concatenate((xh,x,xt),axis=0)
# z = K.zeros((x.shape[0],x.shape[1]*(2*K.params['n']+1)))
# z = K.zeros_like(K.tile(x,(1,2*K.params['n']+1)))
  nc = int(x.shape[1])
  z = x[:,0:0]
  for nn in range(-K.params['n'],K.params['n']+1):
    row_offset = nn + K.params['n']
    col_offset = row_offset * nc
    row_end = nn - K.params['n']
    if row_end == 0:
      row_end = None
    z = K.concatenate((z,y[row_offset:row_end,:] ),axis=1)
#   z[:,col_offset:col_offset+nc] = y[row_offset:row_end,:]
  return z
#x1 = Lambda(slice)(x0)
x1 = MySplice(splice=2)(x0)
"""

