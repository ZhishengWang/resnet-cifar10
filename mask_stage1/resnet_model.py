# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from numpy import nan
from pylint.checkers.similar import Similar

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages
from binary_mask import binaryStochastic_ST
import binary_mask as bm
HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []
    self.mask_loss = tf.constant(0.0)
    self.slope_tensor = 1.0

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self._build_paral_model()
    #self._build_model()
    #self._build_paral_bn2model()
    #self._build_violent_resnet()
    if self.mode == 'train':
      self._build_mask_train_op()
    self.summaries = tf.summary.merge_all()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))
    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func_norm = self._residual
      #res_func = self._residual_conv_mask
      res_func = self._residual_conv_mask
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      #filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 9
    count = 0
    with tf.variable_scope('unit_1_0'):
      x,mask_loss = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
      self.mask_loss = mask_loss
      count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x,mask_loss = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
        self.mask_loss += mask_loss
        count = count + 1
    with tf.variable_scope('unit_2_0'):
      x,mask_loss = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
      self.mask_loss += mask_loss
      count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x,mask_loss = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
        self.mask_loss += mask_loss
        count = count + 1
    
    with tf.variable_scope('unit_3_0'):
      x,mask_loss = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
      self.mask_loss += mask_loss
      count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x,mask_loss = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
        self.mask_loss += mask_loss
        count = count + 1      
    
    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)
      
    self.mask_loss = self.mask_loss/count
    
    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      #tf.summary.histogram('predictions_sum',tf.pow(2.0, -xent))
      self.cost += self._decay()
      self.cost += self.mask_loss
      tf.summary.scalar('cost', self.cost)
  
  def _build_paral_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))
      
    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      #mask, mask_loss = self._bttleneck_paral_mask(x, 3*self.hps.num_residual_units, self.slope_tensor, 'mask', sparsity = 0.50)
      #res_func = self._bottleneck_residual_para_mask
      filters = [16, 64, 128, 256]
    else:
      mask, mask_loss = self._cp_paral_mask(self._images, 3*self.hps.num_residual_units, self.slope_tensor, 'mask', sparsity = 0.4)
      self.mask = mask
      #res_func_norm = self._residual
      res_func = self._residual_para_mask
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      #filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 9
    count = 0
    with tf.variable_scope('unit_1_0'):
      mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),mask_slice,
                   activate_before_residual[0])
      count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), mask_slice,False)
        count = count + 1
    with tf.variable_scope('unit_2_0'):
      mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),mask_slice,
                   activate_before_residual[1])
      count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), mask_slice,False)
        count = count + 1
    
    with tf.variable_scope('unit_3_0'):
      mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
      x = res_func(x, filters[2], filters[3],self._stride_arr(strides[2]),mask_slice,
                   activate_before_residual[2])
      count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
      with tf.variable_scope('unit_3_%d' % i):
        x= res_func(x, filters[3], filters[3], self._stride_arr(1),mask_slice, False)
        count = count + 1      
    
    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)
      
    self.mask_loss = mask_loss
    
    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      #sm = tf.nn.softmax(logits)

      #regu = tf.reduce_mean(tf.reduce_sum(-sm*tf.log(sm),axis=1))
      #tf.summary.scalar('regularizer',regu)
      
      self.cost = tf.reduce_mean(xent, name='xent')
      tf.summary.histogram('predictions_sum',tf.pow(2.0, -xent))
      #sr = tf.cond(self.global_step <= 25000, lambda: 0.1 + tf.to_float(self.global_step)*0.3/25000, lambda: tf.constant(0.4))
      #self.cost -= regu*sr
      self.cost += self._decay()
      self.cost += self.mask_loss
      tf.summary.scalar('cost', self.cost)

  def _build_paral_bn2model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))
      
    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      #mask, mask_loss = self._bttleneck_paral_mask(x, 3*self.hps.num_residual_units, self.slope_tensor, 'mask', sparsity = 0.50)
      #res_func = self._bottleneck_residual_para_mask
      filters = [16, 64, 128, 256]
    else:
      mask, mask_loss = self._paral_mask(x, 3*self.hps.num_residual_units, self.slope_tensor, 'mask', sparsity = 0.4)
      self.mask = mask
      #res_func_norm = self._residual
      res_func = self._residual_para2bn_mask
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      #filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 9
    count = 0
    with tf.variable_scope('unit_1_0'):
      mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
      x, orig_x = res_func(x, None, filters[0], filters[1], self._stride_arr(strides[0]),mask_slice,
                   activate_before_residual[0])
      #count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
        x, orig_x = res_func(x, orig_x, filters[1], filters[1], self._stride_arr(1), mask_slice,False)
        count = count + 1
    with tf.variable_scope('unit_2_0'):
      mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
      x, orig_x = res_func(x, orig_x,filters[1], filters[2], self._stride_arr(strides[1]),mask_slice,
                   activate_before_residual[1])
      count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
        x, orig_x = res_func(x, orig_x, filters[2], filters[2], self._stride_arr(1), mask_slice,False)
        count = count + 1
    
    with tf.variable_scope('unit_3_0'):
      mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
      x, orig_x = res_func(x, orig_x, filters[2], filters[3],self._stride_arr(strides[2]),mask_slice,
                   activate_before_residual[2])
      count = count + 1
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
        x, orig_x= res_func(x, orig_x, filters[3], filters[3], self._stride_arr(1),mask_slice, False)
        count = count + 1      
    
    mask_slice = tf.slice(mask, [0,count], [self.hps.batch_size,1])
    count = count + 1
    
    with tf.variable_scope('unit_last'):
      mask_expand = tf.expand_dims(tf.expand_dims(mask_slice,axis=-1),axis=-1)
      decoder_mask = tf.tile(mask_expand,[1,tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3]])
      decoder_mask = tf.cond(self.global_step <= 55000, lambda:decoder_mask,lambda: bm.binaryRound(decoder_mask))
      x_bn1 = self._batch_norm('final_bn1', orig_x)
      x_bn2 = self._batch_norm('final_bn2', tf.add(orig_x,x))
      x = decoder_mask*x_bn2 + (1-decoder_mask)*x_bn1
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)
      
    self.mask_loss = mask_loss
    
    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      sm = tf.nn.softmax(logits)

      self.cost = tf.reduce_mean(xent, name='xent')
      tf.summary.histogram('predictions_sum',tf.pow(2.0, -xent))
      self.cost += self._decay()
      self.cost += self.mask_loss
      tf.summary.scalar('cost', self.cost)
              
  def _build_violent_resnet(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x_init = x
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))
    self.mask_loss = tf.constant(0.0)
    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      #res_func = self._residual_viol_mask
      res_func = self._residual
      filters = [16, 16, 32, 64]
      #x_init = x
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      #filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 9
    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
    #x, mask1 = self._conv_mask(x_init, x, self.slope_tensor, 1, 'violent_mask1')     
    
    
    with tf.variable_scope('unit_2_0'):
      x= res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    #x_init = x
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)  
    #x, mask2 = self._conv_mask(x_init, x, self.slope_tensor, 1, 'violent_mask2')
      
    
    
    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    #x_init = x
    
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)    
    #x, mask3 = self._conv_mask(x_init, x, self.slope_tensor, 1, 'violent_mask3')
    
    #self.mask_loss = mask3
    
    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)
      logits = self._fully_connected(x, self.hps.num_classes)
    
    with tf.variable_scope('mask_last'):
      mask, mask_loss = self._conv_mask_v2(x_init, self.slope_tensor, 'mask', sparsity = 0.50)
      mask = tf.tile(mask,[1,self.hps.num_classes])
      x_init = self._conv('mask_init_conv', x_init, 3, 3, 16, self._stride_arr(1))
      x_init = res_func(x_init, 16, 16, self._stride_arr(strides[0]), False)
      x_init = self._batch_norm('final_bn', x_init)
      x_init = self._relu(x_init, self.hps.relu_leakiness)
      x_init = self._global_avg_pool(x_init)
      logits_mask = self._fully_connected(x_init, self.hps.num_classes)
    
    self.mask_loss = mask_loss
    
    with tf.variable_scope('softmax'):
      logits = tf.add(logits*mask,logits_mask*(1-mask))
      #logits = tf.where(mask,logits_mask,logits)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      tf.summary.histogram('predictions_sum',tf.pow(2.0, -xent))

      self.cost += self._decay()
      self.cost += self.mask_loss
      tf.summary.scalar('cost', self.cost)
    tf.identity(self.cost, name='loss_by_example')
    

      
  def _build_vggmodel(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
          
    params = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', \
               'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    num_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    input_channels = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
    is_pool = [False,True,False,True,False,False,True,False,False,True,False,False,True]
    strides = [1,2,1,2,1,1,2,1,1,2,1,1,2]
    num_conv_layer = 2 #init: 13
    count = 0
    self.mask_loss = tf.constant(0.0)
    
    with tf.variable_scope(params[0]):
        x = self._conv('conv', x, filter_size=3, in_filters=input_channels[0], out_filters=num_filters[0], strides=[1,1,1,1])
        #init_x = x
        x = self._batch_norm('bn',x)
        x = self._relu(x, self.hps.relu_leakiness)
        if is_pool[0]:
            x = tf.nn.max_pool(x,ksize=[1, 2, 2, 1],strides = [1, 2, 2, 1], padding='SAME')   
    for i in six.moves.range(1, num_conv_layer):
        with tf.variable_scope(params[i]):
          x = self._conv('conv', x, filter_size=3, in_filters=input_channels[i], out_filters=num_filters[i], strides=[1,1,1,1])
          #x, mask_loss = self._mlid_mask(init_x, x, self.slope_tensor, 'mask', stride=strides[i-1])
          #self.mask_loss += mask_loss
          count = count + 1
          #init_x = x
          x = self._batch_norm('bn',x)
          x = self._relu(x, self.hps.relu_leakiness)
          if is_pool[i]:
            x = tf.nn.max_pool(x,ksize=[1, 2, 2, 1],strides = [1, 2, 2, 1], padding='SAME')            
    
    with tf.variable_scope('fc1'):
      x =  self._fully_connected(x, 4096)
      #x = self._batch_norm('bn_fc_1',x)
      x = tf.nn.dropout(x, 0.50)
    
    x = self._fully_connected_v2(x,'fc2', 4096)
    x = tf.nn.dropout(x, 0.50)
    logits =  self._fully_connected_v2(x,'fc3', self.hps.num_classes)

    self.predictions = tf.nn.softmax(logits)
    #self.mask_loss = self.mask_loss/count
    
    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()
      #self.cost += self.mask_loss
      tf.summary.scalar('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()   
    grads = tf.gradients(self.cost, trainable_variables)
    
    """
    2017.3.18 add weight clipping to avoid gradient exposing.
    """
    #clipped_grads,_ = tf.clip_by_global_norm(grads, 50.0)
    
    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    elif self.hps.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(self.lrn_rate)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)
  
  def _build_mask_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning rate', self.lrn_rate)

    #trainable_variables = tf.trainable_variables()
    
    mask_var, other_var = [], []
    #tf.trainable_variables().op.name.find()
    for var in tf.trainable_variables():
      if var.op.name.find(r'mask') >= 0:
        mask_var.append(var)
      else:
        other_var.append(var)

    optim1 = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    optim2 = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)            
    
    grads = tf.gradients(self.cost, other_var + mask_var)
    grads1 = grads[:len(other_var)]
    grads2 = grads[len(other_var):]

    apply_op1 = optim1.apply_gradients(
        zip(grads1, other_var),name='train_step1')
    apply_op2 = optim2.apply_gradients(
        zip(grads2, mask_var),
        global_step=self.global_step, name='train_step2')
    train_ops = [apply_op1] + [apply_op2] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y
  """
  input is expected to be a two-dimension matrix, not a high dimension tensor.
  """
  def _batch_norm_vec(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y
  
  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x
  
  def _residual_mask_v2(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        init_x = x
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        init_x = x
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x, mask_loss = self._mlid_mask(init_x, x, self.slope_tensor, 'mask')
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x, mask_loss
  
  def _residual_mask_v3(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
        init_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        init_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)
      x, mask_loss1 = self._mlid_mask(init_x, x, self.slope_tensor, 'mask1', True, stride[1])
      init_x = x
      
    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
      x, mask_loss2 = self._mlid_mask(init_x, x, self.slope_tensor, 'mask2', True, 1)

    tf.logging.debug('image after unit %s', x.get_shape())
    return x, mask_loss1 + mask_loss2

  def _residual_conv_mask(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        init_x = x
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        init_x = x
    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x, mask_loss = self._conv_mlid_mask(init_x, x, self.slope_tensor, 'mask', append_input=False, stride = stride[1])
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x, mask_loss

  def _residual_para_mask(self, x, in_filter, out_filter, stride,
                mask,activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):   
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
        x_init = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x_init = x

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    
    """
    
    with tf.variable_scope('sub_small'):
      x_small = self._conv('conv3', x_init, 1, in_filter, out_filter, stride)
    """
    
    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      
      mask_expand = tf.expand_dims(tf.expand_dims(mask,axis=-1),axis=-1)
      decoder_mask = tf.tile(mask_expand,[1,tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3]])

      #decoder_mask = tf.cond(self.global_step <= 33000, lambda:decoder_mask,lambda: bm.binaryRound(decoder_mask))
      #x = (1-decoder_mask)*x_small + decoder_mask*x
      x = decoder_mask*x
      #te = tf.cond(self.global_step <= 20000, lambda: 0.5 - tf.to_float(self.global_step)*0.5/20000, lambda: tf.constant(0.0))
      #x = tf.add(x*decoder_mask,x*(1-decoder_mask)*te)
      
      #here(tf.reshape(mask <= 0.5,[-1]), zeros, x)
      """
      x_mean = tf.reduce_mean(tf.abs(tf.reshape(x,[self.hps.batch_size,-1])),axis=-1)
      orig_x_mean = tf.reduce_mean(tf.abs(tf.reshape(orig_x,[self.hps.batch_size,-1])),axis=-1)
      rt = 1.0*x_mean/orig_x_mean
      mask_loss = tf.reduce_mean(tf.abs(rt-mask))
      
      mask_show = tf.to_float(tf.less_equal(rt, 0.1))
      tf.summary.scalar('mask_ratio', tf.reduce_mean(mask_show))
      
      fw = tf.where(rt <= 0.1, orig_x, x + orig_x)
      bw = x + orig_x
      x = bw + tf.stop_gradient(fw - bw)
      """
      #x = tf.cond(self.global_step <= 55000, lambda:decoder_mask*x,lambda: bm.binaryRound(decoder_mask)*x)
      #x = decoder_mask*x
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x
   
  def _residual_para2bn_mask(self, x, x_res, in_filter, out_filter, stride,
                mask,activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):   
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        
        mask_expand = tf.expand_dims(tf.expand_dims(mask,axis=-1),axis=-1)
        decoder_mask = tf.tile(mask_expand,[1,tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3]])
        decoder_mask = tf.cond(self.global_step <= 55000, lambda:decoder_mask,lambda: bm.binaryRound(decoder_mask))
        orig_x = x*decoder_mask + x_res
        
        x_bn1 = self._batch_norm('init_bn1', x_res)
        x_bn2 = self._batch_norm('init_bn2', x_res + x)
        x = decoder_mask*x_bn2 + (1-decoder_mask)*x_bn1
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    
    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      #x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x, orig_x

  def _residual_viol_mask(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        init_x = x
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        init_x = x

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x, mask_loss = self._conv_mask(init_x, x, self.slope_tensor, stride[1],'mask', sparsity = 0.25)
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x, mask_loss
 

  def _violent_mask(self, in_data, out_data, slope_tensor, stride,name_scope):
    
    a = in_data.get_shape().as_list()[1]
    b = in_data.get_shape().as_list()[2]
    if in_data.get_shape()[3].value >= 256:
        stride_0 = a
        stride_1 = b
    elif in_data.get_shape()[3].value >= 64:
        stride_0 = a/2
        stride_1 = b/2
    else:
        stride_0 = a/4
        stride_1 = b/4
    pooled_in_data = tf.nn.max_pool(in_data,ksize=[1, stride_0, stride_1, 1],strides = [1, stride_0, stride_1, 1], padding='VALID')  
    pooled_in_data = self._batch_norm(name_scope + '_bn', pooled_in_data)
    reshape = tf.reshape(pooled_in_data, [self.hps.batch_size, -1])
    dim_in = reshape.get_shape()[1].value
    #dim_out = out_data.get_shape()[3].value
    dim_out = 1
    with tf.variable_scope(name_scope):
        weights_en  = tf.get_variable(
            'DW', [dim_in, 1],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        bias_en = tf.get_variable('en_b', [1],initializer=tf.constant_initializer(0.0))
        encoder_mask = tf.matmul(reshape, weights_en) + bias_en #encoder_mask has size (batch_size, 1)
        encoder_mask = self._batch_norm_vec('mask_bn', encoder_mask)
    
    #mask = binaryStochastic_ST(encoder_mask,slope_tensor=slope_tensor, pass_through=False, stochastic=False) #,pass_through=False, 
    mask = tf.sigmoid(slope_tensor*encoder_mask)
    tf.summary.histogram('mask',mask)
    
    mask = bm.binaryRound(mask)
    
    mask_expand = tf.expand_dims(tf.expand_dims(mask,axis=-1),axis=-1)
    decoder_mask = tf.tile(mask_expand,[1,tf.shape(out_data)[1],tf.shape(out_data)[2],tf.shape(out_data)[3]])   
    
    in_filter = in_data.get_shape()[3].value
    out_filter = out_data.get_shape()[3].value
    if in_filter != out_filter:
        in_data = tf.nn.avg_pool(in_data, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
        in_data = tf.pad(
            in_data, [[0, 0], [0, 0], [0, 0],
                 [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
    
    #pre_activation_mask = tf.cond(mask, lambda: out_data, lambda: in_data)
    pre_activation_mask = tf.add(out_data*decoder_mask,in_data*(1-decoder_mask), name=name_scope +'_mask')
    #_activation_summary(pre_activation_mask)

    #self.slope_tensor = tf.cond(self.global_step <= 10000, lambda: 1.0 + tf.to_float(self.global_step)/10000, lambda: tf.constant(2.0))
    #s = tf.cond(self.global_step <= 10000, lambda: 1.0-tf.to_float(self.global_step)*0.75/10000, lambda: tf.constant(0.25))
    s = 0.4
    #mask_loss = -tf.reduce_sum(s*tf.log(mask) + (1-s)*tf.log(1-mask))/FLAGS.batch_size    
    mask_loss = 1.0*tf.reduce_mean(mask)
    tf.summary.scalar('mask_ratio_initial', mask_loss)
    
    mask_show = mask
    tf.summary.histogram('mask_tensor',mask_show)
    
    #mask_loss = tf.abs(mask_loss - s)
    """    
    if s >= 0.5:
        max_loss = s
    else:
        max_loss = 1-s
    self.slope_tensor = -0.5*mask_loss/max_loss + 1
    """
        
    return pre_activation_mask, mask_loss

  def _conv_mask(self, in_data, out_data, slope_tensor, stride,name_scope, sparsity = 0.25):
    
    #in_data = tf.identity(in_data)
    #in_data = tf.stop_gradient(in_data)
    
    #in_data_buf = self._batch_norm(name_scope + '/mask_bn_in', in_data)
    #in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)
    in_data_buf = self._conv(name_scope + '/mask_conv', x=in_data, filter_size=3, in_filters=in_data.get_shape().as_list()[3], out_filters=1, strides=[1,2,2,1])
    in_data_buf = self._batch_norm(name_scope + '/mask_bn_ou', in_data_buf)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)

    reshape = tf.reshape(in_data_buf, [self.hps.batch_size, -1])
    dim_in = reshape.get_shape()[1].value
    #dim_out = out_data.get_shape()[3].value
    dim_out = 1
    with tf.variable_scope(name_scope):
        weights_en  = tf.get_variable(
            'DW', [dim_in, 1],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        bias_en = tf.get_variable('en_b', [1],initializer=tf.constant_initializer(0.0))
        encoder_mask = tf.matmul(reshape, weights_en) + bias_en #encoder_mask has size (batch_size, 1)
        #encoder_mask = self._batch_norm_vec('mask_bn', encoder_mask)
        tf.summary.histogram(name_scope + '/encoder_mask',encoder_mask)
    
    #mask = binaryStochastic_ST(encoder_mask,slope_tensor=slope_tensor, pass_through=False, stochastic=False) #,pass_through=False, 
    
    #mask  = tf.minimum(tf.maximum(slope_tensor*encoder_mask, 0), 1)
    #mask = (1.5*tf.tanh(encoder_mask) + 0.5*tf.tanh(-3*encoder_mask))/2.0 + 0.5
    mask = tf.sigmoid(slope_tensor*encoder_mask)
    tf.summary.histogram(name_scope + '/mask',mask)
    
    mask = bm.binaryRound(mask)
    
    #mask = tf.cond(self.global_step <= 20000, lambda: mask, lambda: mask1)
    #mask = bm.bernoulliSample(mask)
    #tf.summary.histogram(name_scope + '/mask_tensor',mask)
    
    
    mask_expand = tf.expand_dims(tf.expand_dims(mask,axis=-1),axis=-1)
    decoder_mask = tf.tile(mask_expand,[1,tf.shape(out_data)[1],tf.shape(out_data)[2],tf.shape(out_data)[3]])   
    
    in_filter = in_data.get_shape()[3].value
    out_filter = out_data.get_shape()[3].value
    
    """
    if in_filter != out_filter:
        in_data = self._conv('append_conv', x=in_data, filter_size=3, in_filters=in_data.get_shape().as_list()[3], out_filters=out_data.get_shape().as_list()[3], strides=[1, stride, stride, 1])
    """
    """
    if in_filter != out_filter:
        in_data = tf.nn.avg_pool(in_data, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
        in_data = tf.pad(
            in_data, [[0, 0], [0, 0], [0, 0],
                 [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
    """

    te = tf.cond(self.global_step <= 5000, lambda: 0.5 - tf.to_float(self.global_step)*0.5/5000, lambda: tf.constant(0.0))
    pre_activation_mask = tf.add(out_data*decoder_mask,out_data*(1-decoder_mask)*te, name=name_scope +'_mask')
    #pre_activation_mask = out_data*decoder_mask
    
    #s = tf.cond(self.global_step <= 10000, lambda: 1.0-tf.to_float(self.global_step)*0.75/10000, lambda: tf.constant(0.25))
    s = sparsity
    mask_loss = 1.0*tf.reduce_mean(mask)
    tf.summary.scalar(name_scope + '/mask_ratio_initial', mask_loss)

    
    #mask_loss = tf.abs(s - mask_loss)
    #mask_loss = tf.cond(tf.less_equal(mask_loss, 0.05),lambda:tf.constant(0.05),lambda:mask_loss)
    #mask_loss = -(s*tf.log(mask_loss) + (1-s)*tf.log(1-mask_loss)) + (s*tf.log(s) + (1-s)*tf.log(1-s))
    """    
    if s >= 0.5:
        max_loss = s
    else:
        max_loss = 1-s
    self.slope_tensor = -0.5*mask_loss/max_loss + 1
    """
        
    return pre_activation_mask, mask_loss

  def _conv_mask_v2(self, in_data, slope_tensor ,name_scope, sparsity = 0.25):
    

    in_data_buf = self._batch_norm(name_scope + '/mask_bn_in', in_data)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)
    in_data_buf = self._conv(name_scope + '/mask_conv', x=in_data_buf, filter_size=3, in_filters=in_data.get_shape().as_list()[3], out_filters=1, strides=[1,2,2,1])
    in_data_buf = self._batch_norm(name_scope + '/mask_bn_ou', in_data_buf)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)

    reshape = tf.reshape(in_data_buf, [self.hps.batch_size, -1])
    dim_in = reshape.get_shape()[1].value
    #dim_out = out_data.get_shape()[3].value
    dim_out = 1
    with tf.variable_scope(name_scope):
        weights_en  = tf.get_variable(
            'DW', [dim_in, 1],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        bias_en = tf.get_variable('en_b', [1],initializer=tf.constant_initializer(0.0))
        encoder_mask = tf.matmul(reshape, weights_en) + bias_en #encoder_mask has size (batch_size, 1)
        encoder_mask = self._batch_norm_vec('mask_bn', encoder_mask)
        tf.summary.histogram(name_scope + '/encoder_mask',encoder_mask)
    #mask  = tf.minimum(tf.maximum(slope_tensor*encoder_mask, 0), 1)
    #mask = (1.5*tf.tanh(encoder_mask) + 0.5*tf.tanh(-3*encoder_mask))/2.0 + 0.5

    #mask = tf.sigmoid(slope_tensor*encoder_mask)
    mask = (1.5*tf.tanh(encoder_mask) + 0.5*tf.tanh(-3*encoder_mask))/2.0 + 0.5
    tf.summary.histogram(name_scope + '/mask',mask)

    mask = bm.bernoulliSample(mask)
    tf.summary.histogram(name_scope + '/mask_tensor',mask)
    
    """
    threshold = tf.cond(self.global_step <= 15000, lambda: 1.0-tf.to_float(self.global_step)*0.51/15000, lambda: tf.constant(0.51))
    rand = tf.random_uniform(shape=mask.get_shape(), minval=0.0, maxval=threshold, dtype=tf.float32)
    rand_bin = tf.to_float(tf.less_equal(0.5, rand))
    mask =  (1-mask)*rand_bin + mask
    """

    s = sparsity
    mask_loss = 1.0*tf.reduce_mean(mask)
    tf.summary.scalar(name_scope + '/mask_ratio_initial', mask_loss)
    mask_loss = tf.abs(mask_loss - s)
        
    return mask, mask_loss


  def _conv_mlid_mask(self, in_data, out_data, slope_tensor, name_scope, append_input=False, stride = 1):
    
    #in_data_buf = self._batch_norm(name_scope + '/mask_bn_in', in_data)
    #in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)
    #in_data = tf.stop_gradient(in_data)
    in_data_buf = self._conv(name_scope + '/mask_conv', x=in_data, filter_size=3, in_filters=in_data.get_shape().as_list()[3], out_filters=1, strides=[1,2,2,1])
    in_data_buf = self._batch_norm(name_scope + '/mask_bn_ou', in_data_buf)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)

    reshape = tf.reshape(in_data_buf, [self.hps.batch_size, -1])
    dim_in = reshape.get_shape()[1].value
    dim_out = out_data.get_shape()[3].value

    with tf.variable_scope('encoder'):
        weights_en  = tf.get_variable(
            'DW', [dim_in, 2],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        biases_en = tf.get_variable(name_scope + 'bias', [2],
                        initializer=tf.constant_initializer(0.0))
    #decoder_mask = tf.matmul(reshape, weights_en) + biases_en #encoder_mask has size (batch_size, 1)
    encoder_mask = tf.tanh(tf.matmul(reshape, weights_en) + biases_en) #encoder_mask has size (batch_size, 1)
    #encoder_mask = self._relu(tf.matmul(reshape, weights_en) + biases_en, self.hps.relu_leakiness)
    #encoder_mask = self._batch_norm_vec(name_scope + '/mask_bn', encoder_mask)

    with tf.variable_scope('dencoder'):
        weights_de  = tf.get_variable(
            name_scope + 'DW', [2, dim_out],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        #biases_de = tf.get_variable(name_scope + 'bias', [dim_out],initializer=tf.constant_initializer(0.0))
    decoder_mask = tf.matmul(encoder_mask, weights_de)# + biases_de #encoder_mask has size (batch_size, 1)
    decoder_mask = self._batch_norm_vec(name_scope + '/mask_de', decoder_mask)
    
    """
    mean = 0.0
    std = tf.cond(self.global_step <= 50000, lambda: 0.2 + tf.to_float(self.global_step)*3.8/50000, lambda: tf.constant(4.0))
    dist = tf.contrib.distributions.Normal(mean, std)
    smp = dist.sample(sample_shape=dim_out*self.hps.batch_size)
    smp = tf.reshape(smp, [self.hps.batch_size, dim_out])
    decoder_mask = tf.add(decoder_mask, smp)   
    """
    
    self.slope_tensor = tf.cond(self.global_step <= 32000, lambda: 1.0 + 0.04*tf.floor(tf.to_float(self.global_step/400)), lambda: tf.constant(5.0))
    mask = tf.sigmoid(self.slope_tensor*decoder_mask)

    tf.summary.histogram(name_scope + '/mask',mask)
    
    mask = bm.binaryRound(mask)
    #mask = bm.bernoulliSample(mask)
    
    mask_expand = tf.expand_dims(tf.expand_dims(mask,axis=1),axis=1)
    decoder_mask = tf.tile(mask_expand,[1,tf.shape(out_data)[1],tf.shape(out_data)[2],1])
    
    #pre_activation_mask = tf.multiply(out_data ,decoder_mask, name=name_scope +'_mask')
    if append_input:
        in_filter = in_data.get_shape()[3].value
        out_filter = out_data.get_shape()[3].value
        if in_filter != out_filter:
            in_data = tf.nn.avg_pool(in_data, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
            in_data = tf.pad(
                in_data, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        pre_activation_mask = tf.add(out_data*decoder_mask,in_data*(1-decoder_mask), name=name_scope +'_mask')
    else:
        pre_activation_mask = decoder_mask*out_data
    
    s = 0.40
    
    
    avg_s_each_acti = tf.reduce_mean(mask,axis=0)
    #l1 = tf.reduce_mean(tf.abs(avg_s_each_acti - s))
    #tf.summary.scalar(name_scope + '/l1', l1)
    tf.summary.histogram(name_scope + '/avg_data', avg_s_each_acti)
    
    
    avg_s_each_batch = tf.reduce_mean(mask,axis=1)
    #l2 = tf.reduce_mean(tf.abs(avg_s_each_acti - s))
    #tf.summary.scalar(name_scope + '/l2', l2)
    tf.summary.histogram(name_scope + '/avg_bath', avg_s_each_batch)
    
    """
    l3 = -tf.reduce_sum(tf.reduce_mean(tf.square(tf.sub(mask, avg_s_each_acti)),axis=0))
    tf.summary.scalar(name_scope + '/l3', l3)
    """
    
    l1  = tf.abs(tf.reduce_mean(mask) - s)
    mask_loss = l1
    
    return pre_activation_mask, mask_loss

  def _mlid_mask(self, in_data, out_data, slope_tensor, name_scope, append_input=False, stride = 1):
    
    a = in_data.get_shape().as_list()[1]
    b = in_data.get_shape().as_list()[2]
    if in_data.get_shape()[3].value >= 256:
        stride_0 = a
        stride_1 = b
    elif in_data.get_shape()[3].value >= 64:
        stride_0 = a/2
        stride_1 = b/2
    else:
        stride_0 = a/4
        stride_1 = b/4
    pooled_in_data = tf.nn.max_pool(in_data,ksize=[1, stride_0, stride_1, 1],strides = [1, stride_0, stride_1, 1], padding='SAME')        
    
    reshape = tf.reshape(pooled_in_data, [self.hps.batch_size, -1])
    dim_in = reshape.get_shape()[1].value
    
    dim_out = out_data.get_shape()[3].value

    weights_en  = tf.get_variable(
        name_scope + '_DW_en', [dim_in, 2],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    biases_en = tf.get_variable(name_scope + '_en_b', [2],
                        initializer=tf.constant_initializer())
    #encoder_mask = tf.sigmoid(tf.matmul(reshape, weights_en) + biases_en) #encoder_mask has size (batch_size, 1)
    encoder_mask = tf.tanh(tf.matmul(reshape, weights_en) + biases_en) #encoder_mask has size (batch_size, 1)
    weights_de  = tf.get_variable(
        name_scope + '_DW_de', [2, dim_out],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    biases_de = tf.get_variable(name_scope + '_de_b', [dim_out],
                        initializer=tf.constant_initializer())
    decoder_mask = tf.matmul(encoder_mask, weights_de) + biases_de #encoder_mask has size (batch_size, 1)
    mask = binaryStochastic_ST(decoder_mask,slope_tensor=slope_tensor ,pass_through=False, stochastic=False)
    #mask = binaryStochastic_REINFORCE(decoder_mask, stochastic = False, loss_op_name="loss_by_example")
    #mask = binaryStochastic_RE(decoder_mask, slope_tensor=slope_tensor, stochastic=False)
    
    mask_expand = tf.expand_dims(tf.expand_dims(mask,axis=1),axis=1)
    decoder_mask = tf.tile(mask_expand,[1,tf.shape(out_data)[1],tf.shape(out_data)[2],1])   
    
    #pre_activation_mask = tf.multiply(out_data ,decoder_mask, name=name_scope +'_mask')
    if append_input:
        in_filter = in_data.get_shape()[3].value
        out_filter = out_data.get_shape()[3].value
        if in_filter != out_filter:
            in_data = tf.nn.avg_pool(in_data, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
            in_data = tf.pad(
                in_data, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        pre_activation_mask = tf.add(out_data*decoder_mask,in_data*(1-decoder_mask), name=name_scope +'_mask')
    else:
        pre_activation_mask = tf.multiply(out_data ,decoder_mask, name=name_scope +'_mask')

    #mask_loss = -tf.reduce_sum(s*tf.log(mask) + (1-s)*tf.log(1-mask))/FLAGS.batch_size    
    mask_loss = 1.0*tf.reduce_mean(mask)
    tf.summary.scalar('mask_ratio_initial', mask_loss)
    
    mask_show = tf.reduce_mean(mask, axis=0)
    tf.summary.histogram('mask_tensor',mask_show)
    
    #s = tf.cond(self.global_step <= 10000, lambda: 1.0-tf.to_float(self.global_step)*0.75/10000, lambda: tf.constant(0.25))
    s = 0.25
    mask_loss = tf.abs(mask_loss - s)
    
    return pre_activation_mask, mask_loss


  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _cal_maskloss(self):
    """Calculating mask loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'mask_cal') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)
    sum_loss = tf.add_n(costs)
    tf.summary.scalar('weight_loss', sum_loss)        
    return tf.multiply(self.hps.weight_decay_rate, sum_loss)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _fully_connected_v2(self, x, name, out_dim):
    """FullyConnected layer for final output."""
    #x = tf.reshape(x, [self.hps.batch_size, -1])
    with tf.variable_scope(name):
      w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
      return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
  
  def _paral_mask(self, in_data, num_masks, slope_tensor, name_scope, sparsity = 0.25):
    
    #in_data = tf.stop_gradient(in_data)

    in_data_buf = self._batch_norm(name_scope + '/mask_bn_in', in_data)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)
    in_data_buf = self._conv(name_scope + '/mask_conv', x=in_data, filter_size=3, in_filters=in_data.get_shape().as_list()[3], out_filters=2, strides=[1,2,2,1])
    in_data_buf = self._batch_norm(name_scope + '/mask_bn_ou', in_data_buf)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)
 
    
    #in_data_buf = tf.nn.max_pool(in_data_buf, [1,4,4,1], [1,4,4,1], padding='SAME')
    
    reshape = tf.reshape(in_data_buf, [self.hps.batch_size, -1])
    dim_in = reshape.get_shape()[1].value
    #dim_out = out_data.get_shape()[3].value

    with tf.variable_scope(name_scope):
        weights_en  = tf.get_variable(
            'DW', [dim_in, num_masks],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        #bias_en = tf.get_variable('en_b', [num_masks],initializer=tf.constant_initializer(0.0))
        encoder_mask = tf.matmul(reshape, weights_en)#encoder_mask has size (batch_size, 1)
        encoder_mask = self._batch_norm_vec('mask_bn', encoder_mask)
        tf.summary.histogram(name_scope + '/encoder_mask',encoder_mask)
    """
    mean = 0.0
    std = tf.cond(self.global_step <= 30000, lambda: 0.2 + tf.to_float(self.global_step)*2.8/30000, lambda: tf.constant(3.0))
    dist = tf.contrib.distributions.Normal(mean, std)
    smp = dist.sample(sample_shape=num_masks*self.hps.batch_size)
    smp = tf.reshape(smp, [self.hps.batch_size, num_masks])
    encoder_mask = tf.add(encoder_mask, smp)    
    """
    #mask = bm.passThroughSigmoid(encoder_mask)
    
    #fw = tf.sigmoid(encoder_mask)
    #bw = tf.identity(encoder_mask)
    #mask = bw + tf.stop_gradient(fw - bw)
    
    self.slope_tensor = tf.cond(self.global_step <= 32000, lambda: 1.0 + 0.04*tf.floor(tf.to_float(self.global_step/400)), lambda: tf.constant(5.0))
    mask = tf.sigmoid(self.slope_tensor*encoder_mask)
  
    tf.summary.histogram(name_scope + '/mask_sigmoid',mask)
    mask_out = mask

    mask = bm.binaryRound(mask) #resnet 
    #mask = bm.bernoulliSample(mask) #viol
    tf.summary.histogram(name_scope + '/mask',mask)
    
    s = sparsity
    #s = tf.constant([0.6,0.25,0.25,0.25,0.25,0.6,0.2,0.2,0.2,0.2,0.6,0.6,0.6,0.6,0.6])
    
    avg_s_each_acti = 1.0*tf.reduce_mean(mask,axis=0)
    tf.summary.histogram(name_scope + '/avg_acti',avg_s_each_acti)
    mask_loss = tf.reduce_mean(tf.abs(avg_s_each_acti - s))
    
    avg_s_each_data = tf.reduce_mean(mask,axis=1)
    tf.summary.histogram(name_scope + '/avg_data',avg_s_each_data)
    
    #mask_loss = 0.8*tf.abs(tf.reduce_mean(mask) -s) + 0.2*mask_loss
    #mask_loss = tf.abs(tf.reduce_mean(mask) -s)
    
    """
    s = tf.expand_dims(s,axis=[-1])
    s = tf.tile(s,[num_masks])
    """
    mask_loss = tf.reduce_mean(mask)
    #mask_loss = avg_s_each_acti
    mask_loss = tf.where(tf.equal(mask_loss, -0.0) | tf.equal(mask_loss, 1.0), -(s*tf.log(s) + (1-s)*tf.log(1-s)), -(s*tf.log(mask_loss/s) + (1-s)*tf.log((1-mask_loss)/(1-s))))
    #mask_loss = tf.reduce_mean(mask_loss)
    
    
    tf.summary.scalar(name_scope + '/mask_loss', mask_loss)
    return mask, mask_loss
  
  def _cp_paral_mask(self, in_data, num_masks, slope_tensor, name_scope, sparsity = 0.25):
    
    #in_data = tf.stop_gradient(in_data)

    #in_data_buf = self._batch_norm(name_scope + '/mask_bn_in', in_data)
    #in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)
    
    in_data_buf = self._conv(name_scope + '/mask_conv', x=in_data, filter_size=3, in_filters=in_data.get_shape().as_list()[3], out_filters=16, strides=[1,2,2,1])
    in_data_buf = self._batch_norm(name_scope + '/mask_bn_ou', in_data_buf)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)

    in_data_buf = self._conv(name_scope + '/mask_conv2', x=in_data, filter_size=3, in_filters=in_data.get_shape().as_list()[3], out_filters=2, strides=[1,2,2,1])
    in_data_buf = self._batch_norm(name_scope + '/mask_bn_ou2', in_data_buf)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)
    
    #in_data_buf = tf.nn.max_pool(in_data_buf, [1,4,4,1], [1,4,4,1], padding='SAME')
    
    reshape = tf.reshape(in_data_buf, [self.hps.batch_size, -1])
    dim_in = reshape.get_shape()[1].value
    #dim_out = out_data.get_shape()[3].value

    with tf.variable_scope(name_scope):
        weights_en  = tf.get_variable(
            'DW', [dim_in, num_masks],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        #bias_en = tf.get_variable('en_b', [num_masks],initializer=tf.constant_initializer(0.0))
        encoder_mask = tf.matmul(reshape, weights_en)#encoder_mask has size (batch_size, 1)
        encoder_mask = self._batch_norm_vec('mask_bn', encoder_mask)
        tf.summary.histogram(name_scope + '/encoder_mask',encoder_mask)
    """
    mean = 0.0
    std = tf.cond(self.global_step <= 30000, lambda: 0.2 + tf.to_float(self.global_step)*2.8/30000, lambda: tf.constant(3.0))
    dist = tf.contrib.distributions.Normal(mean, std)
    smp = dist.sample(sample_shape=num_masks*self.hps.batch_size)
    smp = tf.reshape(smp, [self.hps.batch_size, num_masks])
    encoder_mask = tf.add(encoder_mask, smp)    
    """
    
    #mask = bm.passThroughSigmoid(encoder_mask)
    
    #fw = tf.sigmoid(encoder_mask)
    #bw = tf.identity(encoder_mask)
    #mask = bw + tf.stop_gradient(fw - bw)
    
    self.slope_tensor = tf.cond(self.global_step <= 32000, lambda: 1.0 + 0.04*tf.floor(tf.to_float(self.global_step/400)), lambda: tf.constant(5.0))
    #mask = tf.sigmoid(self.slope_tensor*encoder_mask)
    mask = (1.5*tf.tanh(self.slope_tensor*encoder_mask) + 0.5*tf.tanh(-3*self.slope_tensor*encoder_mask))/2.0 + 0.5
    tf.summary.histogram(name_scope + '/mask_sigmoid',mask)
    mask_out = mask

    mask = bm.binaryRound(mask) #resnet 
    #mask = bm.bernoulliSample(mask) #viol
    tf.summary.histogram(name_scope + '/mask',mask)
    
    s = sparsity
    #s = tf.constant([0.6,0.25,0.25,0.25,0.25,0.6,0.2,0.2,0.2,0.2,0.6,0.6,0.6,0.6,0.6])
    
    avg_s_each_acti = 1.0*tf.reduce_mean(mask,axis=0)
    tf.summary.histogram(name_scope + '/avg_acti',avg_s_each_acti)
    #mask_loss = tf.reduce_mean(tf.abs(avg_s_each_acti - s))
    
    avg_s_each_data = tf.reduce_mean(mask,axis=1)
    tf.summary.histogram(name_scope + '/avg_data',avg_s_each_data)
    
    #mask_loss = 0.8*tf.abs(tf.reduce_mean(mask) -s) + 0.2*mask_loss
    #mask_loss = tf.abs(tf.reduce_mean(mask) -s)
    
    
    """
    """
    s = tf.expand_dims(s,axis=[-1])
    s = tf.tile(s,[num_masks])
    
    
    mask_loss = avg_s_each_acti
    mask_loss = tf.where(tf.equal(mask_loss, -0.0) | tf.equal(mask_loss, 1.0), -(s*tf.log(s) + (1-s)*tf.log(1-s)), -(s*tf.log(mask_loss/s) + (1-s)*tf.log((1-mask_loss)/(1-s))))
    mask_loss = tf.reduce_mean(mask_loss)
    
    """
    KL_D + bernoullisample(train) results in better sparsity control, but can not get better results.
    ~40.9% sparsity,90.2% accuracy 
    """
    #mask_loss = tf.reduce_mean(mask)
    tf.summary.scalar(name_scope + '/mask_loss', mask_loss)
    return mask, mask_loss