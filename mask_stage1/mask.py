  def _paral_mask(self, in_data, num_masks, slope_tensor, name_scope, sparsity = 0.25):
    
    """
    uncommit this to avoid that the gradients of the mask affect other bottom layers 
    """
    #in_data = tf.stop_gradient(in_data)

    in_data_buf = self._batch_norm(name_scope + '/mask_bn_in', in_data)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)
    in_data_buf = self._conv(name_scope + '/mask_conv', x=in_data, filter_size=3, in_filters=in_data.get_shape().as_list()[3], out_filters=2, strides=[1,2,2,1])
    in_data_buf = self._batch_norm(name_scope + '/mask_bn_ou', in_data_buf)
    in_data_buf = self._relu(in_data_buf, self.hps.relu_leakiness)

    reshape = tf.reshape(in_data_buf, [self.hps.batch_size, -1])
    dim_in = reshape.get_shape()[1].value

    with tf.variable_scope(name_scope):
        weights_en  = tf.get_variable(
            'DW', [dim_in, num_masks],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        #bias_en = tf.get_variable('en_b', [num_masks],initializer=tf.constant_initializer(0.0))
        encoder_mask = tf.matmul(reshape, weights_en)#encoder_mask has size (batch_size, 1)
        encoder_mask = self._batch_norm_vec('mask_bn', encoder_mask)
        tf.summary.histogram(name_scope + '/encoder_mask',encoder_mask)

    """
    below: add normal distribution noise to the mask. 
    
    (optional)with this used, change the first output from "mask" to "mask_out" and add:
        decoder_mask = tf.cond(self.global_step <= 35000, lambda:decoder_mask,lambda: bm.binaryRound(decoder_mask))
    inside "self._residual_para_mask"
    """
    """
    mean = 0.0
    std = tf.cond(self.global_step <= 30000, lambda: 0.2 + tf.to_float(self.global_step)*2.8/30000, lambda: tf.constant(3.0))
    dist = tf.contrib.distributions.Normal(mean, std)
    smp = dist.sample(sample_shape=num_masks*self.hps.batch_size)
    smp = tf.reshape(smp, [self.hps.batch_size, num_masks])
    encoder_mask = tf.add(encoder_mask, smp)    
    """
    
    """
    uncommit this to use increased slope tensor
    """
    #self.slope_tensor = tf.cond(self.global_step <= 32000, lambda: 1.0 + 0.04*tf.floor(tf.to_float(self.global_step/400)), lambda: tf.constant(5.0))
    
    mask = tf.sigmoid(self.slope_tensor*encoder_mask)
  
    tf.summary.histogram(name_scope + '/mask_sigmoid',mask)
    mask_out = mask

    
    mask = bm.binaryRound(mask)
    
    """
    compared to binaryRound, bernoulliSample(mask) may be more robust to input changes, but not guarantee to get better result.
    sometimes train with bernoulliSample(mask) and eval with binaryRound(mask) can get better result (depends on current training strategy)
    """
    #mask = bm.bernoulliSample(mask)
    
    tf.summary.histogram(name_scope + '/mask',mask)
    
    s = sparsity
    
    """
    below: use different sparsity for each layer
    """
    #s = tf.constant([0.8,0.8,0.8,0.8,0.8,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3])
    
    avg_s_each_acti = 1.0*tf.reduce_mean(mask,axis=0)
    tf.summary.histogram(name_scope + '/avg_acti',avg_s_each_acti)
    mask_loss = tf.reduce_mean(tf.abs(avg_s_each_acti - s))
    
    avg_s_each_data = tf.reduce_mean(mask,axis=1)
    tf.summary.histogram(name_scope + '/avg_data',avg_s_each_data)
    
    """
    below: use KL-Divergence instead of L1-loss. generally KL-divergence will have better control of expected sparsity (NOTE: not accuracy), but it is not stable sometimes and may results \
    in NaN value of the parameters
    """
    """
    s = tf.expand_dims(s,axis=[-1])
    s = tf.tile(s,[num_masks])
    
    mask_loss = avg_s_each_acti
    mask_loss = tf.where(tf.equal(mask_loss, -0.0) | tf.equal(mask_loss, 1.0), -(s*tf.log(s) + (1-s)*tf.log(1-s)), -(s*tf.log(mask_loss/s) + (1-s)*tf.log((1-mask_loss)/(1-s))))
    mask_loss = tf.reduce_mean(mask_loss)*0.5 #0.5 is used to avoid the nan value in parameters
    """
    tf.summary.scalar(name_scope + '/mask_loss', mask_loss)
    
    return mask, mask_loss