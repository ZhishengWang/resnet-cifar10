from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from enum import Enum
import os
import re
import sys
import tarfile

from six.moves import urllib


def layer_linear(inputs, shape, scope='linear_layer'):
    with tf.variable_scope(scope):
        w = tf.get_variable('w',shape)
        b = tf.get_variable('b',shape[-1:])
    return tf.matmul(inputs,w) + b

def layer_softmax(inputs, shape, scope='softmax_layer'):
    with tf.variable_scope(scope):
        w = tf.get_variable('w',shape)
        b = tf.get_variable('b',shape[-1:])
    return tf.nn.softmax(tf.matmul(inputs,w) + b)

def accuracy(y, pred):
    correct = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

class StochasticGradientEstimator(Enum):
    ST = 0
    REINFORCE = 1

def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)

        # For Tensorflow v0.11 and below use:
        #with g.gradient_override_map({"Floor": "Identity"}):
        #    return tf.round(x, name=name)

def SigmRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("SigmRound") as name:
        with g.gradient_override_map({"Round": "Sigmoid"}):
            return (tf.sign(x, name=name)+1)/2

        # For Tensorflow v0.11 and below use:
        #with g.gradient_override_map({"Floor": "Identity"}):
        #    return tf.round(x, name=name)

def approx_nonlinear(slope,x):
    g = tf.get_default_graph()

    with ops.name_scope("approx_nonlinear") as name:
        with g.gradient_override_map({"Add": "Identity","Mul":""}):
            return  tf.clip_by_value(tf.add(tf.mul(slope/6,x),0.5),0,1)
        tf.round(x, name=name)


def bernoulliSample(x):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    E.g.,:
    if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
    and the gradient will be pass-through (identity).
    """
    g = tf.get_default_graph()

    with ops.name_scope("BernoulliSample") as name:
        with g.gradient_override_map({"Ceil": "Identity","Sub": "BernoulliSample_ST"}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)

@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]

def passThroughSigmoid(x, slope=1):
    """Sigmoid that uses identity function as its gradient"""
    g = tf.get_default_graph()
    with ops.name_scope("PassThroughSigmoid") as name:
        with g.gradient_override_map({"Sigmoid": "Identity"}):
            return tf.sigmoid(x, name=name)

def binaryStochastic_RE(x, slope_tensor, stochastic=False):
    """
    ReLU followed by either a random sample from a bernoulli distribution according
    to the result (binary stochastic neuron) (default), or a ReLU followed by a binary
    step function (if stochastic == False). Uses the straight through estimator.
    (Experimental)

    Arguments:
    * x: the pre-activation / logit tensor
    * stochastic: binary stochastic neuron if True (default), or step function if False
    """
    if slope_tensor is None:
        slope_tensor = tf.constant(1.0)
    #p = tf.nn.relu(x)
    p = tf.maximum(0.0, tf.minimum(1.0, (slope_tensor*x + 0.5)))

    if stochastic:
        return bernoulliSample(p)
    else:
        return binaryRound(p)
def binaryStochastic_ST(x, slope_tensor=None, pass_through=True, stochastic=True):
    """
    Sigmoid followed by either a random sample from a bernoulli distribution according
    to the result (binary stochastic neuron) (default), or a sigmoid followed by a binary
    step function (if stochastic == False). Uses the straight through estimator.
    See https://arxiv.org/abs/1308.3432.

    Arguments:
    * x: the pre-activation / logit tensor
    * slope_tensor: if passThrough==False, slope adjusts the slope of the sigmoid function
        for purposes of the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)
    * pass_through: if True (default), gradient of the entire function is 1 or 0;
        if False, gradient of 1 is scaled by the gradient of the sigmoid (required if
        Slope Annealing Trick is used)
    * stochastic: binary stochastic neuron if True (default), or step function if False
    """
    if slope_tensor is None:
        slope_tensor = tf.constant(1.0)

    if pass_through:
        p = passThroughSigmoid(x)
    else:
        p = tf.sigmoid(slope_tensor*x)

    if stochastic:
        return bernoulliSample(p)
    else:
        return binaryRound(p)
    
def binaryStochastic_REINFORCE(x, stochastic = True, loss_op_name="loss_by_example"):
    """
    Sigmoid followed by a random sample from a bernoulli distribution according
    to the result (binary stochastic neuron). Uses the REINFORCE estimator.
    See https://arxiv.org/abs/1308.3432.

    NOTE: Requires a loss operation with name matching the argument for loss_op_name
    in the graph. This loss operation should be broken out by example (i.e., not a
    single number for the entire batch).
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryStochasticREINFORCE"):
        with g.gradient_override_map({"Sigmoid": "BinaryStochastic_REINFORCE",
                                      "Ceil": "Identity"}):
            p = tf.sigmoid(x)

            reinforce_collection = g.get_collection("REINFORCE")
            if not reinforce_collection:
                g.add_to_collection("REINFORCE", {})
                reinforce_collection = g.get_collection("REINFORCE")
            reinforce_collection[0][p.op.name] = loss_op_name

            return tf.ceil(p - tf.random_uniform(tf.shape(x)))


@ops.RegisterGradient("BinaryStochastic_REINFORCE")
def _binaryStochastic_REINFORCE(op, _):
    """Unbiased estimator for binary stochastic function based on REINFORCE."""
    loss_op_name = op.graph.get_collection("REINFORCE")[0][op.name]
    loss_tensor = op.graph.get_operation_by_name(loss_op_name).outputs[0]

    sub_tensor = op.outputs[0].consumers()[0].outputs[0] #subtraction tensor
    ceil_tensor = sub_tensor.consumers()[0].outputs[0] #ceiling tensor

    outcome_diff = (ceil_tensor - op.outputs[0])

    # Provides an early out if we want to avoid variance adjustment for
    # whatever reason (e.g., to show that variance adjustment helps)
    if op.graph.get_collection("REINFORCE")[0].get("no_variance_adj"):
        return outcome_diff * tf.expand_dims(loss_tensor, 1)

    outcome_diff_sq = tf.square(outcome_diff)
    outcome_diff_sq_r = tf.reduce_mean(outcome_diff_sq, reduction_indices=0)
    outcome_diff_sq_loss_r = tf.reduce_mean(outcome_diff_sq * tf.expand_dims(loss_tensor, 1),
                                            reduction_indices=0)

    L_bar_num = tf.Variable(tf.zeros(outcome_diff_sq_r.get_shape()), trainable=False)
    L_bar_den = tf.Variable(tf.ones(outcome_diff_sq_r.get_shape()), trainable=False)

    #Note: we already get a decent estimate of the average from the minibatch
    decay = 0.95
    train_L_bar_num = tf.assign(L_bar_num, L_bar_num*decay +\
                                            outcome_diff_sq_loss_r*(1-decay))
    train_L_bar_den = tf.assign(L_bar_den, L_bar_den*decay +\
                                            outcome_diff_sq_r*(1-decay))


    with tf.control_dependencies([train_L_bar_num, train_L_bar_den]):
        L_bar = train_L_bar_num/(train_L_bar_den+1e-4)
        L = tf.tile(tf.expand_dims(loss_tensor,1),
                    tf.constant([1,L_bar.get_shape().as_list()[0]]))
        return outcome_diff * (L - L_bar)

def binary_wrapper(\
                pre_activations_tensor,
                estimator=StochasticGradientEstimator.ST,
                stochastic_tensor=tf.constant(True),
                pass_through=True,
                slope_tensor=tf.constant(1.0)):
    """
    Turns a layer of pre-activations (logits) into a layer of binary stochastic neurons

    Keyword arguments:
    *estimator: either ST or REINFORCE
    *stochastic_tensor: a boolean tensor indicating whether to sample from a bernoulli
        distribution (True, default) or use a step_function (e.g., for inference)
    *pass_through: for ST only - boolean as to whether to substitute identity derivative on the
        backprop (True, default), or whether to use the derivative of the sigmoid
    *slope_tensor: for ST only - tensor specifying the slope for purposes of slope annealing
        trick
    """

    if estimator == StochasticGradientEstimator.ST:
        if pass_through:
            return tf.cond(stochastic_tensor,
                    lambda: binaryStochastic_ST(pre_activations_tensor),
                    lambda: binaryStochastic_ST(pre_activations_tensor, stochastic=False))
        else:
            return tf.cond(stochastic_tensor,
                    lambda: binaryStochastic_ST(pre_activations_tensor, slope_tensor = slope_tensor,
                                             pass_through=False),
                    lambda: binaryStochastic_ST(pre_activations_tensor, slope_tensor = slope_tensor,
                                             pass_through=False, stochastic=False))
    elif estimator == StochasticGradientEstimator.REINFORCE:
        # binaryStochastic_REINFORCE was designed to only be stochastic, so using the ST version
        # for the step fn for purposes of using step fn at evaluation / not for training
        return tf.cond(stochastic_tensor,
                lambda: binaryStochastic_REINFORCE(pre_activations_tensor),
                lambda: binaryStochastic_ST(pre_activations_tensor, stochastic=False))

    else:
        raise ValueError("Unrecognized estimator.")
