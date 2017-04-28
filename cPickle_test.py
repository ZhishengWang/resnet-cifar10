import cifar_input
import numpy as np
import resnet_model_expert
import resnet_model
import tensorflow as tf
import cPickle

file = open('prob.pkl','rb')

data = cPickle.load(file)
arr = np.array(data)

print 'shape of the list: ', np.shape(data)
print 'number of bytes per value: ', arr[1,0].nbytes
m,n = np.shape(arr)
count = 0
for i in range(m):
    if arr[i][0] >= 0.80:
        count = count + 1
print 'the ratio of expected sparsity: ',1.0 - count*1.0/m

arr.tofile("prob_pkl.bin")

file.close()