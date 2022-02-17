import os
import numpy as np
import math
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, log_loss
import time

from keras.datasets import mnist
from keras.datasets import cifar10

class global_vars:
    input_path = ""
    in_filename1 = input_path + "archi.txt"
    in_toy_filename = input_path + "toy_archi.txt"
    temp_in_filename = input_path + "temp_archi.txt"
    
    in_filename2 = ""
    toy_train_filename = input_path + "trainNN.txt"
    toy_test_filename = input_path + "testNN.txt"
    out_filename1 = ""

    color_channel_count = 0
    sample_count = 32
    check_total_sample_count = 0
    iter_count  = 5
    toy_iter_count = 40
    # prev_channel_count  = 0
    set_nan = 0
    class_count = 10
    toy_class_count = 4

    learning_rate = 0.001

class global_lists:
    check_list = [1, 2, 3]

gb_var_obj = global_vars()
gb_list_obj = global_lists()

gb_var_obj.out_filename1 = "metrics.txt"

f2 = open(gb_var_obj.out_filename1, "a")
f2.truncate(0)

class conv_layer:
    layer_name = ""
    output_channel_count = 0
    filter_dim = 0
    stride = 0
    padding = 0

    input_shape = (0)
    input_arr = np.array([])
    cur_filter = np.array([])
    cur_bias = np.array([])
    output_arr = np.array([])

    d_filter = np.array([])
    d_bias = np.array([])
    d_input = np.array([])

    set_conv_init_param = 0
    set_conv_nan = 0



    def myprint(self):
        print(self.layer_name)
        print(self.output_channel_count)
        print(self.filter_dim)
        print(self.stride)
        print(self.padding)
        # pass

class activation_layer:
    layer_name = ""
    input_arr = np.array([])
    output_arr = np.array([])

    def myprint(self):
        print(self.layer_name)

class pool_layer:
    layer_name = ""
    filter_dim = 0
    stride = 0
    padding = 0 # fixed
    input_arr = np.array([])
    output_arr = np.array([])

    def myprint(self):
        print(self.layer_name)
        print(self.filter_dim)
        print(self.stride)

class fc_layer:
    layer_name = ""
    output_dim = 0

    cur_filter = np.array([])
    cur_bias = np.array([])
    input_arr = np.array([])
    flat_input_arr = np.array([])
    output_arr = np.array([])

    d_filter = np.array([])
    d_bias = np.array([])
    d_input = np.array([])

    set_fc_init_param = 0
    set_conv_nan = 0

    def myprint(self):
        print(self.layer_name)
        print(self.output_dim)

class flat_layer:
    layer_name = ""
    input_shape = (0)
    input_arr = np.array([])
    output_arr = np.array([])

class softmax_layer:
    layer_name = ""
    input_arr = np.array([])
    one_hot_arr = np.array([])
    output_arr = np.array([])

    def myprint(self):
        print(self.layer_name)


# ---------------------------------- dataset fucntions
def preprocessing_mnist(train_X, train_y, test_X, test_y):
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
    train_y = train_y.reshape(train_y.shape[0], 1)
    
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
    test_y = test_y.reshape(test_y.shape[0], 1)

    return train_X, train_y, test_X, test_y

def load_mnist_dataset():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X, train_y, test_X, test_y = preprocessing_mnist(train_X, train_y, test_X, test_y)

    return train_X, train_y, test_X, test_y

def load_cifar10_dataset():
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()

    return train_X, train_y, test_X, test_y

def print_dataset_shape(train_X, train_y, valid_X, valid_y, test_X, test_y):
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_valid: ' + str(valid_X.shape))
    print('Y_valid: ' + str(valid_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))


def get_input():
    with open(gb_var_obj.in_filename1) as f:
        lines = f.readlines()
    
    archi_list = []

    for line in lines:
        temp_line = line.split()

        if temp_line[0] == "Conv":
            conv_layer_inputs = conv_layer()
            conv_layer_inputs.layer_name = temp_line[0]
            conv_layer_inputs.output_channel_count = int(temp_line[1])
            conv_layer_inputs.filter_dim = int(temp_line[2])
            conv_layer_inputs.stride = int(temp_line[3])
            conv_layer_inputs.padding = int(temp_line[4])

            archi_list.append(conv_layer_inputs)

            gb_var_obj.prev_channel_count = conv_layer_inputs.output_channel_count

        elif temp_line[0] == "ReLU":
            activation_layer_inputs = activation_layer()
            activation_layer_inputs.layer_name = temp_line[0]

            archi_list.append(activation_layer_inputs)
        
        elif temp_line[0] == "Pool":
            pool_layer_inputs = pool_layer()
            pool_layer_inputs.layer_name = temp_line[0]
            pool_layer_inputs.filter_dim = int(temp_line[1])
            pool_layer_inputs.stride = int(temp_line[2])

            archi_list.append(pool_layer_inputs)

        elif temp_line[0] == "FC":
            fc_layer_inputs = fc_layer()
            fc_layer_inputs.layer_name = temp_line[0]
            fc_layer_inputs.output_dim = int(temp_line[1])

            archi_list.append(fc_layer_inputs)

        elif temp_line[0] == "Flatten":
            flat_layer_inputs = flat_layer()
            flat_layer_inputs.layer_name = "Flatten"

            archi_list.append(flat_layer_inputs)

        elif temp_line[0] == "Softmax":
            softmax_layer_inputs = softmax_layer()
            softmax_layer_inputs.layer_name = temp_line[0]

            archi_list.append(softmax_layer_inputs)

    return archi_list


def get_input_2(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    archi_list = []

    for line in lines:
        temp_line = line.split()

        if temp_line[0] == "Conv":
            conv_layer_inputs = conv_layer()
            conv_layer_inputs.layer_name = temp_line[0]
            conv_layer_inputs.output_channel_count = int(temp_line[1])
            conv_layer_inputs.filter_dim = int(temp_line[2])
            conv_layer_inputs.stride = int(temp_line[3])
            conv_layer_inputs.padding = int(temp_line[4])

            archi_list.append(conv_layer_inputs)

            gb_var_obj.prev_channel_count = conv_layer_inputs.output_channel_count

        elif temp_line[0] == "ReLU":
            activation_layer_inputs = activation_layer()
            activation_layer_inputs.layer_name = temp_line[0]

            archi_list.append(activation_layer_inputs)
        
        elif temp_line[0] == "Pool":
            pool_layer_inputs = pool_layer()
            pool_layer_inputs.layer_name = temp_line[0]
            pool_layer_inputs.filter_dim = int(temp_line[1])
            pool_layer_inputs.stride = int(temp_line[2])

            archi_list.append(pool_layer_inputs)

        elif temp_line[0] == "FC":
            fc_layer_inputs = fc_layer()
            fc_layer_inputs.layer_name = temp_line[0]
            fc_layer_inputs.output_dim = int(temp_line[1])

            archi_list.append(fc_layer_inputs)

        elif temp_line[0] == "Flatten":
            flat_layer_inputs = flat_layer()
            flat_layer_inputs.layer_name = "Flatten"

            archi_list.append(flat_layer_inputs)

        elif temp_line[0] == "Softmax":
            softmax_layer_inputs = softmax_layer()
            softmax_layer_inputs.layer_name = temp_line[0]

            archi_list.append(softmax_layer_inputs)

    return archi_list

def get_toy_input():
    with open(gb_var_obj.in_toy_filename) as f:
        lines = f.readlines()

    archi_list = []

    for line in lines:
        temp_line = line.split()

        if temp_line[0] == "FC":
            fc_layer_inputs = fc_layer()
            fc_layer_inputs.layer_name = temp_line[0]
            fc_layer_inputs.output_dim = int(temp_line[1])

            archi_list.append(fc_layer_inputs)

        elif temp_line[0] == "Flatten":
            flat_layer_inputs = flat_layer()
            flat_layer_inputs.layer_name = "Flatten"

            archi_list.append(flat_layer_inputs)

        elif temp_line[0] == "Softmax":
            softmax_layer_inputs = softmax_layer()
            softmax_layer_inputs.layer_name = temp_line[0]

            archi_list.append(softmax_layer_inputs)

    return archi_list

def get_new_dim(prev_dim, filter_dim, padding, stride):
    new_dim = int(( (prev_dim - filter_dim + 2*padding) / stride ) + 1)
    return new_dim

def get_subarray_list(full_arr, filter_dim, stride):
# get subarrays with the shape of filter_dim
    sub_arr_list = []
    i = 0
    # while i < full_arr.shape[0]-filter_dim+1:
    iter_range = full_arr.shape[0] - filter_dim + 1
    while i < iter_range:
        j = 0
        while j < iter_range:
            sub_arr = full_arr[i:(i+filter_dim), j:(j+filter_dim)]
            sub_arr_list.append(sub_arr)
            j = j + stride

        i = i + stride
    
    return sub_arr_list

# ---------------------------------- convolution layer ------------------------------------------
def get_single_conv(single_arr, single_filter, layer_inputs, output_dim, cur_bias):
# apply one (2d)filter by sliding on a single (2d)array
    new_h = output_dim
    new_w = output_dim
    output_arr = np.zeros((new_h, new_w))

    sub_arr_list = get_subarray_list(single_arr, layer_inputs.filter_dim, layer_inputs.stride)
    
    x = 0
    for i in range(output_arr.shape[0]):
        for j in range(output_arr.shape[1]):
            sub_arr = sub_arr_list[x]
            mult_res = np.multiply(sub_arr, single_filter)
            sum_res = np.sum(mult_res)
            output_arr[i, j] = sum_res

            # output_arr[i, j] += cur_bias

            x = x + 1

    return output_arr

def conv_forward_prop(batch_input_arr, layer_inputs, layer_no):
    layer_inputs.input_arr = batch_input_arr
    layer_inputs.input_shape = batch_input_arr.shape

    input_dim = batch_input_arr.shape[1]
    output_dim = get_new_dim(input_dim, layer_inputs.filter_dim, layer_inputs.padding, layer_inputs.stride)
    new_h = output_dim
    new_w = output_dim
    sample_count = batch_input_arr.shape[0]
    output_arr = np.zeros((sample_count, new_h, new_w, layer_inputs.output_channel_count))

    for m in range(len(batch_input_arr)):
        input_arr = batch_input_arr[m]

        feature_map_count  = input_arr.shape[2]

        cur_filter = layer_inputs.cur_filter
        cur_bias = layer_inputs.cur_bias

        for i in range(layer_inputs.output_channel_count):
            for j in range(feature_map_count):
                sub_filter = cur_filter[:,:,j,i]
                sub_input = input_arr[:,:,j]

                # padding
                sub_input = np.pad(sub_input, layer_inputs.padding, mode='constant', constant_values=0) # np.pad(a, 1, mode='constant')
    
                output_arr[m,:,:,i] += get_single_conv(sub_input, sub_filter, layer_inputs, output_dim, cur_bias[i])

            output_arr[m,:,:,i] += cur_bias[i]

    layer_inputs.output_arr = output_arr
    return output_arr

def conv_backward_prop(batch_d_output, layer_inputs, learning_rate, layer_no):
    d_filter = np.zeros(layer_inputs.cur_filter.shape)
    d_bias = np.zeros((layer_inputs.output_channel_count))
    batch_d_input = np.zeros(layer_inputs.input_shape)

    filter_dim = layer_inputs.filter_dim
    padding  = layer_inputs.padding

    batch_input_arr_with_pad = np.pad(layer_inputs.input_arr, ((0,0), (padding, padding), (padding, padding), (0,0)), mode='constant', constant_values=0)
    batch_d_input_with_pad = np.pad(batch_d_input, ((0,0), (padding, padding), (padding, padding), (0,0)), mode='constant', constant_values=0)
    
    for m in range(len(batch_d_output)):
        d_output = batch_d_output[m]
        input_arr = layer_inputs.input_arr[m]
        input_arr_with_pad = batch_input_arr_with_pad[m]
        d_input_with_pad = batch_d_input_with_pad[m]

        h = 0
        while h < d_output.shape[0]:
            w = 0
            while w < d_output.shape[1]:
                for c in range(layer_inputs.output_channel_count):
                    u = h * layer_inputs.stride
                    v = w * layer_inputs.stride

                    input_slice = input_arr_with_pad[u:(u+filter_dim), v:(v+filter_dim), :]
                    d_filter[:,:,:,c] += input_slice * d_output[h,w,c]
                    d_bias[c] += d_output[h,w,c]

                    d_input_with_pad[u:(u+filter_dim), v:(v+filter_dim), :] += layer_inputs.cur_filter[:,:,:,c] * d_output[h,w,c]
                
                w = w + 1

            h = h + 1

        if padding != 0:
            batch_d_input[m,:,:,:] = d_input_with_pad[padding:-padding, padding:-padding, :]

    layer_inputs.d_filter = d_filter
    layer_inputs.d_bias = d_bias
    layer_inputs.d_input = batch_d_input
    
    return d_filter, d_bias, batch_d_input


# # ---------------------------------- activation layer ------------------------------------------
def my_ReLU(x):
    factor = 0.001
    res = np.where(x > 0, factor * x, 0)
    return res

def my_ReLu_deriv(x):
    res = np.where(x > 0, 1, 0)
    return res


def activation_forward_prop(input_arr, layer_inputs):
    layer_inputs.input_arr = input_arr
    res_arr = my_ReLU(input_arr)

    layer_inputs.output_arr = res_arr
   
    return res_arr

def activation_backward_prop(d_output, layer_inputs):
    relu_deriv_input_arr = my_ReLu_deriv(layer_inputs.input_arr)

    res_arr = np.multiply(d_output, relu_deriv_input_arr)

    return res_arr

# ---------------------------------- pooling layer ------------------------------------------
def get_single_max_pool(single_arr, layer_inputs, output_dim):
    new_h = output_dim
    new_w = output_dim
    output_arr = np.zeros((new_h, new_w))

    sub_arr_list = get_subarray_list(single_arr, layer_inputs.filter_dim, layer_inputs.stride)

    x = 0
    for i in range(output_arr.shape[0]):
        for j in range(output_arr.shape[1]):
            sub_arr = sub_arr_list[x]
            output_arr[i, j] = np.max(sub_arr)

            x = x + 1

    return output_arr

def pool_forward_prop(batch_input_arr, layer_inputs):
    layer_inputs.input_arr = batch_input_arr

    output_dim = get_new_dim(batch_input_arr.shape[1], layer_inputs.filter_dim, layer_inputs.padding, layer_inputs.stride)
    new_h = output_dim
    new_w = output_dim
    feature_map_count  = batch_input_arr.shape[3]

    sample_count = batch_input_arr.shape[0]
    output_arr = np.zeros((sample_count, new_h, new_w, feature_map_count))

    for m in range(len(batch_input_arr)):
        input_arr = batch_input_arr[m]

        feature_map_count  = input_arr.shape[2]

        for j in range(feature_map_count):
            sub_input = input_arr[:,:,j]
            output_arr[m,:,:,j] = get_single_max_pool(sub_input, layer_inputs, output_dim)

    layer_inputs.output_arr = output_arr

    return output_arr

def set_gradient_single_max_pool(single_arr, layer_inputs, forward_arr):
    filter_dim = layer_inputs.filter_dim

    output_arr = np.zeros((forward_arr.shape[0], forward_arr.shape[0]))

    i = 0
    u = 0
    iter_range = forward_arr.shape[0] - filter_dim + 1

    while i < iter_range:
        j = 0
        v = 0
        while j < iter_range:
            sub_forward_arr = forward_arr[i:(i+filter_dim), j:(j+filter_dim)]

            max_entry = sub_forward_arr.max()

            idx_max = np.where(sub_forward_arr == max_entry)
            (x , y) = (idx_max[0][0], idx_max[1][0])

            x = i + x
            y = j + y

            output_arr[x,y] += single_arr[u,v]

            j = j + layer_inputs.stride
            v = v + 1
        
        i = i + layer_inputs.stride
        u = u + 1

    return output_arr

def pool_backward_prop(batch_d_output, layer_inputs):
    batch_d_input = np.zeros(layer_inputs.input_arr.shape)

    feature_map_count  = layer_inputs.input_arr.shape[3]

    for m in range(len(batch_d_output)):
        d_output = batch_d_output[m]
        input_arr = layer_inputs.input_arr[m]

        for j in range(feature_map_count):
            sub_d_output = d_output[:,:,j]
            sub_input = input_arr[:,:,j]

            batch_d_input[m,:,:,j] = set_gradient_single_max_pool(sub_d_output, layer_inputs, sub_input)
    

    return batch_d_input

# ---------------------------------- flattening layer ------------------------------------------
def flat_forward_prop(batch_input_arr, layer_inputs):
    # print(batch_input_arr.shape)
    layer_inputs.input_arr = batch_input_arr

    sample_count = batch_input_arr.shape[0]
    output_dim = np.prod(batch_input_arr[0].shape)
    # output_dim = batch_input_arr.shape[1] * batch_input_arr.shape[2] * batch_input_arr.shape[3]
    batch_flat_arr = np.zeros((sample_count, output_dim))

    for m in range(len(batch_input_arr)):
        input_arr = batch_input_arr[m]

        input_arr = input_arr.flatten()
        # input_arr = input_arr.reshape(input_arr.shape[0], 1)

        batch_flat_arr[m, :] = input_arr

    layer_inputs.output_arr = batch_flat_arr
    return batch_flat_arr

def flat_backward_prop(batch_d_output, layer_inputs):
    batch_res_arr = np.zeros(layer_inputs.input_arr.shape)
    
    for m in range(len(batch_d_output)):
        d_output = batch_d_output[m]
        input_arr = layer_inputs.input_arr[m]
        res_arr = d_output.reshape(input_arr.shape)

        # batch_res_arr[m,:,:,:] = res_arr
        batch_res_arr[m] = res_arr

    return batch_res_arr

# ---------------------------------- fc layer ------------------------------------------
def fc_forward_prop(batch_input_arr, layer_inputs, layer_no): # input arr must be a column array
    cur_filter = layer_inputs.cur_filter
    cur_bias = layer_inputs.cur_bias

    batch_z_arr = np.add(np.dot(batch_input_arr, cur_filter.T), cur_bias.T)

    layer_inputs.output_arr = batch_z_arr
    
    return batch_z_arr


def fc_backward_prop(batch_d_output, layer_inputs, learning_rate, layer_no):
    batch_flat_arr = layer_inputs.flat_input_arr

    d_filter = np.dot(batch_d_output.T, batch_flat_arr)
    d_bias = np.average(batch_d_output, axis = 0).T
    d_input = np.dot(batch_d_output, layer_inputs.cur_filter)
    
    layer_inputs.d_filter = d_filter
    layer_inputs.d_bias = d_bias
    layer_inputs.d_input = d_input

    return d_filter, d_bias, d_input

# ---------------------------------- softmax layer ------------------------------------------
def softmax_forward_prop(batch_input_arr, layer_inputs):
    input_arr_2d = batch_input_arr.reshape(batch_input_arr.shape[0], batch_input_arr.shape[1])

    input_arr_2d = input_arr_2d - np.max(input_arr_2d)

    exp_arr = np.exp(input_arr_2d)

    arr_sum = np.sum(exp_arr, axis = 1)
    exp_arr = (exp_arr.T / arr_sum).T

    layer_inputs.output_arr = exp_arr
    return exp_arr

def get_one_hot_y(true_y, a_max):
    a = true_y.flatten()

    b = np.zeros((a.size, a_max)) 
    # print(b.shape)
    b[np.arange(a.size), a] = 1 
    one_hot_arr = b

    return one_hot_arr

def softmax_backward_prop(true_y, y_hat, layer_inputs):
    
    batch_one_hot_arr = np.zeros(y_hat.shape)

    a_max = y_hat.shape[1]
    for m in range(len(true_y)):
        single_true_y = true_y[m]
        one_hot_arr = get_one_hot_y(single_true_y, a_max)
        batch_one_hot_arr[m, :] = one_hot_arr

    res_arr = y_hat - batch_one_hot_arr

    return res_arr



def my_forward(input_arr, layer_inputs, iter_no, layer_no):
    output_arr = np.array([])

    if layer_inputs.layer_name == "Conv":
        if iter_no == 0 and layer_inputs.set_conv_init_param == 0:
            factor = 0.001
            feature_map_count = input_arr.shape[3]
            cur_filter = factor * np.random.randn(layer_inputs.filter_dim, layer_inputs.filter_dim, feature_map_count, layer_inputs.output_channel_count)  
            cur_bias = factor * np.random.randn(layer_inputs.output_channel_count)

            layer_inputs.cur_filter = cur_filter
            layer_inputs.cur_bias = cur_bias

            layer_inputs.set_conv_init_param = 1

        output_arr = conv_forward_prop(input_arr, layer_inputs, layer_no)
        # print(layer_inputs.layer_name)
        # print(input_arr.shape)
        # print(output_arr.shape)

    elif layer_inputs.layer_name == "ReLU":
        output_arr = activation_forward_prop(input_arr, layer_inputs)
        # print(layer_inputs.layer_name)
        # print(input_arr.shape)
        # print(output_arr.shape)

    elif layer_inputs.layer_name == "Pool":
        output_arr = pool_forward_prop(input_arr, layer_inputs)
        # print(layer_inputs.layer_name)
        # print(input_arr.shape)
        # print(output_arr.shape)

    elif layer_inputs.layer_name == "FC":
        layer_inputs.input_arr = input_arr
        # print(layer_inputs.layer_name)
        # print(input_arr.shape)

        flat_layer_obj = flat_layer()
        flat_layer_obj.layer_name = "Flatten"
        flat_layer_obj.input_shape = input_arr.shape
        flat_layer_obj.input_arr = input_arr

        input_arr = flat_forward_prop(input_arr, flat_layer_obj)
        layer_inputs.flat_input_arr = input_arr
    
        # print("in fc")
        if iter_no == 0 and layer_inputs.set_fc_init_param == 0:
            factor = 0.001
            sample_count = input_arr.shape[0]
            cur_filter = factor * np.random.randn(layer_inputs.output_dim, input_arr.shape[1])
            cur_filter = cur_filter / input_arr.shape[1]
            cur_bias = factor * np.random.randn(layer_inputs.output_dim)

            layer_inputs.cur_filter = cur_filter
            layer_inputs.cur_bias = cur_bias

            layer_inputs.set_fc_init_param = 1

        output_arr = fc_forward_prop(input_arr, layer_inputs, layer_no)
        # print(layer_inputs.layer_name)
        # # print(input_arr.shape)
        # print(output_arr.shape)

    elif layer_inputs.layer_name == "Flatten":

        output_arr = flat_forward_prop(input_arr, layer_inputs)

    elif layer_inputs.layer_name == "Softmax":
        output_arr = softmax_forward_prop(input_arr, layer_inputs)

    return output_arr

def my_backward_prop(true_y, d_output, layer_inputs, learning_rate, iter_no, layer_no):
    d_input = np.array([])

    if layer_inputs.layer_name == "Conv":
        d_filter, d_bias, d_input = conv_backward_prop(d_output, layer_inputs, learning_rate, layer_no)
        # print(layer_inputs.layer_name)
        # print(d_output.shape)
        # print(d_input.shape)

    elif layer_inputs.layer_name == "ReLU":
        d_input = activation_backward_prop(d_output, layer_inputs)
        # print(layer_inputs.layer_name)
        # print(d_output.shape)
        # print(d_input.shape)

    elif layer_inputs.layer_name == "Pool":
        d_input = pool_backward_prop(d_output, layer_inputs)
        # print(layer_inputs.layer_name)
        # print(d_output.shape)
        # print(d_input.shape)

    elif layer_inputs.layer_name == "FC":
        d_filter, d_bias, d_input = fc_backward_prop(d_output, layer_inputs, learning_rate, layer_no)
        # print(layer_inputs.layer_name)
        # print(d_output.shape)
        # print(d_input.shape)
        flat_layer_obj = flat_layer()
        flat_layer_obj.layer_name = "Flatten"
        flat_layer_obj.input_arr = layer_inputs.input_arr
        # print(layer_inputs.input_arr.shape)

        d_input = flat_backward_prop(d_input, flat_layer_obj)
        # print(d_input.shape)

    elif layer_inputs.layer_name == "Flatten":
        # print(layer_inputs.layer_name)

        d_input = flat_backward_prop(d_output, layer_inputs)

    elif layer_inputs.layer_name == "Softmax":
        # print("in softmax")
        y_hat = d_output
        d_input = softmax_backward_prop(true_y, y_hat, layer_inputs)
        # print(layer_inputs.layer_name)
        # print(d_output.shape)
        # print(d_input.shape)

    return d_input



def my_train(train_X, train_y, archi_list, iter_no, subset_no):
    for i in range(len(archi_list)):
        layer_inputs = archi_list[i]

        if layer_inputs.layer_name == "Conv":
            layer_inputs.set_conv_init_param = 0
        elif layer_inputs.layer_name == "FC":
            layer_inputs.set_fc_init_param = 0

        output_arr  = my_forward(train_X, layer_inputs, iter_no, i) 
        train_X = output_arr
        # setting updated layer info
        archi_list[i] = layer_inputs

    d_output = output_arr
    for i in range(len(archi_list)-1, -1, -1):
        layer_inputs = archi_list[i]

        learning_rate = 0.001

        # if i >= 7:
        output_arr = my_backward_prop(train_y, d_output, layer_inputs, learning_rate, iter_no, i)
        d_output = output_arr 

        # setting updated layer info
        archi_list[i] = layer_inputs


    #update params
    for i in range(len(archi_list)):
        layer_inputs = archi_list[i]

        learning_rate = gb_var_obj.learning_rate

        if layer_inputs.layer_name == "Conv" or layer_inputs.layer_name == "FC":
            # print("iter_no: " + str(iter_no) + "\t batch no: " + str(subset_no))
            layer_inputs.cur_filter = layer_inputs.cur_filter - learning_rate * layer_inputs.d_filter
            layer_inputs.cur_bias = layer_inputs.cur_bias - learning_rate * layer_inputs.d_bias

        # setting updated layer info
        archi_list[i] = layer_inputs

    # print("batch: " + str(subset_no) + "\t iter no: " + str(iter_no))

    return archi_list

def get_cross_entropy_loss(softmax_y_hat, one_hot_arr):
    softmax_y_hat[softmax_y_hat<=0] = 1
    ln_y_hat = np.log(softmax_y_hat)
    loss = 0
    loss = - np.multiply(ln_y_hat, one_hot_arr)
    loss = np.sum(loss)

    return loss

def calculate_loss(valid_y, output_arr):
    a_max = output_arr.shape[1]
    # batch_one_hot_arr = np.zeros((len(valid_y), a_max))

    loss = 0
    for m in range(len(valid_y)):
        single_true_y = valid_y[m]
        one_hot_arr = get_one_hot_y(single_true_y, a_max)
        # batch_one_hot_arr[m, :] = one_hot_arr      

        loss += get_cross_entropy_loss(output_arr[m], one_hot_arr)
        
    validation_loss = loss / len(valid_y)

    return validation_loss

def calculate_loss_2(valid_y, output_arr):
    validation_loss = log_loss(valid_y,  output_arr)
    return validation_loss


def my_validation(valid_X, valid_y, archi_list, iter_no, subset_no, true_labels):
    total_entropy_loss = 0

    pred_y_arr = np.zeros((valid_y.shape[0], 1))

    for i in range(len(archi_list)):
        layer_inputs = archi_list[i]
        output_arr = my_forward(valid_X, layer_inputs, iter_no, i)
        valid_X = output_arr

    
    pred_y = np.argmax(output_arr, axis = 1)
    valid_y = valid_y.flatten()
    
    validation_loss = calculate_loss(valid_y, output_arr)
    # validation_loss = calculate_batch_loss(valid_y, output_arr)
    accuracy = accuracy_score(valid_y, pred_y)
    macro_f1 = f1_score(valid_y, pred_y, average="macro")

    return validation_loss, accuracy, macro_f1

    
def my_main():
    dataset_option = int(input("Enter your value: "))
    train_X, train_y, test_X, test_y = np.array([]), np.array([]), np.array([]), np.array([])
    if dataset_option == 1:
        train_X, train_y, test_X, test_y = load_mnist_dataset()
    elif dataset_option == 2:
        train_X, train_y, test_X, test_y = load_cifar10_dataset()

    # splitting validation and test set
    limit = int(test_X.shape[0] / 2)

    valid_X = test_X[:limit]
    valid_y = test_y[:limit]

    test_X = test_X[limit:]
    test_y = test_y[limit:]
    
    # shuffling
    train_index_array = np.arange(train_X.shape[0])
    sample_train_index = np.random.choice(train_index_array, train_X.shape[0], replace=False)

    train_X = train_X[sample_train_index, :, :, :]
    train_y = train_y[sample_train_index, : ]

    valid_index_array = np.arange(valid_X.shape[0])
    sample_valid_index = np.random.choice(valid_index_array, valid_X.shape[0], replace=False)

    valid_X = valid_X[sample_valid_index, :, :, :]
    valid_y = valid_y[sample_valid_index, :]

    train_X = train_X[:5000]
    train_y = train_y[:5000]

    valid_X = valid_X[:500]
    valid_y = valid_y[:500]

    test_X = test_X[:500]
    test_y = test_y[:500]

    total_sample_count = train_X.shape[0]

    train_subset_count = int(train_X.shape[0] / gb_var_obj.sample_count)

    train_X = train_X.astype("float32") / 255
    valid_X = valid_X.astype("float32") / 255
    test_X = test_X.astype("float32") / 255

    archi_list = get_input()

    for iter_no in range(gb_var_obj.iter_count):
        print("start training iter_no: " + str(iter_no))
        
        for i in range(train_subset_count):
            factor = gb_var_obj.sample_count
            sub_train_X = train_X[i*factor : (i+1)*factor]
            sub_train_y = train_y[i*factor : (i+1)*factor]

            # if i == 0:
            updated_archi_list = my_train(sub_train_X, sub_train_y, archi_list, iter_no, i)
            archi_list = updated_archi_list

        print("start validation: " + str(iter_no))
        subset_no = 0
        true_labels = np.arange(gb_var_obj.class_count)
        validation_loss, accuracy, macro_f1 = my_validation(valid_X, valid_y, archi_list, iter_no, subset_no, true_labels)
        print("validation_loss: " + str(validation_loss))
        print("accuracy: " + str(accuracy))
        print("macro_f1: " + str(macro_f1))
        # f2.write(x_list[i] + "\n")
        f2.write("\nstart validation: " + str(iter))
        f2.write("\nvalidation_loss: " + str(validation_loss))
        f2.write("\naccuracy: " + str(accuracy))
        f2.write("\nmacro_f1: " + str(macro_f1))
        f2.write("\n-------------------\n")

    print("\n\nstart testing: ")
    true_labels = np.arange(gb_var_obj.class_count)
    subset_no = 0
    test_loss, accuracy, macro_f1 = my_validation(test_X, test_y, archi_list, iter_no, subset_no, true_labels)
    print("validation_loss: " + str(test_loss))
    print("accuracy: " + str(accuracy))
    print("macro_f1: " + str(macro_f1))
    # f2.write(x_list[i] + "\n")
    f2.write("\nstart testing: " + str(iter))
    f2.write("\ntest_loss: " + str(test_loss))
    f2.write("\naccuracy: " + str(accuracy))
    f2.write("\nmacro_f1: " + str(macro_f1))
    f2.write("\n-------------------\n")


def toy_dataset_task():
    toy_train_arr = np.loadtxt(gb_var_obj.toy_train_filename)
    toy_test_arr = np.loadtxt(gb_var_obj.toy_test_filename)
    total_col = toy_train_arr.shape[1]
    toy_train_X = toy_train_arr[:, : total_col - 1]
    toy_train_y = toy_train_arr[:, total_col - 1]
    toy_test_X = toy_test_arr[:, : total_col - 1]
    toy_test_y = toy_test_arr[:, total_col - 1]

    # y = x.astype(np.float)
    toy_train_X = toy_train_X.astype(np.float)
    toy_test_X = toy_test_X.astype(np.float)
    toy_train_y = toy_train_y.astype(np.int)
    toy_test_y = toy_test_y.astype(np.int)

    toy_train_y = toy_train_y - 1
    toy_test_y = toy_test_y - 1

    toy_train_X = toy_train_X.reshape(toy_train_X.shape[0], toy_train_X.shape[1], 1, 1)
    toy_test_X = toy_test_X.reshape(toy_test_X.shape[0], toy_test_X.shape[1], 1, 1)

    limit = int(toy_test_X.shape[0] / 2)
    valid_X = toy_test_X[:limit]
    valid_y = toy_test_y[:limit]

    test_X = toy_test_X[limit:]
    test_y = toy_test_y[limit:]

    train_subset_count = int(toy_train_X.shape[0] / gb_var_obj.sample_count)

    toy_train_X = toy_train_X.astype("float32") / 255
    valid_X = valid_X.astype("float32") / 255
    test_X = test_X.astype("float32") / 255

    archi_list = get_toy_input()

    gb_var_obj.learning_rate = 0.1

    for iter_no in range(gb_var_obj.toy_iter_count):
        print("start training iter_no: " + str(iter_no))

        for i in range(train_subset_count):
            factor = gb_var_obj.sample_count
            sub_train_X = toy_train_X[i*factor : (i+1)*factor]
            sub_train_y = toy_train_y[i*factor : (i+1)*factor]

            # if i == 0:
            updated_archi_list = my_train(sub_train_X, sub_train_y, archi_list, iter_no, i)
            archi_list = updated_archi_list

        print("start validation: " + str(iter_no))
        true_labels = np.arange(gb_var_obj.toy_class_count) + 1
        validation_loss, accuracy, macro_f1 = my_validation(valid_X, valid_y, archi_list, iter_no, i, true_labels)
        print("validation_loss: " + str(validation_loss))
        print("accuracy: " + str(accuracy))
        print("macro_f1: " + str(macro_f1))
        # f2.write(x_list[i] + "\n")
        f2.write("\nstart validation: " + str(iter))
        f2.write("\nvalidation_loss: " + str(validation_loss))
        f2.write("\naccuracy: " + str(accuracy))
        f2.write("\nmacro_f1: " + str(macro_f1))
        f2.write("\n-------------------\n")

    print("\n\nstart testing: ")
    true_labels = np.arange(gb_var_obj.toy_class_count) + 1
    test_loss, accuracy, macro_f1 = my_validation(test_X, test_y, archi_list, iter_no, i, true_labels)
    print("validation_loss: " + str(test_loss))
    print("accuracy: " + str(accuracy))
    print("macro_f1: " + str(macro_f1))
    # f2.write(x_list[i] + "\n")
    f2.write("\nstart testing: " + str(iter))
    f2.write("\ntest_loss: " + str(test_loss))
    f2.write("\naccuracy: " + str(accuracy))
    f2.write("\nmacro_f1: " + str(macro_f1))
    f2.write("\n-------------------\n")


def wrap_main_func():
    dataset_option = int(input("Toy dataset?: "))
    if dataset_option == 1:
        toy_dataset_task()
    else:
        my_main()


# def check_back_main():
#     pass


my_main()
# toy_dataset_task()
# check_back_main()
# wrap_main_func()