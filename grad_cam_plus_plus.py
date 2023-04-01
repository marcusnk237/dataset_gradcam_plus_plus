import cv2
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from matplotlib import cm
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib as mpl
mpl.style.use('seaborn')
import itertools
import logging
# Set random seed
np.random.seed(123)
from skimage.transform import resize

"""
A function to compute GRAD-CAM ++
Arguments :
- model: The model trained
- signal: The data sample
- layer_name : The last layer of the feature extraction part of the model. Usually, it is the last layer before the Flattening operation. 
"""
def compute_cam(model, signal, layer_name , eps=1e-8):
        grad_model = tf.keras.models.Model(inputs=[model.inputs],
                                           outputs=[model.get_layer(layer_name).output, model.output])     
       
        with tf.GradientTape() as tape:
            inputs = np.expand_dims(signal,axis=0)
            conv_outs, predictions = grad_model(inputs) 
            class_idx = tf.argmax(predictions[0])
            y_c = predictions[:, class_idx]

        # compute the gradient of the score for the class c, with respect to feature maps Ak of a convolutional layer
        batch_grads = tape.gradient(y_c, conv_outs) 
        grads = batch_grads[0]
        first = tf.exp(y_c) * grads
        second = tf.exp(y_c) * tf.pow(grads, 2)
        third = tf.exp(y_c) * tf.pow(grads, 3)
        global_sum = tf.reduce_sum(tf.reshape(conv_outs[0], shape=(-1, first.shape[1])), axis=0)
        alpha_num = second
        alpha_denom = second * 2.0 + third * tf.reshape(global_sum, shape=(1,1,first.shape[1]))
        alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones(shape=alpha_denom.shape))
        alphas = alpha_num / alpha_denom
        weights = tf.maximum(first, 0.0)
        alpha_normalization_constant = tf.reduce_sum(tf.reduce_sum(alphas, axis=0), axis=0)
        alphas /= tf.reshape(alpha_normalization_constant, shape=(1,1,first.shape[1]))
        alphas_thresholding = np.where(weights, alphas, 0.0)

        alpha_normalization_constant = tf.reduce_sum(tf.reduce_sum(alphas_thresholding, axis=0),axis=0)
        alpha_normalization_constant_processed = tf.where(alpha_normalization_constant != 0.0, alpha_normalization_constant,
                                                          tf.ones(alpha_normalization_constant.shape))

        alphas /= tf.reshape(alpha_normalization_constant_processed, shape=(1,1,first.shape[1]))
        deep_linearization_weights = tf.reduce_sum(tf.reshape((weights*alphas), shape=(-1,first.shape[1])), axis=0)
        grad_CAM_map = tf.reduce_sum(deep_linearization_weights * conv_outs[0], axis=-1)
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  
        return cam
"""
Function to plot all the GRAD-CAM++ segments.
Arguments:
- y : The data sample
- x : The time range. The user can create a linspace vector and use it as the time range.
- heatmap : The GRAD-CAM ++ gradient values
"""
def multicolored_lines(x,y,heatmap,title_name):
    fig, ax = plt.subplots()
    lc = colorline(x, y, heatmap,cmap='rainbow')
    clb = plt.colorbar(lc)
    clb.ax.tick_params(labelsize=12) 
    clb.ax.set_title('Gradient value',fontsize=12)
    lc.set_linewidth(2)
    lc.set_alpha(0.8)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.title(title_name)
    plt.grid(False)
    plt.show()
"""
Function to color the signal segment depending of the GRAD-CAM ++ gradient value.
Arguments:
- signal : The data sample
- time : The time range. The user can create a linspace vector and use it as the time range.
- heatmap : The GRAD-CAM ++ gradient values
"""
def colorline(time, signal, heatmap,cmap='rainbow'):
    z = np.array(heatmap)
    points = np.array([time, signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc
"""
A function to plot the dataset-level feature importance.
Arguments :
- values: feature importance frequency
- labels : feature labels
"""

def plot_dataset_features_importances(values,labels):
    y = np.array(values)
    indexes = np.arange(len(labels))
    colors = cm.rainbow(y / float(max(y)))
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values,color= colors)
    ax.bar_label(bars)
    plt.ylabel('Dataset-level classification importance (%)')
    plt.xlabel('Signal features')
    
""""
Make a local classification explanation.
Arguments:
- model : The model trained
- data : The data sample
- time : The time range. The user can create a linspace vector and use it as the time range.
- layer_name : The last layer of the feature extraction part of the model. Usually, it is the last layer before the Flattening operation.
- label : The label output list
"""
def local_features_importances(model,data,time,layer_name,label):
    pred = model.predict(np.expand_dims(data,axis=0))
    index=np.argmax(pred[0])
    plt.style.use("seaborn-whitegrid")
    big_heatmap = cv2.resize(np.array([compute_cam(model, data , layer_name, eps=1e-8).tolist()]), dsize=(data.shape[0], 300),interpolation=cv2.INTER_CUBIC)
    x = np.linspace(0, time, data.shape[0])
    multicolored_lines(x,np.array([i[0] for i in data]),big_heatmap[0],f"Model prediction = {label[index]} ({round(pred[0][index]*100)})")

""""
Make a dataset-level classification explanation.
Arguments:
- model : The model trained
- datas : the dataset. Due of Hardware limitations, the max size of datas is 5000 samples.
- layer_name : The last layer of the feature extraction part of the model. Usually, it is the last layer before the Flattening operation.
- feature_names : The feature list
"""
def dataset_features_importances(model,datas,layer_name,feature_names,flag=True):
    grads= pd.DataFrame(np.array([compute_cam(model, i , layer_name, eps=1e-8).tolist()  for i in datas])).dropna()
    max_index=[grads.values.tolist()[i].index(max(grads.values.tolist()[i])) for i in range(len(grads.values.tolist()))]
    features_importances = Counter(max_index).most_common()
    labels, values = [features_importances[i][0] for i in range(len(features_importances[:][0]))],[features_importances[i][1] for i in range(len(features_importances[:][1]))]
    if flag ==False:
        try:
            return [(100/sum(values))*float(i) for i in values], [feature_names[i] for i in labels]
        except:
            return [(100/sum(values))*float(i) for i in values], labels
    if flag == True:
        try:
            plot_dataset_features_importances(values,feature_names)
        except:
            plot_dataset_features_importances(values,labels)


