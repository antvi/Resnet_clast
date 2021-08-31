import cv2
import numpy as np
import shutil
from keras import Sequential
from keras.applications import ResNet50
from keras_applications.densenet import preprocess_input
import os

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans



def extract_vector(path): #функция 
    resnet_feature_list = [] #объявляем пустой массив
    files = os.listdir(path)#получаем список файлов в директории os.listdir(path=".") - список файлов и директорий в папке.
    tmp_mas = [] #объявляем еще пустой массив
    for im_name in files: #создаем цикл от 
        im = cv2.imread(path + '/' + im_name) 
        im = cv2.resize(im,(224,224))
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = my_new_model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())
        tmp_mas.append(im_name)
    print(tmp_mas)
    return np.array(resnet_feature_list), tmp_mas

