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

resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

array, tmp_mas = extract_vector('img/')
kmeans = KMeans(init='random', n_clusters=5, n_init=10).fit(array)
kmeans = DBSCAN(eps=0.6).fit(array)
print(kmeans.labels_)
for i in range(len(kmeans.labels_)):
    if not os.path.exists(str(kmeans.labels_[i])):
        os.mkdir(str(kmeans.labels_[i]))
    shutil.copyfile('img/' + tmp_mas[i], str(kmeans.labels_[i]) + '/' + tmp_mas[i])

