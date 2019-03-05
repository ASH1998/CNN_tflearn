from tqdm import tqdm
import os
import cv2
import numpy as np
from random import shuffle

TRAIN_DIR = "E:\Python Coding\CNN dogs and cats\\traindogcat\\train"
TEST_DIR = "E:\Python Coding\CNN dogs and cats\\test1dogcat\\test1"

IMG_SIZE = 64
LR = 1e-03

MODEL_NAME = 'dogcatclf'

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]

def create_train_data():
    train = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        train.append([np.array(img), np.array(label)])
    shuffle(train)
    np.save('train.npy', train)
    return train

def create_test_data():
    test = []
    for img in tqdm(os.listdir(TEST_DIR)):
        img_num = img.split('.')[0]
        path = os.path.join(TEST_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        test.append([np.array(img), img_num])
    np.save('test.npy', test)
    return test

#train_data = create_train_data()
#test_data = create_test_data()