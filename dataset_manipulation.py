#Author:Silvio Severino
#Date:30/12/18

import shutil
import os
import cv2
from imutils import paths
import random

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import img_to_array



def split_test_into_subfolders(path):
    
    dest = 'sc5-Test-tensorflow'
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    with open(path) as file:
        
        for line in file:
            file_name, class_name = line.split(';')
            class_name = class_name.replace('\n','')
            
            subdest = dest + '/' + class_name
            if not os.path.exists(subdest):
                os.makedirs(subdest)
            shutil.copy('sc5-Test/'+file_name, subdest)
            
        file.close()


def split_test_into_family_class(path):
    
    dest = 'sc5-test-tensorflow'
    start = 'sc5-Test-tensorflow-tmp'
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    with open(path) as file:
        
        for line in file:
            file_name, class_name = line.split(';')
            class_name = class_name.replace('\n','')
            
            if os.path.exists(start+'/'+file_name):
                subdest = dest + '/' + class_name
                
                if not os.path.exists(subdest):
                    os.makedirs(subdest)
                shutil.move(start+'/'+file_name,subdest)
        file.close()  



def split_into_family_class(path):
    
    dest = 'sc5-tensorflow'
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    with open(path) as file:
        
        for line in file:
            file_name, class_name = line.split(';')
            class_name = class_name.replace('\n','')
            
            if os.path.exists('sc5/'+file_name):
                subdest = dest + '/' + class_name
                
                if not os.path.exists(subdest):
                    os.makedirs(subdest)
                shutil.move('sc5/'+file_name,subdest)
        file.close()  



def balance_train_and_test(source_train, source_test):
    
    for family in os.listdir(source_train):
        
        for classes in os.listdir(source_train+'/'+family):
            
            subfolder_train = source_train + '/' + family + '/' + classes
            subfolder_test = source_test + '/' + family + '/' + classes
            
            if not os.path.exists(subfolder_test):
                print(classes)
                os.mkdir(subfolder_test)
                pics_to_move = int(0.2*len(os.listdir(source_train + '/' + family + '/' + classes)))
                images = os.listdir(subfolder_train)
                
                for i in range(pics_to_move):
                    image = subfolder_train + '/' + images[i]
                    shutil.move(image, subfolder_test)



#balance_train_and_test('sc5-tensorflow','sc5-test-tensorflow')
#split_into_subfolders('sc5-Test/ground_truth.txt')
#split_into_family_class('sc5/ground-truth-family.txt')
#split_test_into_family_class('sc5-Test/ground-truth-family.txt')


#depth:
# -1 image directory
# -2 boats classification
# -3 family classification
# -4 source path
def read_images(path, depth, height, width):
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (height, width))
        image = img_to_array(image)
        data.append(image)

        label = imagePath.split(os.path.sep)[-depth]
        labels.append(label)
    return data, labels



#a, b = read_images("sc5-tensorflow", 2, height, width)
#b.shape

