# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-11-02 01:11:00
# @Last Modified by:   vamshi
# @Last Modified time: 2018-11-30 03:34:54
from six.moves import urllib
import cv2
import numpy as np
import os

def store_raw_images():
    # neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04285008'   
    # neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

    f= open('dataset3_urls','r')
    neg_image_urls=f.read()
    pic_num = 1234
    
    if not os.path.exists('neg'):
        os.makedirs('neg')
        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (192, 108))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))  

def find_uglies():
    match = False
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))

def create_neg():
    for file_type in ['neg']:
        
        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
def resize_template():
    im=cv2.imread('template3.jpeg',0)
    resized_im=cv2.resize(im,(50,50))
    cv2.imwrite('template3.jpeg',resized_im)

def combine_datasets():
    pic_num=662
    pwd= "/media/vamshi/Education/wpi/cv_project/neg"
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            im=cv2.imread('neg/'+img,0)
            # print "here",im
            # cv2.imshow('image',im)
            cv2.imwrite('neg/'+str(pic_num)+'.jpg',im)

            # os.rename(pwd+img,pwd+str(pic_num)+".jpg")
            pic_num+=1


# store_raw_images()
# find_uglies()
# create_neg() 
# resize_template()
# combine_datasets()

