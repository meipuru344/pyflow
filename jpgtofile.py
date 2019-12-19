# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from PIL import Image
import glob
import time
import cv2
import argparse
import pyflow


start = time.time()


# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0



img_array_list = []
for k in range(1):
    #fileNum = len(glob.glob('/data/pictures/'+ str(k+1) +'/*'))
    for i in range(100):#fileNum-2):
        formatch = '{:0>6}'.format(i+1)
        formatchm = '{:0>6}'.format(i+2)
        formatchn = '{:0>6}'.format(i+3)
        #strf = 'data/pictures/'+ str(k+1) + '/' + formatch + '.jpg'
        ##strm = 'data/pictures/'+ str(k+1) + '/' + formatchm + '.jpg'
        #strn = 'data/pictures/'+ str(k+1) + '/' + formatchn + '.jpg'
        #img_array_list.append((strf,strn,strm))
        #print(strf)#im = np.array(Image.open(strf))
        #image_bgr = cv2.imread(strf)
        #im = cv2.cvtColor(cv2.imread(strf), cv2.COLOR_BGR2RGB)#imadd = np.concatenate([imadd,im],axis=0)

    average_array = []
    max_array = []
    output_array = []
    count = 0
    for img in img_array_list:
        print(img)
        #####縮小#####
        offset = (300, 300, 450, 450)

        #####画像から150x150を切り出す#####
        im1 = Image.open(img[0])
        im1 = im1.resize((150,150),Image.NEAREST, offset)#s^3
        im1 = np.asarray(im1)
        im1 = im1.astype(float) / 255.
        im2 = Image.open(img[1])
        im2 = im2.resize((150,150),Image.NEAREST, offset)
        im2 = np.asarray(im2)
        im2 = im2.astype(float) / 255.
        #####OpticalFlowを計算し,im1,im2に保存、大きさは2x150x150##### im1dtype=numpy range=0~1(圧縮済み)
        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        #####平均計算, opticalflowの評価数値の算出#####
        optical_flow = np.swapaxes(flow,0,2) #2x150x150
        flow1 = optical_flow[0]
        flow2 = optical_flow[1]
        flow_dif = flow1 - flow2
        flow_d = np.sqrt(flow_dif * flow_dif)
        flow_d_sum = np.sum(flow_d)
        flow_d_average = flow_d_sum/flow1.size
        flow_d_max = np.amax(flow_d_average)
        print("average:",flow_d_average)
        print("max:",np.amax(flow_d))
        average_array.append(flow_d_average)
        max_array.append(flow_d_max)
        #####opticalflowの評価#####
        print(np.amax(im1))
        if(flow_d_average > 8 and flow_d_max < 39 and count < 100):
            #####条件を満たすopticalflowの保存#####
            im_mid = Image.open(img[2])
            im_mid = im_mid.resize((150,150),Image.NEAREST, offset)
            im_mid = np.asarray(im_mid)
            im_mid = im_mid.astype(float) / 255.
            count += 1
            output_array.append(np.array([im1,im_mid,im2]))
            print("ave > 8 かつ max < 39", (count+1), "個")
        elif(count>100):
            break
        ##########for終わり、フレーム毎の層
    #####ndarrayを.npyとして保存
    output_array = np.array(output_array)
    output_array = np.swapaxes(output_array, 2, 4)
    np.save('../data/train_data/'+str(k+1)+'.npy',output_array)

###############for終わり、動画毎の層


end = time.time() - start

#####データの表示
sorted_average_array = average_array.copy()
sorted_average_array.sort()
sorted_average_array.reverse()
sort_num_ave = []
for i in range(len(average_array)):
    sort_num_ave.append(average_array.index(sorted_average_array[i]))


sorted_max_array = max_array.copy()
sorted_max_array.sort()
sorted_max_array.reverse()
sort_num_max = []
for i in range(len(max_array)):
    sort_num_max.append(max_array.index(sorted_max_array[i]))


average_array = np.array(sorted_average_array)
max_array = np.array(sorted_max_array)
np.set_printoptions(precision=3)
#print(average_array[:40])
#print(max_array[:40])
##print(sort_num_ave[:40])
#p#rint(sort_num_max[:40])



print(output_array.shape)
print("time:",end)
print("条件を満たすデータは", count, "個ありました。")







#
