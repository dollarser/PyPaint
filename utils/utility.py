from io import BufferedReader
from matplotlib import colors
from numpy.core.defchararray import index, split
from numpy.lib.function_base import delete
from numpy.lib.histograms import histogram
from random import random

import math
import sys
import time
import json
from skimage.util import dtype
import statsmodels.api as sm

from matplotlib.pyplot import axis, imshow, vlines
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import shutil
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from skimage import exposure
from collections import Counter
from skimage import io
from skimage.filters import gaussian
from scipy import signal


def show_figure(data,data2=None,data3 = None, data4= None,season=None):
    plt.subplot(411)
    plt.plot(data)
    ax = plt.gca()
    # x_major_locator=MultipleLocator(200000)
    # ax.xaxis.set_major_locator(x_major_locator)
    if season is not None:
        for i in range(1,2):
            plt.axvline(i*season)

    if data2 is not None:
        plt.subplot(412)
        plt.plot(data2,color='red')
        if season is not None:
            plt.axvline(int(len(data2)/2))
    if data3 is not None:
        plt.subplot(413)
        plt.plot(data3,color='red')
    if data4 is not None:
        plt.subplot(414)
        plt.plot(data4,color='red')

        #Show full screen
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    plt.show()

def stl(img, direct='col'):
    ''' 请使用灰度图 stl分解'''
    if direct == 'col':
        img = np.transpose(img)
    h,w=img.shape
    # reshape成一维数据训练
    img_flatten = img.reshape(-1)
    # 转换为pandas类型的数据进行分解
    series = pd.Series(img_flatten) 
    #周期为图片的宽
    res = sm.tsa.seasonal_decompose(series, period=w, extrapolate_trend='freq')
    #获取分量 还原图片
    resid = np.array(res.resid).reshape((h,w))
    season = np.array(res.seasonal).reshape((h,w))
    trend = np.array(res.trend).reshape((h,w))
    observed = np.array(res.observed).reshape((h,w))
    if direct == 'col':
        resid = np.transpose(resid)
        season = np.transpose(season)
        trend = np.transpose(trend)
        observed = np.transpose(observed)
    resid = (resid - np.min(resid))*255/(np.max(resid)-np.min(resid))
    season = (season - np.min(season))*255/(np.max(season)-np.min(season))
    trend = (trend - np.min(trend))*255/(np.max(trend)-np.min(trend))
    observed = (observed - np.min(observed))*255/(np.max(observed)-np.min(observed))
    resid = resid.astype(np.uint8)
    season = season.astype(np.uint8)
    trend = trend.astype(np.uint8)
    observed = observed.astype(np.uint8)

    return season, trend, resid, observed

def rescale_intensity(img,thres=3):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    crange = np.arange(256)
    cnum = np.zeros_like(crange)
    for i in crange:
        cnum[i] = np.sum(img == i)
    st = 0
    while(cnum[st]<thres):
        st += 1
    ed = 255
    while(cnum[ed]<thres):
        ed -= 1
    if st < ed:
        img[img<st] = st
        img[img>ed] = ed
        img = (img-st)/(ed-st) * 255
        img = np.asarray(img,dtype=np.uint8)
    print(st,ed)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    return img

def pixel_histogram(img):
    pic = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    crange = np.arange(256)
    cnum = np.zeros_like(crange)
    for i in crange:
        cnum[i] = np.sum(pic == i)
    cnum = cnum / (pic.shape[0]*pic.shape[1])
    return cnum
    
def binary(img, thres = -1):
    img_out = np.array(img)
    if thres == -1:
        thres = np.mean(img_out)
    idx = img_out > thres
    img_out[idx] = 255
    img_out[idx == False] = 0
    return img_out

def binarization(img,win, beta = 0.9):
    image_copy = np.array(img)
    if len(image_copy.shape) == 3:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    if win % 2 == 0: win = win - 1
    # 边界的均值有点麻烦
    # 这里分别计算和 和 邻居数再相除
    kern = np.ones([win, win])
    print(image_copy.shape)
    sums = signal.correlate2d(image_copy, kern, 'same')
    cnts = signal.correlate2d(np.ones_like(image_copy), kern, 'same')
    means = sums // cnts
    # 如果直接采用均值作为阈值，背景会变花
    # 但是相邻背景颜色相差不大
    # 所以乘个系数把它们过滤掉
    image_copy = np.where(image_copy < means * beta, 0, 255)
    image_copy = np.asarray(image_copy,dtype=np.uint8)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
    return image_copy

def adaptive_threshold(img, blockSize=31, brightness_th=5):
    image_copy = np.array(img)
    # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    image_copy = cv2.adaptiveThreshold(image_copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=blockSize, C=brightness_th)
    # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
    return image_copy

def gaussianblur_iten_binary(img,gaus_sigma=2,binary_brightness_th=11,kernal_size=33):
    img_out = np.array(img)
    # 第二个参数，高斯矩阵的长宽，第三个参数，X和Y方向上的高斯核标准差
    img_out = cv2.GaussianBlur(img_out,(33,33),gaus_sigma)
    img_out = exposure.rescale_intensity(img_out)
    img_out = adaptive_threshold(img_out,kernal_size,binary_brightness_th)
    return img_out

def white__balance(img):
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    print(b.shape)
    sum = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    while Y >= 0:
        num += hists[Y]
        if num > m * n * 0.01 / 100:
            key = Y
            break
        Y = Y - 1

    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum[i][j] >= Y:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1

    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    for i in range(m):
        for j in range(n):
            b[i][j] = b[i][j] * 255 / avg_b
            g[i][j] = g[i][j] * 255 / avg_g
            r[i][j] = r[i][j] * 255 / avg_r
            if b[i][j] > 255:
                b[i][j] = 255
            if b[i][j] < 0:
                b[i][j] = 0
            if g[i][j] > 255:
                g[i][j] = 255
            if g[i][j] < 0:
                g[i][j] = 0
            if r[i][j] > 255:
                r[i][j] = 255
            if r[i][j] < 0:
                r[i][j] = 0

    img_0 = cv2.merge([b, g, r])
    return img_0

def best_f1(y_test,score):
    min_s = np.min(score) + 1e-5
    max_s = np.max(score) - 1e-5
    thres = np.linspace(min_s,max_s,100)
    best_f1 = 0
    best_th = 0
    for th in thres:
        predict = np.zeros(len(y_test),dtype=np.int)
        predict[score >= th] = 1
        f1 = f1_score(y_test,predict,labels=1)
        if f1 > best_f1:
            # print('best_f1: ',f1,' th: ',th)
            best_f1 = f1
            best_th = th
    predict = np.zeros(len(y_test),dtype=np.int)
    predict[score >= best_th] = 1
    print("best_f1: ",best_f1,' th: ',round(best_th,4),' pre: ',round(precision_score(y_test,predict,labels=1),4),
                ' rec: ',round(recall_score(y_test,predict,labels=1),4))
    return best_th, best_f1

def best_rec(y_test,score):
    min_s = np.min(score) + 1e-5
    max_s = np.max(score) - 1e-5
    thres = np.linspace(min_s,max_s,100)
    thres = thres[::-1]
    best_rec = 0
    n_anom = math.floor(np.sum(y_test))
    idxs = np.argsort(score[:n_anom])
    idx90 = idxs[math.floor(n_anom*0.1 + 1)]
    idx95 = idxs[math.floor(n_anom*0.05 + 1)]
    idx99 = idxs[math.floor(n_anom*0.01 + 1)]
    thres = [score[idx90],score[idx95],score[idx99]]
    for th in thres:
        predict = np.zeros(len(y_test),dtype=np.int)
        predict[score >= th] = 1
        rec = recall_score(y_test,predict,labels = 1)
        print('best_rec: ',round(rec,4),' th: ',round(th,4),
            ' pre: ',round(precision_score(y_test,predict,labels=1),4),' f1: ',round(f1_score(y_test,predict,labels=1),4))
        best_rec = rec
    return best_rec


def rm_HLine(img,line_th=0.9,expand=4,fill='none'):
    # num是up_row与down_row之间相差的行数，num  = idx_up_row - idx_down_row - 1
    def _tool(up_row,down_row,num):
        up_row = up_row.astype('float32')
        down_row = down_row.astype('float32')
        res = np.zeros((num,len(up_row),3),dtype='float32')
        space = (down_row - up_row) / (num+1)
        for i in range(num):
            res[i] = up_row + (i+1)*space
        return np.asarray(res,dtype=np.uint8)
    pic = np.array(img)
    img_out = np.array(img)
    pic = gaussianblur_iten_binary(pic,gaus_sigma=2,binary_brightness_th=2)
    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('e',img)
    # cv2.imshow('c',pic)
    pic_flag = np.zeros_like(pic)
    pic_mean = np.mean(pic)
    pic_flag[pic < pic_mean] = 1
    pic_flag = np.sum(pic_flag,axis=1)

    row_idxs = np.arange(pic.shape[0])
    pixel_num = math.ceil(pic.shape[1] * line_th)
    row_idxs = row_idxs[pic_flag > pixel_num]
    fi_row_idxs = np.array([],dtype=np.int)
    for i in range(0,expand+1):
        fi_row_idxs = np.concatenate([fi_row_idxs,row_idxs+i])
        fi_row_idxs = np.concatenate([fi_row_idxs,row_idxs-i])
    fi_row_idxs = np.unique(fi_row_idxs)

    fi_row_idxs = fi_row_idxs[fi_row_idxs>=0]
    fi_row_idxs = fi_row_idxs[fi_row_idxs< img.shape[0]]
    if fill == 'none':
        img_out = np.delete(img,fi_row_idxs,axis=0)
    if fill == 'linear':
        del_cols = []
        if len(fi_row_idxs) > 0:
            i = 0
            st = 0
            ed = st + 1
            while ed < len(fi_row_idxs):
                while fi_row_idxs[ed] - fi_row_idxs[ed-1] == 1 and ed != len(fi_row_idxs)-1:
                    ed += 1
                s_idx = fi_row_idxs[st]
                e_idx = fi_row_idxs[ed-1]
                if s_idx != 0 and e_idx != len(img_out) - 1:
                    up_row = img_out[s_idx-1]
                    down_row = img_out[e_idx+1]
                    img_out[s_idx:e_idx+1] = _tool(up_row,down_row,e_idx-s_idx+1)
                else:
                    for idx in range(s_idx,e_idx+1):
                        del_cols.append(idx)
                
                st = ed
                ed += 1
        img_out = np.delete(img_out,del_cols,axis=0)

    return img_out

def set_brightness(img,mean = 128):
    # cv2.imshow('before,',img)
    # print(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    mean_l = np.mean(img[:,:,1])
    img[:,:,1] = img[:,:,1] - mean_l + mean
    img = cv2.cvtColor(img,cv2.COLOR_HLS2BGR)  
    # cv2.imshow('after',img)
    # print(img)
    # cv2.waitKey(0)
    return img

def get_brightness(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    mean_l = np.mean(img[:,:,1])
    return mean_l

def contrast(img,ratio = 1.):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    L = img[:,:,1]
    mean_l = np.mean(L)
    print(mean_l)
    L = np.where( np.logical_and(L < (mean_l+10), L > (mean_l-10)), np.ones_like(L)*mean_l, L)
    L = mean_l + (L - mean_l) * ratio
    L = np.where(L < 0,np.zeros_like(L),L)
    L = np.where(L > 255, np.ones_like(L)*255,L)
    img[:,:,1] = L.astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_HLS2BGR)
    return img

def get_pics(dir_path,st=-1,ed=-1):
    files = os.listdir(dir_path)
    pics = []
    for file in files:
        pics.append(cv_imread(dir_path+'/'+file))
    return pics

def pixel_diff(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgc = img[1:-1,1:-1]
    imgu = img[:-2,1:-1]
    imgd = img[2:,1:-1]
    imgl = img[1:-1,:-2]
    imgr = img[1:-1,2:]
    imglu = img[:-2,:-2]
    imgld = img[2:,:-2]
    imgru = img[:-2,2:]
    imgrd = img[2:,2:]
    neighbors = [imgu,imgd,imgl,imgr,imglu,imgld,imgru,imgrd]
    diff = np.zeros_like(imgc)
    for neigh in neighbors:
        diff += np.abs(imgc - neigh)
    diff = diff / (imgc.shape[0]*imgc.shape[1])
    return diff

def rmLine(img,hLine_th=0.9,vLine_th=0.95,h_expand=4,v_expand=1,fill='none'):
    img = rm_HLine(img,line_th=hLine_th,expand=h_expand,fill=fill)
    img = np.transpose(img,(1,0,2))
    img = rm_HLine(img,line_th=vLine_th,expand=v_expand,fill=fill)
    img = np.transpose(img,(1,0,2))
    return img

def imshow(imgs, name='',resize = 1):
    if name == '':
        name = str(random())
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    width,height = imgs[0].shape[1],imgs[0].shape[0]
    img1 = imgs[0]
    if len(imgs)>1:
        for img in imgs[1:]:
            img1 = np.concatenate([img1,img],axis=1)
            width += img.shape[1]
    cv2.imshow(name,img1)
    cv2.resizeWindow(name,width*resize,height*resize)
    key = cv2.waitKey(0)
    cv2.destroyWindow(name)
    if key == ord('q'):
        quit()
    try:
        re_key = chr(key)
    except:
        re_key = None
    return re_key

def imshow_with_trackbar(img,trackbar_list,onChange,win_name=''):
    if win_name == '':
        win_name = str(random())
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    width,height = img.shape[1],img.shape[0]
    cv2.resizeWindow(win_name,width,height)
    for trackbar in trackbar_list:
        cv2.createTrackbar(trackbar['name'],win_name,trackbar['min'],trackbar['count'],onChange)
    cv2.imshow(win_name,img)
    cv2.waitKey(0)

# 寻找一维曲线中的极值点
def findPeaks(pic,data,distance=1):
    from scipy import signal
    peaks_idx, _ = signal.find_peaks(-data,distance=distance)
    # cv2.line(pic,(0,peaks_idx[0]),(pic.shape[1],peaks_idx[0]),color=CV_COLOR_RED,thickness=1)
    # cv2.line(pic,(0,peaks_idx[-1]),(pic.shape[1],peaks_idx[-1]),color=CV_COLOR_RED,thickness=1)
    # imshow(pic[-100:,500:1000])
    plt.plot(data)
    for i in range(len(peaks_idx)):
        plt.plot(peaks_idx[i],data[peaks_idx[i]],'*',markersize=10)
    plt.show()

def row_histogram(pic):
    return np.mean(pic,axis=1,dtype=np.float32)

def row_histogram_half(pic):
    pic = np.array(pic,dtype=np.float32)
    half_row = pic.shape[0]//2
    pic_up = pic[:half_row]
    pic_down = pic[half_row+1:]
    up,down = np.mean(pic_up,axis=1,dtype=np.float32), np.sum(pic_down, axis=1,dtype=np.float32)
    # left = left - np.mean(left)
    # right = right - np.mean(right)
    return up,down

def row_histogram_left_half(pic):
    pic = np.array(pic,dtype=np.float)
    half_col = pic.shape[1]//2
    half_row = pic.shape[0]//2
    pic_left_up = pic[:half_row,:half_col]
    pic_left_down = pic[half_row+1:,:half_col]
    return np.mean(pic_left_up,axis=1,dtype=np.float), np.sum(pic_left_down,axis=1,dtype=np.float)

def row_histogram_right_half(pic):
    pic = np.array(pic,dtype=np.float)
    half_col = pic.shape[1]//2
    half_row = pic.shape[0]//2
    pic_right_up = pic[:half_row,half_col:]
    pic_right_down = pic[half_row+1:,half_col:]
    return np.mean(pic_right_up,axis=1,dtype=np.float), np.sum(pic_right_down,axis=1,dtype=np.float)

def col_histogram_trisector(pic):
    pic = np.array(pic,dtype=np.float)
    pic = cv2.resize(pic,(pic.shape[1],pic.shape[0]//3*3))
    h_gap = pic.shape[0] // 3
    pic1 = pic[:h_gap]
    pic2 = pic[h_gap:2*h_gap]
    pic3 = pic[2*h_gap:]
    return np.mean(pic1,axis=0,dtype=np.float), np.sum(pic2,axis=0,dtype=np.float), np.sum(pic3,axis=0,dtype=np.float)

def col_histogram(pic):
    return np.mean(pic,axis=0,dtype=np.float32)

def count_statistic(data,count = 100):
    ''' 输入一个一维数据，统计一维数据从大到小落在各个区间上的元素个数 '''
    count_arr = np.zeros(101,dtype=np.int)
    minElement = np.min(data)
    maxElement = np.max(data)
    gap = (maxElement - minElement)/100.0
    idxs = np.arange(100)*gap + minElement
    for element in data:
        count_arr[math.floor((element-minElement)/gap)] += 1
    count_arr[-2] += count_arr[-1]
    return count_arr[:-1],idxs

def cos_sim(data1,data2):
    return np.sum(data1*data2) / (np.sqrt(np.sum(np.square(data1))) * np.sqrt(np.sum(np.square(data2))) + 1e-5)

def save_mask(mask_batch):
    ''' 输入144张小图，拼成大图 '''
    mask_batch = np.array(mask_batch)
    height = mask_batch[0].shape[0]
    width = mask_batch[0].shape[1]
    big_img = np.zeros(shape=(6*height,24*width),dtype=np.uint8)
    for row in range(6):
        for col in range(24):
            big_img[row*height:row*height+height,col*width:col*width+width] = (mask_batch[row*24+col]).astype(np.uint8)
    big_img = cv2.resize(big_img,dsize=(5800,3499),interpolation=cv2.INTER_AREA)
    return big_img

def CCOEFF_NORMED(img, template, mask = None) -> float:
    _img = np.array(img,dtype=float)
    _template = np.array(template,dtype=float)
    _img = _img - np.mean(_img)
    _template = _template - np.mean(_template)
    if mask is not None:
        _mask = np.array(mask,dtype=float)
        return np.sum(_img * _template * _mask) / (np.sqrt(np.sum(np.square(_img)))*np.sqrt(np.sum(np.square(_template))) + 1e-5)
    else:
        return np.sum(_img * _template) / (np.sqrt(np.sum(np.square(_img)))*np.sqrt(np.sum(np.square(_template))) + 1e-5)


if __name__ == '__main__':

    pass
