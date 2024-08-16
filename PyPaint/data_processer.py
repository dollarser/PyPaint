""" 历史参考文件，已无用 """
from matplotlib.pyplot import axis
import cv2
import numpy as np
import math
from numpy.ma.core import right_shift
from skimage import exposure
import json
import os
import pandas as pd
import statsmodels.api as sm
from scipy import signal
from utils.utility import *


class Data_processer:

    def __init__(self):
        self.split_result = np.zeros((6, 24), dtype=object)
        self.boundries = np.zeros((7,25,4)) # (x, y, height, width)

    def brightness_adjust(self,img):
        ''' 调整图片的亮度到128  '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        mean_l = np.mean(img[:,:,1])
        img[:,:,1] = img[:,:,1] - mean_l + 128
        img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
        return img

    def rm_HLine(self, img, line_th=0.9, expand=4, fill='none', channel=3):
        ''' 去除图片中的横线，一行中的黑色像素超过整行的0.9时去除该行以及该行上下expand行
            先将图片二值化，再统计每行的黑色像素数
            fill='none'时只去除，不填充
            fill='linear'时去除的地方线性填充
        '''
        # num是up_row与down_row之间相差的行数，num  = idx_up_row - idx_down_row - 1
        def _tool(up_row,down_row,num):
            up_row = up_row.astype('float32')
            down_row = down_row.astype('float32')
            if channel ==3:
                res = np.zeros((num,len(up_row),3),dtype='float32')
            else:
                res = np.zeros((num,len(up_row)),dtype='float32')
            space = (down_row - up_row) / (num+1)
            for i in range(num):
                res[i] = up_row + (i+1)*space
            return np.asarray(res,dtype=np.uint8)
        pic = np.array(img)
        img_out = np.array(img)
        pic = self.gaussianblur_iten_binary(pic,gaus_sigma=2,binary_brightness_th=2)
        if channel == 3:
            pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
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
            # img_out = np.delete(img_out,del_cols,axis=0)

        return img_out

    def rmLine(self,img,hLine_th=0.9,vLine_th=0.95,h_expand=4,v_expand=1,fill='none',channel=3):
        img = self.rm_HLine(img,line_th=hLine_th,expand=h_expand,fill=fill,channel=channel)
        if channel == 3:
            img = np.transpose(img,(1,0,2))
        else:
            img = np.transpose(img)
        img = self.rm_HLine(img,line_th=vLine_th,expand=v_expand,fill=fill,channel=channel)
        if channel == 3:
            img = np.transpose(img,(1,0,2))
        else:
            img = np.transpose(img)
        return img
    

    def white_balance(self, img):
        b, g, r = cv2.split(img)
        # detection(img)
        m, n, t = img.shape
        # print(b.shape)
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

    def adaptive_threshold(self,img,brightness_th=5,block_size=31):
        ''' 自适应二值化 '''
        image_copy = np.array(img)
        if len(img.shape) == 3:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        image_copy = cv2.adaptiveThreshold(image_copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, brightness_th)
        if len(img.shape) == 3:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        return image_copy
        
    def gaussianblur_iten_binary(self,img,gaus_sigma=2, binary_brightness_th=11):
        ''' 高斯滤波降噪 加 强度调整 加 自适应二值化 '''
        img_out = np.array(img,dtype=np.uint8)
        img_out = cv2.GaussianBlur(img_out,(33,33),gaus_sigma)
        img_out = exposure.rescale_intensity(img_out)
        img_out = self.adaptive_threshold(img_out,binary_brightness_th)
        return img_out

    def gaussianblur_binary(self,img,gaus_sigma=2,binary_brightness_th=10,bin_block_size=31):
        ''' 高斯滤波降噪 加 自适应二值化 '''
        img_out = np.array(img,dtype=np.uint8)
        img_out = cv2.GaussianBlur(img_out,(7,3),gaus_sigma)
        img_out = self.adaptive_threshold(img_out,binary_brightness_th,bin_block_size)
        return img_out

    def auto_binary(self, img, win = 45, beta = 0.92):
        if win % 2 == 0: win = win - 1
        # 边界的均值有点麻烦
        # 这里分别计算和和邻居数再相除
        kern = np.ones([win, win])
        sums = signal.correlate2d(img, kern, 'same')
        cnts = signal.correlate2d(np.ones_like(img), kern, 'same')
        means = sums // cnts
        # print(means)
        # print(img)
        zero = np.zeros(means.shape)
        # 如果直接采用均值作为阈值，背景会变花
        # 但是相邻背景颜色相差不大
        # 所以乘个系数把它们过滤掉
        means = np.where(means <= 83, 83, means)
        # print(means)

        img = np.where(img <= means * beta, 0, 255)
        return img
    
    def gauss_filter(self, img, kernal_X = 7, kernal_Y = 3, gauss_sigma=2):
        gauss = cv2.GaussianBlur(img, (kernal_X, kernal_Y), gauss_sigma)
        return gauss
    
    def Canny(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gauss = cv2.GaussianBlur(gray, (3, 3), 1)
        canny = cv2.Canny(gauss, 13, 13*2.5)
        canny = np.expand_dims(canny, 2)
        canny = np.tile(canny, (1,3))
        return canny

    def inten(self,img):
        ''' skimage图像强度调整，假设图像的像素范围是 (minPixel, maxPixel)，
        将图像的像素范围拉伸到(0,255) '''
        return exposure.rescale_intensity(img)
    
    def mWindow(self,img):
        '''
        滑动窗口切分图片
        '''
        height = img.shape[0]
        width = img.shape[1]
        win_height = 128
        win_width = 64
        height_sp = np.linspace(win_height,height,4,dtype=int)
        width_sp = np.linspace(win_width,width,4,dtype=int)
        pics = []
        for h in height_sp:
            for w in width_sp:
                pics.append(img[h-win_height:h,w-win_width:w])
        return pics

    def half(self,img):
        half_row = img.shape[0]//2
        pic1 = img[:half_row]
        pic2 = img[half_row:]
        pic_partitions = [pic1,pic2]
        return pic_partitions
    
    def quarter(self,pic):
        half_row = pic.shape[0]//2
        half_col = pic.shape[1]//2
        pic1 = pic[:half_row,:half_col]
        pic2 = pic[:half_row,half_col:]
        pic3 = pic[half_row:,:half_col]
        pic4 = pic[half_row:,half_col:]
        pic_partitions = [pic1,pic2,pic3,pic4]
        return pic_partitions

    def dct_dct(self,img,percentage=0.004,reverse=0):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 数据类型转换 转换为浮点型
        img1 = img.astype(np.float)

        # 进行离散余弦变换
        img_dct = cv2.dct(img1)

        # 获取percentage处的阈值
        w,h=img_dct.shape
        score = img_dct.reshape(w*h)
        idxs = np.argsort(score)

        # 去除高频信息
        if reverse:
            threshold = idxs[int(len(score)*percentage + 1)]
            # print("threshold",threshold)
            img_dct2 = img_dct.copy()
            img_dct2[img_dct2>score[idxs[threshold]]] = 0  
        # 去除低频信息
        else:
            idxs = idxs[::-1]
            threshold = idxs[int(len(score)*percentage + 1)]
            img_dct2 = img_dct.copy()
            img_dct2[img_dct2<score[idxs[threshold]]] = 0 

        # 进行log处理
        # img_dct_log = np.log(abs(img_dct2))
        img_dct_log = img_dct2

        # # 进行离散余弦反变换
        # img_idct = cv2.idct(img_dct2)
        # plt.imshow(img_idct,'gray')
        # plt.savefig("temp001.jpg")
        # res = img_idct.astype(np.uint8) # 浮点型转整型 小数部分截断
        img_dct_log = np.expand_dims(img_dct_log, axis=2).repeat(3, axis=2)
        return img_dct_log
    
    def stl(self,img):
        ''' stl分解 请使用灰度图, 返回 季节， 趋势， 残差， 原图'''
        img = np.transpose(img)

        h,w=img.shape
        # reshape成一维数据训练
        img_flatten = img.reshape(-1)
        # 转换为pandas类型的数据进行分解
        series = pd.Series(img_flatten) 
        # 周期为图片的宽
        res = sm.tsa.seasonal_decompose(series, period=w, extrapolate_trend='freq')
        # res = sm.tsa.seasonal_decompose(s,period=w)

        # 获取分量 还原图片
        resid = np.array(res.resid).reshape((h,w))
        season = np.array(res.seasonal).reshape((h,w))
        trend = np.array(res.trend).reshape((h,w))
        observed = np.array(res.observed).reshape((h,w))
        
        resid = np.transpose(resid)
        season = np.transpose(season)
        trend = np.transpose(trend)
        observed = np.transpose(observed)
        
        resid = (resid - np.min(resid))*255/(np.max(resid)-np.min(resid))
        season = (season - np.min(season))*255/(np.max(season)-np.min(season))
        trend = (trend - np.min(trend))*255/(np.max(trend)-np.min(trend))
        observed = (observed - np.min(observed))*255/(np.max(observed)-np.min(observed))

        return season, trend, resid, observed
    
    def norm(self, array, up = 1, down = 0):
        return (array - np.min(array)) / (np.max(array) - np.min(array)) * (up-down) + down

    def row_histogram(self, pic):
        return np.mean(pic,axis=1,dtype=np.float32)

    def col_histogram(self, pic):
        return np.mean(pic,axis=0,dtype=np.float32)
    
    
    def gamma(self,pic,gamma = 1):
        table = np.arange(256,dtype=np.float)
        table = (table / 255)**gamma*255
        table = np.array(table).astype(np.uint8)
        pic = cv2.LUT(pic,table)
        return pic


    def get_big_img(self, pic):
        big_img = np.zeros(shape=(pic.shape[0] * 3, pic.shape[1] * 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                big_img[i * pic.shape[0] : (i+1) * pic.shape[0], j * pic.shape[1] : (j+1) * pic.shape[1]] = pic
        return big_img

 
    # 上下翻转数据
    def flip_image_ud(self, pic):
        pic = np.flip(pic, axis=0)
        return pic
    
    # 左右翻转
    def flip_image_lr(self, pic):
        pic = np.flip(pic, axis=1)
        return pic
    
    # 中心翻转
    def flip_image_central(self, pic):
        return self.flip_image_lr(self.flip_image_ud(pic))
    
    def el_split_equal_space(self, pic,row = 6, col = 24):
        '''
        将大图切成小图，6行24列，底部黑边101像素
        返回结果是长度144的小图列表
        '''
        height,width = pic.shape[0],pic.shape[1]
        height_sp = np.asarray(np.linspace(0,height,num = row+1),dtype = np.int)
        width_sp = np.asarray(np.linspace(0,width,num=col+1),dtype = np.int)

        subpics = []
        hs = 0
        ws = 0
        for h in height_sp[1:]:
            ws = 0
            for w in width_sp[1:]:
                tmp_pic = np.array(pic[hs:h,ws:w],dtype=np.uint8)
                # tmp_pic = cv2.resize(tmp_pic,dsize=(256,512))
                subpics.append(tmp_pic)
                ws = w
            hs = h
        return subpics

    def el_split_based_pixel(self, pic, row = 6, col = 24):
        ''' 请使用去除底部黑条后的大片图像 '''
        ROW_REFERENCED_POS = np.linspace(0, pic.shape[0], num=row+1,dtype=np.int)
        COL_REFERENCED_POS = np.linspace(0, pic.shape[1], num=col+1,dtype=np.int)
        row_border_range = np.ones(row,dtype=np.int) * 8
        col_border_range = np.ones(col,dtype=np.int) * 10
        col_border_range[col // 2] = 15
        col_border_range[0] = 15
        col_border_range[-1] = 15
        if row == 6:
            row_border_range[2] = 15
            row_border_range[4] = 15
        def _offset_left(histo,idxs,border_range):
            idx_offset = np.zeros_like(idxs,dtype=np.int)
            for i,idx in enumerate(idxs):
                idx = idx - 1
                mean_idx = np.arange(idx - border_range[i], idx)
                thres = np.mean(histo[mean_idx],dtype=np.float)
                # plt.hlines(thres,idx-20, idx, colors='red')
                while(idx >= 0 and histo[idx] < thres):
                    idx = idx - 1
                idx_offset[i] = idx
            return idx_offset

        def _offset_right(histo,idxs,border_range):
            idx_offset = np.zeros_like(idxs,dtype=np.int)
            for i,idx in enumerate(idxs):
                idx = idx + 1
                mean_idx = np.arange(idx, idx + border_range[i])
                thres = np.mean(histo[mean_idx],dtype=np.float)
                # plt.hlines(thres,idx, idx+20,colors='red')
                while(idx < len(histo) and histo[idx] < thres):
                    idx = idx + 1
                idx_offset[i] = idx
            return idx_offset

        expand = 100
        # row_peaks_distance = pic.shape[0] // (row+0.5)
        # col_peaks_distance = pic.shape[1] // (col+1)
        row_prominence_window = pic.shape[0] // row // 10
        row_peaks_distance = pic.shape[0] // row // 10
        col_prominence_window = pic.shape[1] // col // 5
        col_peaks_distance = pic.shape[1] // col * 0.8

        row_histo = self.row_histogram(pic)
        row_histo = self.norm(row_histo, 255, 0)
        row_histo_body = row_histo[expand:-expand]
        if row == 1:
            row_peaks = []
        else:
            row_peaks, row_peaks_info = signal.find_peaks(-row_histo_body,prominence=20,distance=row_peaks_distance)
            prominences = self.find_prominences(row_peaks, -row_histo_body, window = row_prominence_window)
            idxs = np.argsort(prominences)[::-1][:row-1]
            row_peaks = row_peaks[idxs]
            row_peaks = np.sort(row_peaks)
            row_peaks = row_peaks + expand
        row_peaks = np.concatenate([ [0], row_peaks, [pic.shape[0]]],axis=0).astype(int)
        # assert len(row_peaks) == row + 1
        # print(row_peaks,ROW_REFERENCED_POS)
        # print(np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)))
        # assert np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)) < 30
        ''' 画行峰值图 '''
        # plt.plot(row_histo)
        # for peak in row_peaks[1:-1]:
        #     plt.plot((peak), (row_histo[peak]), '*')
        # plt.show()

        if not (len(row_peaks) == row+1 and np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)) < 45):
            print(row_peaks, ROW_REFERENCED_POS)
            print('row error', len(row_peaks), row+1, np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)))
            return None, None
        
        row_ed = _offset_left(row_histo,row_peaks[1:],row_border_range[::-1])
        row_st = _offset_right(row_histo,row_peaks[:-1],row_border_range)

        col_histo = self.col_histogram(pic)
        col_histo = self.norm(col_histo, 255, 0)
        col_histo_body = col_histo[expand:-expand]
        if col == 1:
            col_peaks = []
        else:
            col_peaks,_ = signal.find_peaks(-col_histo_body,prominence=20, distance=col_peaks_distance)
            col_peaks = col_peaks + expand
        col_peaks = np.concatenate([[0],col_peaks,[pic.shape[1]]],axis=0).astype(int)
        assert len(col_peaks) == col + 1
        # assert np.sum(np.abs(col_peaks - COL_REFERENCED_POS)) < 115
        # print(col_peaks,COL_REFERENCED_POS,np.sum(np.abs(col_peaks - COL_REFERENCED_POS)))

        ''' 画列峰值图 '''
        # plt.plot(col_histo)
        # for peak in col_peaks[1:-1]:
        #     plt.plot((peak), (col_histo[peak]), '*')
        # plt.show()

        if not (len(col_peaks) == col + 1):
            print('col error')
            return None, None
        col_ed = _offset_left(col_histo,col_peaks[1:],col_border_range[::-1])
        col_st = _offset_right(col_histo,col_peaks[:-1],col_border_range)
        
        spics = []
        for i in range(len(row_st)):
            for j in range(len(col_st)):
                spic = pic[row_st[i]:row_ed[i], col_st[j]:col_ed[j]]
                # spic = cv2.resize(spic,dsize=(256,512),interpolation=cv2.INTER_AREA)
                spics.append(spic)
        spicsWithBorder = []
        for i in range(1,len(row_peaks)):
            for j in range(1,len(col_peaks)):
                spicWithBorder = pic[row_peaks[i-1]:row_peaks[i],col_peaks[j-1]:col_peaks[j]]
                spicWithBorder = cv2.resize(spicWithBorder, dsize=(256,512),interpolation=cv2.INTER_AREA)
                spicsWithBorder.append(spicWithBorder)

        return spics, spicsWithBorder

    def el_split_based_pixel1(self, pic, row = 6, col = 24):
        '''找边框线用峰值，找边框宽度用二值化，
         请使用去除底部黑条后的大片图像 '''
        from paint.utils.utility import CV_COLOR_RED

        ROW_REFERENCED_POS = np.linspace(0, pic.shape[0], num=row+1,dtype=np.int)
        COL_REFERENCED_POS = np.linspace(0, pic.shape[1], num=col+1,dtype=np.int)
        pic_bin = self.adaptive_threshold(pic, 4.5,51)

        def _offset_left(pic,idxs):
            idx_offset = np.zeros_like(idxs,dtype=np.int)
            for i,idx in enumerate(idxs):
                st = idx - 30
                ed = idx
                window = pic[st:ed]
                window_bin = self.adaptive_threshold(window, 4.5, 51)
                # idx_o = ed - 2
                idx_o = st
                while idx_o < ed and np.sum(window_bin[idx_o - st] == 0) < 0.5 * window_bin.shape[1]:
                    idx_o += 1
                idx_offset[i] = idx_o
                cv2.line(pic_bin, (0,idx_o),(5800,idx_o), CV_COLOR_RED, 1)
            return idx_offset

        def _offset_right(pic,idxs):
            idx_offset = np.zeros_like(idxs,dtype=np.int)
            for i,idx in enumerate(idxs):
                st = idx
                ed = idx + 30
                window = pic[st:ed]
                window_bin = self.adaptive_threshold(window, 4.5, 51)
                # idx_o = st
                idx_o = ed - 1
                while idx_o >= st and np.sum(window_bin[idx_o - st] == 0) < 0.5 * window_bin.shape[1]:
                    idx_o -= 1
                idx_offset[i] = idx_o+1
            return idx_offset

        expand = 100
        # row_peaks_distance = pic.shape[0] // (row+0.5)
        # col_peaks_distance = pic.shape[1] // (col+1)
        row_prominence_window = pic.shape[0] // row // 10
        row_peaks_distance = pic.shape[0] // row // 10
        col_prominence_window = pic.shape[1] // col // 5
        col_peaks_distance = pic.shape[1] // col * 0.8

        row_histo = self.row_histogram(pic)
        row_histo = self.norm(row_histo, 255, 0)
        row_histo_body = row_histo[expand:-expand]
        if row == 1:
            row_peaks = []
        else:
            row_peaks, row_peaks_info = signal.find_peaks(-row_histo_body,prominence=20,distance=row_peaks_distance)
            prominences = self.find_prominences(row_peaks, -row_histo_body, window = row_prominence_window)
            idxs = np.argsort(prominences)[::-1][:row-1]
            row_peaks = row_peaks[idxs]
            row_peaks = np.sort(row_peaks)
            row_peaks = row_peaks + expand
        row_peaks = np.concatenate([ [0], row_peaks, [pic.shape[0]]],axis=0).astype(int)
        # assert len(row_peaks) == row + 1
        # print(row_peaks,ROW_REFERENCED_POS)
        # print(np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)))
        # assert np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)) < 30
        ''' 画行峰值图 '''
        # plt.plot(row_histo)
        # for peak in row_peaks[1:-1]:
        #     plt.plot((peak), (row_histo[peak]), '*')
        # plt.show()

        if not (len(row_peaks) == row+1 and np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)) < 45):
            print(row_peaks, ROW_REFERENCED_POS)
            print('row error', len(row_peaks), row+1, np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)))
            return None, None
        
        row_ed = _offset_left(pic,row_peaks[1:])
        row_st = _offset_right(pic,row_peaks[:-1])


        col_histo = self.col_histogram(pic)
        col_histo = self.norm(col_histo, 255, 0)
        col_histo_body = col_histo[expand:-expand]
        if col == 1:
            col_peaks = []
        else:
            col_peaks,_ = signal.find_peaks(-col_histo_body,prominence=20, distance=col_peaks_distance)
            col_peaks = col_peaks + expand
        col_peaks = np.concatenate([[0],col_peaks,[pic.shape[1]]],axis=0).astype(int)

        # assert len(col_peaks) == col + 1
        # assert np.sum(np.abs(col_peaks - COL_REFERENCED_POS)) < 115
        # print(col_peaks,COL_REFERENCED_POS,np.sum(np.abs(col_peaks - COL_REFERENCED_POS)))

        ''' 画列峰值图 '''
        # plt.plot(col_histo)
        # for peak in col_peaks[1:-1]:
        #     plt.plot((peak), (col_histo[peak]), '*')
        # plt.show()

        if not (len(col_peaks) == col + 1):
            print('col error')
            return None, None

        pic = np.transpose(pic)
        print('col...')
        col_ed = _offset_left(pic,col_peaks[1:])
        col_st = _offset_right(pic,col_peaks[:-1])
        pic = np.transpose(pic)

        spics = []
        for i in range(len(row_st)):
            for j in range(len(col_st)):
                # print(row_st[i], row_ed[i], col_st[i], col_ed[i])
                spic = pic[row_st[i]:row_ed[i], col_st[j]:col_ed[j]]
                spic = cv2.resize(spic,dsize=(256,512),interpolation=cv2.INTER_AREA)
                spics.append(spic)
        spicsWithBorder = []
        for i in range(1,len(row_peaks)):
            for j in range(1,len(col_peaks)):
                spicWithBorder = pic[row_peaks[i-1]:row_peaks[i],col_peaks[j-1]:col_peaks[j]]
                spicWithBorder = cv2.resize(spicWithBorder, dsize=(256,512),interpolation=cv2.INTER_AREA)
                spicsWithBorder.append(spicWithBorder)

        pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)
        for i in range(len(row_st)):
            cv2.line(pic, (0,row_st[i]),(5800, row_st[i]),CV_COLOR_RED, 1)
            cv2.line(pic, (0,row_ed[i]),(5800, row_ed[i]),CV_COLOR_RED, 1)
        for i in range(len(col_st)):
            cv2.line(pic, (col_st[i],0),(col_st[i], 3499),CV_COLOR_RED, 1)
            cv2.line(pic, (col_ed[i],0),(col_ed[i], 3499),CV_COLOR_RED, 1)

        # cv2.namedWindow('a')
        # cv2.imshow('a',pic_bin)
        # key = cv2.waitKey(0)
        # cv2.destroyAllWindows()


        return spics, spicsWithBorder, pic
        # return spics, spicsWithBorder

    def el_split_based_pixel_overlap(self, pic, row = 6, col = 24):
        ''' 找边框用峰值， 找边框宽度用平均值阈值 
        请使用去除底部黑条后的大片图像 '''
        ROW_REFERENCED_POS = np.linspace(0, pic.shape[0], num=row+1,dtype=np.int)
        COL_REFERENCED_POS = np.linspace(0, pic.shape[1], num=col+1,dtype=np.int)
        row_border_range = np.ones(row,dtype=np.int) * 8
        row_border_range[0] = 20
        row_border_range[-1] = 20
        col_border_range = np.ones(col,dtype=np.int) * 10
        col_border_range[col // 2] = 15
        col_border_range[0] = 15
        col_border_range[-1] = 15
        if row == 6:
            row_border_range[2] = 15
            row_border_range[4] = 15
        def _offset_left(histo,idxs,border_range):
            idx_offset = np.zeros_like(idxs,dtype=np.int)
            for i,idx in enumerate(idxs):
                idx = idx - 1
                mean_idx = np.arange(idx - border_range[i], idx)
                thres = np.mean(histo[mean_idx],dtype=np.float)
                while(idx >= 0 and histo[idx] < thres):
                    idx = idx - 1
                idx_offset[i] = idx
            return idx_offset

        def _offset_right(histo,idxs,border_range):
            idx_offset = np.zeros_like(idxs,dtype=np.int)
            for i,idx in enumerate(idxs):
                idx = idx + 1
                mean_idx = np.arange(idx, idx + border_range[i])
                thres = np.mean(histo[mean_idx],dtype=np.float)
                while(idx < len(histo) and histo[idx] < thres):
                    idx = idx + 1
                idx_offset[i] = idx
            return idx_offset

        expand = 100
        row_prominence_window = pic.shape[0] // row // 10
        row_peaks_distance = pic.shape[0] // row // 10
        col_prominence_window = pic.shape[1] // col // 5
        col_peaks_distance = pic.shape[1] // col * 0.8

        row_histo = self.row_histogram(pic)
        row_histo = self.norm(row_histo, 255, 0)
        row_histo_body = row_histo[expand:-expand]
        if row == 1:
            row_peaks = []
        else:
            row_peaks, row_peaks_info = signal.find_peaks(-row_histo_body,prominence=20,distance=row_peaks_distance)
            prominences = self.find_prominences(row_peaks, -row_histo_body, window = row_prominence_window)
            idxs = np.argsort(prominences)[::-1][:row-1]
            row_peaks = row_peaks[idxs]
            row_peaks = np.sort(row_peaks)
            row_peaks = row_peaks + expand
        row_peaks = np.concatenate([ [0], row_peaks, [pic.shape[0]]],axis=0).astype(int)

        # if not (len(row_peaks) == row+1 and np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)) < 40):
        if not len(row_peaks) == row+1:
            print(row_peaks, ROW_REFERENCED_POS)
            print('row error', len(row_peaks), row+1, np.sum(np.abs(row_peaks - ROW_REFERENCED_POS)))
            return None, None
        
        row_ed = _offset_left(row_histo,row_peaks[1:],row_border_range[::-1])
        row_st = _offset_right(row_histo,row_peaks[:-1],row_border_range)

        col_histo = self.col_histogram(pic)
        col_histo = self.norm(col_histo, 255, 0)
        col_histo_body = col_histo[expand:-expand]
        if col == 1:
            col_peaks = []
        else:
            col_peaks,_ = signal.find_peaks(-col_histo_body,prominence=20, distance=col_peaks_distance)
            # prominences = self.find_prominences(col_peaks, -col_histo_body, window = col_prominence_window)
            # idxs = np.argsort(prominences)[::-1][:col-1]
            # col_peaks = col_peaks[idxs]
            # col_peaks = np.sort(col_peaks)
            col_peaks = col_peaks + expand
        col_peaks = np.concatenate([[0],col_peaks,[pic.shape[1]]],axis=0).astype(int)
        # assert len(col_peaks) == col + 1
        if not (len(col_peaks) == col + 1 and np.sum(np.abs(col_peaks - COL_REFERENCED_POS)) < (200 * col / 24)):
            print('col error',col_peaks,COL_REFERENCED_POS)
            return None, None
        col_ed = _offset_left(col_histo,col_peaks[1:],col_border_range[::-1])
        col_st = _offset_right(col_histo,col_peaks[:-1],col_border_range)
        
        spics = []
        for i in range(len(row_st)):
            for j in range(len(col_st)):
                spic = pic[row_st[i]:row_ed[i], col_st[j]:col_ed[j]]
                # spic = cv2.resize(spic,dsize=(256,512),interpolation=cv2.INTER_AREA)
                spics.append(spic)
        spicsWithBorder = []
        for i in range(1,len(row_peaks)):
            for j in range(1,len(col_peaks)):
                spicWithBorder = pic[row_peaks[i-1]:row_peaks[i],col_peaks[j-1]:col_peaks[j]]
                spicWithBorder = cv2.resize(spicWithBorder, dsize=(256,512),interpolation=cv2.INTER_AREA)
                spicsWithBorder.append(spicWithBorder)
        # new
        big_img = self.get_big_img(pic) # 整合一张大图
        spicsWithMoreBorder = []
        row_more_border_st = []
        row_more_border_ed = []
        col_more_border_st = []
        col_more_border_ed = []
        overlap_width = 40 # 调整裁剪后包含的overlap区域的水平方向宽度
        overlap_height = 40
        # overlap_height = i nt(overlap_width / 241 * 583) # 按原图比例调整竖直方向的overlap宽度
        row_more_border_st.append(row_peaks[0] - overlap_height)
        col_more_border_st.append(col_peaks[0] - overlap_width)
        for i in range(1, len(row_st)):
            row_more_border_st.append(row_ed[i-1] - overlap_height)
            row_more_border_ed.append(row_st[i] + overlap_height)
        for i in range(1, len(col_st)):
            col_more_border_st.append(col_ed[i-1] - overlap_width)
            col_more_border_ed.append(col_st[i] + overlap_width)
        row_more_border_ed.append(row_peaks[-1] + overlap_height)
        col_more_border_ed.append(col_peaks[-1] + overlap_width)    
        for i in range(len(row_more_border_st)):
            for j in range(len(col_more_border_st)):
                spicWithMoreBorder = big_img[row_more_border_st[i] + pic.shape[0]:row_more_border_ed[i] + pic.shape[0], col_more_border_st[j] + pic.shape[1]:col_more_border_ed[j] + pic.shape[1]]
                spicWithMoreBorder = cv2.resize(spicWithMoreBorder,dsize=(256,512),interpolation=cv2.INTER_AREA)
                spicsWithMoreBorder.append(spicWithMoreBorder)
        # end new
        return spics, spicsWithBorder, spicsWithMoreBorder

    def el_split_based_binary_overlap(self, pic, row = 6, col = 24):
        ''' 使用二值化找边框，兼容明暗片，短路。不适用于不上电。返回有重叠小片
        请使用去除底部黑条后的大片图像 '''
        ROW_REFERENCED_POS = np.linspace(0, pic.shape[0], num=row+1, dtype=int)
        COL_REFERENCED_POS = np.linspace(0, pic.shape[1], num=col+1, dtype=int)

        def _get_splits_row(pic):
            reference_pos = ROW_REFERENCED_POS
            end = pic.shape[0]
            offset = 30
            splits = []
            for idx in reference_pos:
                if idx == 0:
                    splits.append(idx)
                    continue
                if idx == end:
                    splits.append(idx - 1)
                    continue
                st = idx - offset
                ed = idx + offset
                window = pic[st:ed]
                window_bin = self.adaptive_threshold(window, 4.5, 41)
                window_bin_mean = np.sum(window_bin, axis=1, dtype=np.int64)
                idxs_max = np.argwhere(window_bin_mean == np.min(window_bin_mean))
                split = np.mean(idxs_max, dtype=int) + st
                splits.append(split)
            return splits
        
        def _get_splits_col(pic):
            reference_pos = COL_REFERENCED_POS
            end = pic.shape[1]
            offset = int(pic.shape[1]/col/4)
            splits = []
            for idx in reference_pos:
                if idx == 0:
                    splits.append(idx)
                    continue
                if idx == end:
                    splits.append(idx - 1)
                    continue
                st = idx - offset
                ed = idx + offset
                window = pic[:,st:ed]
                window_bin = self.adaptive_threshold(window, 4.5, 41)
                window_bin_mean = np.sum(window_bin, axis=0, dtype=np.int64)
                idxs_max = np.argwhere(window_bin_mean == np.min(window_bin_mean))
                split = np.mean(idxs_max, dtype=int) + st
                splits.append(split)
            return splits
        
        row_splits = _get_splits_row(pic)
        # row_st = _offset_right(pic, row_splits[:-1])
        # row_ed = _offset_left(pic, row_splits[1:])

        col_splits = _get_splits_col(pic)
        row_splits = np.array(row_splits)
        col_splits = np.array(col_splits)
        row_splits = row_splits + pic.shape[0]
        col_splits = col_splits + pic.shape[1]
        overlap_width = 50 # 调整裁剪后包含的overlap区域的水平方向宽度
        overlap_height = int(overlap_width / pic.shape[1] * pic.shape[0]) # 按原图比例调整竖直方向的overlap宽度

        row_splits_st = row_splits[:-1] - overlap_height
        row_splits_ed = row_splits[1:] + overlap_height
        col_splits_st = col_splits[:-1] - overlap_width
        col_splits_ed = col_splits[1:] + overlap_width

        pic33 = self.get_big_img(pic)
        spicsWithBorder = []
        for i in range(len(row_splits_st)):
            for j in range(len(col_splits_st)):
                spicWithBorder = pic33[row_splits_st[i]:row_splits_ed[i], col_splits_st[j]:col_splits_ed[j]]
                spicWithBorder = cv2.resize(spicWithBorder, dsize=(256,512),interpolation=cv2.INTER_AREA)
                spicsWithBorder.append(spicWithBorder)


        pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)

        CV_COLOR_RED = (51,0,255)
        row_splits = row_splits - pic.shape[0]
        col_splits = col_splits - pic.shape[1]
        for i in range(len(row_splits)):
            cv2.line(pic, (0,row_splits[i]),(5800,row_splits[i]), CV_COLOR_RED, 20)

            # cv2.line(pic, (0,row_st[i]),(5800,row_st[i]), CV_COLOR_RED, 1)
            # cv2.line(pic, (0,row_ed[i]),(5800,row_ed[i]), CV_COLOR_RED, 1)

        for i in range(len(col_splits)):
            cv2.line(pic, (col_splits[i],0),(col_splits[i],3500), CV_COLOR_RED, 20)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        return spicsWithBorder, pic

    def el_split_based_binary(self, pic, row = 6, col = 24):
        '''找边框线用二值化，找边框宽度用二值化，
         请使用去除底部黑条后的大片图像 '''

        ROW_REFERENCED_POS = np.linspace(0, pic.shape[0], num=row+1,dtype=np.int)
        COL_REFERENCED_POS = np.linspace(0, pic.shape[1], num=col+1,dtype=np.int)
        pic_bin = self.adaptive_threshold(pic, 4.5,51)

        def _get_splits_row(pic):
            reference_pos = ROW_REFERENCED_POS
            end = pic.shape[0]
            offset = 30
            splits = []
            for idx in reference_pos:
                if idx == 0:
                    splits.append(idx)
                    continue
                if idx == end:
                    splits.append(idx - 1)
                    continue
                st = idx - offset
                ed = idx + offset
                window = pic[st:ed]
                window_bin = self.adaptive_threshold(window, 4.5, 41)
                window_bin_mean = np.sum(window_bin, axis=1, dtype=np.int64)
                idxs_max = np.argwhere(window_bin_mean == np.min(window_bin_mean))
                split = np.mean(idxs_max, dtype=int) + st
                splits.append(split)
            return splits
        
        def _get_splits_col(pic):
            reference_pos = COL_REFERENCED_POS
            end = pic.shape[1]
            offset = int(pic.shape[1]/col/4)
            splits = []
            for idx in reference_pos:
                if idx == 0:
                    splits.append(idx)
                    continue
                if idx == end:
                    splits.append(idx - 1)
                    continue
                st = idx - offset
                ed = idx + offset
                window = pic[:,st:ed]
                window_bin = self.adaptive_threshold(window, 4.5, 41)
                window_bin_mean = np.sum(window_bin, axis=0, dtype=np.int64)
                idxs_max = np.argwhere(window_bin_mean == np.min(window_bin_mean))
                split = np.mean(idxs_max, dtype=int) + st
                splits.append(split)
            return splits
        


        def _offset_left(pic,idxs):
            idx_offset = np.zeros_like(idxs,dtype=np.int)
            for i,idx in enumerate(idxs):
                st = idx - 30
                ed = idx
                window = pic[st:ed]
                window_bin = self.adaptive_threshold(window, 4.5, 51)
                # idx_o = ed - 2
                idx_o = st
                while idx_o < ed and np.sum(window_bin[idx_o - st] == 0) < 0.5 * window_bin.shape[1]:
                    idx_o += 1
                idx_offset[i] = idx_o
                cv2.line(pic_bin, (0,idx_o),(5800,idx_o), CV_COLOR_RED, 1)
            return idx_offset

        def _offset_right(pic,idxs):
            idx_offset = np.zeros_like(idxs,dtype=np.int)
            for i,idx in enumerate(idxs):
                st = idx
                ed = idx + 30
                window = pic[st:ed]
                window_bin = self.adaptive_threshold(window, 4.5, 51)
                # idx_o = st
                idx_o = ed - 1
                while idx_o >= st and np.sum(window_bin[idx_o - st] == 0) < 0.5 * window_bin.shape[1]:
                    idx_o -= 1
                idx_offset[i] = idx_o+1
            return idx_offset

        row_peaks = _get_splits_row(pic)
        row_ed = _offset_left(pic,row_peaks[1:])
        row_st = _offset_right(pic,row_peaks[:-1])

        col_peaks = _get_splits_col(pic)
        pic = np.transpose(pic)
        col_ed = _offset_left(pic,col_peaks[1:])
        col_st = _offset_right(pic,col_peaks[:-1])
        pic = np.transpose(pic)

        spics = []
        for i in range(len(row_st)):
            for j in range(len(col_st)):
                # print(row_st[i], row_ed[i], col_st[i], col_ed[i])
                spic = pic[row_st[i]:row_ed[i], col_st[j]:col_ed[j]]
                spic = cv2.resize(spic,dsize=(256,512),interpolation=cv2.INTER_AREA)
                spics.append(spic)
        spicsWithBorder = []
        for i in range(1,len(row_peaks)):
            for j in range(1,len(col_peaks)):
                spicWithBorder = pic[row_peaks[i-1]:row_peaks[i],col_peaks[j-1]:col_peaks[j]]
                spicWithBorder = cv2.resize(spicWithBorder, dsize=(256,512),interpolation=cv2.INTER_AREA)
                spicsWithBorder.append(spicWithBorder)

        return spics, spicsWithBorder

    def find_prominences(self, peaks, series, window):
        # 找峰值
        peaks_value = series[peaks]
        left_idxs = peaks - window
        left_idxs = np.where(left_idxs < 0, 0, left_idxs)
        right_idxs = peaks + window
        right_idxs = np.where(right_idxs >= len(series), len(series), right_idxs)
        left_mins = []
        right_mins = []
        for li, i, ri in zip(left_idxs, peaks, right_idxs):
            left_mins.append(np.min(series[li:i]))
            right_mins.append(np.min(series[i:ri]))
        left_mins = np.array(left_mins)
        right_mins = np.array(right_mins)
        prominences = 2*peaks_value - left_mins - right_mins
        return prominences


    def prepareToModel(self,pic,preprocess,gaus_sigma=2, bin_brightness_th=10, bin_block_size= 31):
        ''' 
            请输入二维[height,width]的图片
            返回预处理后[height,width,channel]的三通道图
        '''
        if len(pic.shape) != 2:
            raise Exception('please input 2-dimension pic')
        if pic.shape[0] != 512 or pic.shape[1] != 256:
            print('pic`s shape is not (512,256)')
            pic = cv2.resize(pic,dsize=(256,512),interpolation=cv2.INTER_AREA)
        
        if preprocess == 'stl':
            # 使用stl分解三通道图
            season, trend, resid, ori = self.stl(pic)
            ''' 使用季节，趋势，残差叠加的三通道图进行训练 '''
            season = np.expand_dims(season,axis=2)
            trend = np.expand_dims(trend,axis=2)
            resid = np.expand_dims(resid,axis=2)
            res = np.concatenate([resid,season,trend],axis=2)
            return res

        if preprocess == 'bin':
            bin = self.gaussianblur_binary(pic,gaus_sigma=gaus_sigma, binary_brightness_th=bin_brightness_th, bin_block_size=bin_block_size)
            return bin

        if preprocess == 'stl_channel1':
            # 使用stl分解三通道图
            season, trend, resid, ori = self.stl(pic)
            ''' 使用季节，趋势，残差叠加的三通道图进行训练 '''
            res = np.expand_dims(resid,axis=2)
            return res
        
        if preprocess == 'stl_ori_bin':
            season, trend, resid, ori = self.stl(pic)
            resid = np.expand_dims(resid,axis = 2)
            ori = np.expand_dims(ori,axis = 2)
            bin = self.gaussianblur_binary(pic, gaus_sigma, bin_brightness_th, bin_block_size)
            bin = np.expand_dims(bin,axis = 2)
            res = np.concatenate([resid,ori,bin],axis=2)
            return res


        if preprocess == 'stl_ori_diff':
            pic = np.array(pic,dtype=float)
            season, trend, resid, ori = self.stl(pic)
            resid = np.expand_dims(resid,axis = 2)
            ori = np.expand_dims(ori,axis = 2)
            diff = np.zeros_like(pic)
            diff[:,:-1] = pic[:,1:] - pic[:,:-1]
            diff[:,-1] = pic[:,0] - pic[:,-1] 
            diff = (diff - np.min(diff))/(np.max(diff) - np.min(diff)) * 255
            diff = np.expand_dims(diff, axis = 2)
            res = np.concatenate([resid,ori,diff],axis = 2)
            return res

        if preprocess == 'overlay':
            season, trend, resid, ori = self.stl(pic)
            resid = np.expand_dims(resid,axis = 2)
            ori = np.expand_dims(ori,axis = 2)
            bin = self.gaussianblur_binary(pic)
            bin = np.expand_dims(bin,axis = 2)
            res = np.concatenate([resid,bin],axis=2)
            res = np.min(res,axis = 2).astype(np.uint8)
            res = np.expand_dims(res, axis=2)
            return res

        if preprocess == 'sob_norm':
            season, trend, resid, ori = self.stl(pic)
            resid = np.expand_dims(resid,axis = 2)
            ori = np.expand_dims(ori,axis = 2)
            bin = self.gaussianblur_binary(pic)
            bin = np.expand_dims(bin,axis = 2)
            res = np.concatenate([resid,ori,bin],axis=2).astype(np.float)
            res = (res - 127) / 128.0
            return res

        if preprocess == 'none':
            res = np.expand_dims(pic,axis=2)
            return res

        return None
   

    def get_brightness_spics(self,spics):
        ''' 输入小图列表，输出每个小图的亮度 '''
        spics = np.array(spics, dtype=float)
        spics = spics / 255.0
        brightness_list = np.mean(spics, axis = (1,2),dtype = np.float)
        return brightness_list
    
    def get_std_spics(self,spics):
        ''' 输入小图列表，输出每个小图的亮度 '''
        spics = np.array(spics, dtype=float)
        spics = spics / 255.0
        std_list = np.std(spics, axis = (1,2),dtype = np.float)
        return std_list

    def get_maxBrightness_spics(self, spics):
        ''' 输入小图列表， 输出每个小图的最大像素值 '''
        spics = np.array(spics)
        max_brightness_list = np.max(spics, axis = (1,2))
        return max_brightness_list

    def get_peek_brightness_spics(self, spics, peeks = 200):
        spics = np.array(spics)
        peek_num = np.sum(spics>peeks, axis=(1,2))
        return peek_num
    
    def get_cos_sim_spics(self, spics):
        ''' 比较的对象为所有小图的均值 '''
        spics = np.array(spics)
        spics_mean = np.mean(spics, axis = 0, dtype = np.float)
        # spics_mean_histo = np.mean(spics_mean, axis = 1, dtype = np.float)
        spics_mean_histo = np.max(spics_mean, axis = 1)

        spics_mean_histo = np.repeat(np.expand_dims(spics_mean_histo, axis = 0), repeats = len(spics), axis = 0)
        # spics_row_histo = np.mean(spics, axis = 2, dtype = np.float)
        spics_row_histo = np.max(spics, axis = 2)

        spics_cos_sim = np.sum(spics_mean_histo * spics_row_histo, axis = 1) /\
                             (np.sqrt(np.sum(np.square(spics_mean_histo), axis = 1)) * np.sqrt(np.sum(np.square(spics_row_histo), axis = 1)) + 1e-5)
        return spics_cos_sim
    
    def el_remove_border(self, spics):
        def _el_remove_border(pic):
            pic_bin = self.gaussianblur_binary(pic, 1.0, 6, 49)
            width = pic_bin.shape[1]
            height = pic_bin.shape[0]
            pic_bin = np.where(pic_bin < 1, 1, 0)
            left = -1
            right = width
            for w in range(int(width*0.1), 0, -1):
                if np.sum(pic_bin[:,w]) > 0.7*height:
                    left = w
                    break
            for w in range(int(width*0.9), width, 1):
                if np.sum(pic_bin[:,w]) > 0.7*height:
                    right = w
                    break
            res = pic[:,left+1:right]
            # res = cv2.resize(res, (pic.shape[1],pic.shape[0]),cv2.INTER_AREA)
            return res
        res = list(map(_el_remove_border, spics))
        res = [np.transpose(item) for item in res]
        res = list(map(_el_remove_border, res))
        res = [np.transpose(item) for item in res]
        return res

    def el_split_segments(self, pic, res):
        ''' 格式为[[图片段，段包含的行数，正常段/异常段标志]...] '''
        from paint.utils.utility import CV_COLOR_RED
        name= 'a'
        # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    
        show_pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)
        row_mean = np.mean(pic, axis=1)
        row_mean_diff = np.abs(row_mean[1:] - row_mean[:-1])
        def _split_point(pic, split):
            if split == 0:
                return split
            if split == pic.shape[0]:
                return split
            offset = 30
            st = max(0,split - offset)
            ed = min(pic.shape[0], split+offset)
            window = pic[st:ed]
            window_bin = self.adaptive_threshold(window,brightness_th=4.5, block_size=41)
            window_bin_mean = np.mean(window_bin, axis=1)
            # print(window_bin_mean)
            idxs_max = np.argwhere(window_bin_mean == np.min(window_bin_mean))
            idx = int(np.mean(idxs_max))
            # print(idx)
            # cv2.line(window_bin, (0,idx),(5800, idx), CV_COLOR_RED, 1)
            # cv2.imshow(name,window_bin)
            # cv2.resizeWindow(name,3000,40)

            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     quit()
            return idx + st

        pic = np.array(pic, dtype=np.uint8)
        pic = cv2.resize(pic, (pic.shape[1],pic.shape[0]//len(res) * len(res)))
        h_gap = pic.shape[0] // len(res)
        i = 0
        segments = []
        while i < len(res):
            st = i
            ed = i + 1
            while res[st] == 0 and ed < len(res) and res[ed] == res[st]:
                ed += 1
                
            st_split = _split_point(pic,st*h_gap)
            ed_split = _split_point(pic,ed*h_gap)
            segments.append([pic[st_split:ed_split], ed - st, res[st]])
            i = ed
            # cv2.line(show_pic,(0,st_split),(5800,st_split),CV_COLOR_RED,1)
            # cv2.line(show_pic,(0,ed_split),(5800,ed_split),CV_COLOR_RED,1)
        # cv2.destroyWindow(name)
        
        return segments
        return segments, show_pic
    
    def remove_black_edges(self, pic, margin=10):
        # cv_imwrite(pic, f"ori-{round(random()*1230)}.jpg")
        def find_consective_zero(arr, n):
            consect_count = 0
            for i, x in enumerate(arr):
                if x != 0:
                    consect_count = 0
                else:
                    consect_count += 1
                    if consect_count > n:
                        return i - consect_count + 1
            return -1
        _, pic_bin = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
        row_hist = self.row_histogram(pic_bin)
        col_hist = self.col_histogram(pic_bin)
        row_hist = np.where(row_hist < 150, 0, 1)
        col_hist = np.where(col_hist < 120, 0, 1)
        left = find_consective_zero(col_hist, 300)
        right = len(col_hist) - 1 - find_consective_zero(col_hist[::-1], 300)
        top = find_consective_zero(row_hist, 500) 
        bottom = len(row_hist) - 1 - find_consective_zero(row_hist[::-1], 500)
        # print(f'bottom:{bottom}, {len(row_hist)}, {find_consective_zero(row_hist[::-1], 500)}')
        top = max(70, top - margin)
        bottom = min(len(row_hist)-1-198, bottom + margin)
        left = max(92, left - margin)
        right = min(len(col_hist)-1-92, right + margin)
        # print(f'bottom:{bottom}')
        # print(np.array(pic[top:bottom, left:right]).shape, "###")
        # cv_imwrite(pic[bottom: , :], f"shrink-{round(random()*1230)}.jpg")
        return pic[top:bottom, left:right], pic_bin[top:bottom, left:right]
        
    def split_recursively(self, pic, row=6, col=24):
        _, pic_bin = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
        self.find_split_line(pic, pic_bin, col, row, 0, 0, 0, 0)
        width = np.array(pic).shape[1]
        height = np.array(pic).shape[0]
        # for arr in self.boundries:
        #     arr[-1][1] = width - 1
        self.boundries[:, -1, 1] = width - 1
        """for a in self.boundries[-1]:
            a[0] = height - 1"""
        self.boundries[-1, :, 0] = height - 1
        return self.split_result

    def find_split_line(self, pic, pic_bin, horizontal_blocks_amount, vertical_blocks_amount, left_pos, top_pos, left_pixel_pos, top_pixel_pos, min_width=5, filename_prefix='default', folder_name='default'):
        def find_widest_split_line(row_hist, col_hist):
            row_max_width = 0
            row_cur_width = 0
            col_max_width = 0
            col_cur_width = 0
            row_e, col_e = -1, -1
            for i, x in enumerate(row_hist[200:-200]):
                if x == 1:
                    row_cur_width += 1
                else:
                    row_cur_width = 0
                if row_cur_width > row_max_width:
                    row_max_width = row_cur_width
                    row_e = i + 200
            for i, x in enumerate(col_hist[200:-200]):
                if x == 1:
                    col_cur_width += 1
                else:
                    col_cur_width = 0
                if col_cur_width > col_max_width:
                    col_max_width = col_cur_width
                    col_e = i + 200
            return (0, row_e, row_max_width) if row_max_width > col_max_width else (1, col_e, col_max_width) 
        print(top_pos, left_pos, vertical_blocks_amount, horizontal_blocks_amount)
        if horizontal_blocks_amount < 2 and vertical_blocks_amount < 2:
            print("single pic output..")
            self.boundries[top_pos][left_pos][0] = top_pixel_pos
            self.boundries[top_pos][left_pos][1] = left_pixel_pos
            self.boundries[top_pos][left_pos][2] = np.array(pic).shape[0]
            self.boundries[top_pos][left_pos][3] = np.array(pic).shape[1]
            self.split_result[top_pos][left_pos] = pic
            # self.write_small_pic(pic, pic_bin, left_pos, top_pos, filename_prefix=filename_prefix, folder_name=folder_name)
            return
        
        row_hist = self.row_histogram(pic_bin)
        col_hist = self.col_histogram(pic_bin)
        row_hist = np.where(row_hist < 200, 0, 1)
        col_hist = np.where(col_hist < 120, 0, 1)
        isCol, end, width = find_widest_split_line(row_hist, col_hist)
        if width > 30:
            start = end - min_width
            end = end - width + 1 + min_width
        else:
            start = end - min_width
            end = end - width + 1 + min_width
        _hba_1 = round(horizontal_blocks_amount * end / len(col_hist))
        _hba_2 = round(horizontal_blocks_amount * (len(col_hist) - start) / len(col_hist))
        _vba_1 = round(vertical_blocks_amount * end / len(row_hist))
        _vba_2 = round(vertical_blocks_amount * (len(row_hist) - start) / len(row_hist))
        print(f'line width {width}')
        if width < min_width:
            print("split equally")
            self.split_equally(pic, pic_bin, horizontal_blocks_amount, vertical_blocks_amount, left_pos, top_pos, left_pixel_pos, top_pixel_pos, filename_prefix=filename_prefix, folder_name=folder_name)
        elif isCol:
            print("divide vertical to", f'({vertical_blocks_amount},{_hba_1}) and ({vertical_blocks_amount},{_hba_2})')
            self.find_split_line(pic[:, 0:end], pic_bin[:, 0:end], _hba_1, vertical_blocks_amount, left_pos, top_pos, left_pixel_pos, top_pixel_pos, filename_prefix=filename_prefix, folder_name=folder_name)
            self.find_split_line(pic[:, start:], pic_bin[:, start:], _hba_2, vertical_blocks_amount, left_pos+_hba_1, top_pos, left_pixel_pos+start, top_pixel_pos, filename_prefix=filename_prefix, folder_name=folder_name)
        else:
            print("divide horizontal to", f'({_vba_1},{horizontal_blocks_amount}) and ({_vba_2},{horizontal_blocks_amount})')
            self.find_split_line(pic[0:end, :], pic_bin[0:end, :], horizontal_blocks_amount, _vba_1, left_pos, top_pos, left_pixel_pos, top_pixel_pos, filename_prefix=filename_prefix, folder_name=folder_name)
            self.find_split_line(pic[start:, :], pic_bin[start:, :], horizontal_blocks_amount, _vba_2, left_pos, top_pos+_vba_1, left_pixel_pos, top_pixel_pos+start, filename_prefix=filename_prefix, folder_name=folder_name)
        return

    def split_equally(self, pic, pic_bin, horizontal_blocks_amount, vertical_blocks_amount, left_pos, top_pos, left_pixel_pos, top_pixel_pos, filename_prefix='default', folder_name='default'):
        row_width = len(self.col_histogram(pic_bin))
        col_width = len(self.row_histogram(pic_bin))
        row_step = row_width / horizontal_blocks_amount
        col_step = col_width / vertical_blocks_amount
        for i in range(horizontal_blocks_amount):
            for j in range(vertical_blocks_amount):
                left = max(0, round(i * row_step))
                right = min(row_width-1, round((i + 1) * row_step))
                top = max(0, round(j * col_step))
                bottom = min(col_width-1, round((j+1) * col_step))
                self.split_result[top_pos+j][left_pos+i] = pic[top:bottom, left:right]
                self.boundries[top_pos+j][left_pos+i][0] = top_pixel_pos+top
                self.boundries[top_pos+j][left_pos+i][1] = left_pixel_pos+left
                self.boundries[top_pos+j][left_pos+i][2] = col_step
                self.boundries[top_pos+j][left_pos+i][3] = row_step
                # self.write_small_pic(pic[top:bottom, left:right], pic_bin[top:bottom, left:right], left_pos + i, top_pos + j, filename_prefix=filename_prefix, folder_name=folder_name)
        return

    def write_small_pic(self, pic, pic_bin, left_pos, top_pos, filename_prefix='default', folder_name='default'):
        if not os.path.exists(os.path.dirname(os.getcwd() + f'\\{folder_name}\\')):
            os.makedirs(os.path.dirname(os.getcwd() + f'\\{folder_name}\\'))
        cv_imwrite(pic, filepath=f'{folder_name}/{filename_prefix}_{top_pos}_{left_pos}.jpg')