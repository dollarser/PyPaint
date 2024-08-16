from numpy.lib import utils
from PyPaint.globval import *
import cv2
import numpy as np
import math
import os
from scipy import signal
from PyPaint.model import Stack
from PIL import ImageTk, Image


from PyPaint.image_process import ImageProcessor as ip
from utils.constants import CV_COLOR_GREEN, CV_COLOR_ORANGE, CV_COLOR_RED
from utils import cv_imread, cv_imwrite


class ImageProcesser():

    def __init__(self, dis_width, dis_height, args=None):
        """ 
        dis_width: 显示宽度
        dis_height：显示高度
        """
        self.image_buffer = Stack()
        self.image_copy = None
        self.ori_width = 0
        self.ori_height = 0
        self.dis_width = dis_width
        self.dis_height = dis_height
        self.dp = ip()
        self.show_split_result = True
        self.avg_gray_val = 127
        self.el_img_path = None
        self.args = args

    def set_display_scale(self, width, height):
        self.dis_width = width
        self.dis_height = height

    def set_ori_scale(self,width,height):
        self.ori_width = width
        self.ori_height = height
    
    def save_img(self,save_path):
        """ 保存图片调用 """
        if self.image_buffer.empty():
            return False
        cv_imwrite(self.image_buffer.top(), save_path)
        return True

    def create_image(self,width,height):
        ''' 创建一个空白图片 '''
        self.ori_width = width
        self.ori_height = height
        image = np.zeros((height,width,3),dtype = np.uint8)
        image.fill(255)
        self.image_buffer.clear()
        self.image_buffer.push(image)
        
    def load_image(self,image):
        ''' 载入一张新的图片 '''
        self.ori_width = image.shape[1]
        self.ori_height = image.shape[0]
        self.image_buffer.clear()
        self.image_buffer.push(image)
        
    def get_scale(self):
        return self.dis_width/self.ori_width

    def convert_to_display_image(self, image):
        scale = self.dis_width/self.ori_width
        image = cv2.resize(image, (math.floor(image.shape[1]*scale), math.floor(image.shape[0]*scale)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return ImageTk.PhotoImage(image=image)


    def insert_image(self, dis_cx, dis_cy, image):
        """ dis_cx, dis_cy，是插入图片的中心再画板上的坐标。且是放缩后的坐标。"""
        width = image.shape[1]
        height = image.shape[0]
        # 放缩回去
        x = math.floor(dis_cx*self.ori_width/self.dis_width) - width/2
        y = math.floor(dis_cy*self.ori_height/self.dis_height) - height/2

        x_start = math.floor(max(0,x))
        x_end = math.floor(min(x+width,self.ori_width))
        y_start = math.floor(max(0,y))
        y_end = math.floor(min(y+height,self.ori_height))

        x_ins_start = math.floor(max(0, -x))
        x_ins_end = math.floor(min(width,self.ori_width - x))
        y_ins_start = math.floor(max(0,-y))
        y_ins_end = math.floor(min(height,self.ori_height - y))

        self.image_copy = np.array(self.image_buffer.top())
        self.image_copy[y_start:y_end, x_start:x_end] = image[y_ins_start:y_ins_start+ y_end - y_start,x_ins_start:x_ins_start + x_end - x_start]
        self.image_buffer.push(np.array(self.image_copy))

    def get_dis_image(self):
        if self.image_buffer.empty():
            raise Exception('无图片缓存')
        image = self.image_buffer.top()
        image = cv2.resize(image,(self.dis_width,self.dis_height))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return ImageTk.PhotoImage(image=image)
    
    def clear_screen(self):
        self.image_copy = np.zeros((self.ori_height,self.ori_width,3),dtype=np.uint8)
        self.image_copy.fill(255)
        self.image_buffer.push(np.array(self.image_copy))

    def dis_to_ori(self,x,y):
        ''' canvas上的展示坐标点转换为原图上的坐标点 '''
        height_scale = self.ori_height/self.dis_height
        width_scale = self.ori_width/self.dis_width
        return math.floor(x * width_scale),math.floor(y * height_scale)

    def rgb_to_bgr(self,color):
        return (color[2],color[1],color[0])
    
    def draw_line(self, sx, sy, ex, ey, thickness=1):
        self.image_copy = np.array(self.image_buffer.top())
        sx,sy = self.dis_to_ori(sx,sy)
        ex,ey = self.dis_to_ori(ex,ey)
        cv2.line(self.image_copy, (sx,sy), (ex,ey), color=self.rgb_to_bgr(get_value('color_rgb')), thickness=thickness)
        self.image_buffer.push(np.array(self.image_copy))
    
    def draw_rectangle(self, sx, sy, ex, ey, thickness=1):
        self.image_copy = np.array(self.image_buffer.top())
        sx,sy = self.dis_to_ori(sx,sy)
        ex,ey = self.dis_to_ori(ex,ey)
        cv2.rectangle(self.image_copy,(sx,sy),(ex,ey),color=self.rgb_to_bgr(get_value('color_rgb')), thickness=thickness)
        self.image_buffer.push(np.array(self.image_copy))

    def clear_image_copy(self):
        self.image_copy = np.array(self.image_buffer.top())

    def draw_pencil(self, sx, sy, ex, ey, thickness=1):
        sx,sy = self.dis_to_ori(sx,sy)
        ex,ey = self.dis_to_ori(ex,ey)
        cv2.line(self.image_copy, (sx,sy), (ex,ey), color=self.rgb_to_bgr(get_value('color_rgb')), thickness=thickness)

    def draw_pencil_complete(self):
        self.image_buffer.push(np.array(self.image_copy))
        self.image_copy = None
    
    def draw_eraser(self, sx, sy, ex, ey):
        sx,sy = self.dis_to_ori(sx,sy)
        ex,ey = self.dis_to_ori(ex,ey)
        sx = np.clip(sx, 0, self.ori_width)
        sy = np.clip(sy, 0, self.ori_height)
        ex = np.clip(ex, 0, self.ori_width)
        ey = np.clip(ey, 0, self.ori_height)
        fill = np.zeros((ey-sy, ex-sx, 3), dtype=np.uint8)
        fill.fill(255)
        self.image_copy[sy:ey,sx:ex] = fill

    def draw_eraser_complete(self):
        self.image_buffer.push(np.array(self.image_copy))
        self.image_copy = None
    
    
    ##################################################
    def EL_select_init(self, el_path, agv):
        """ 没有图片无法分割 """
        if el_path is None:
            return
        self.el_img_path = el_path
        if agv != self.avg_gray_val:
            self.avg_gray_val = agv
            self.dp.agv = agv
            self.el_img, _ = self.dp.remove_black_edge(self.dp.cv_imread(el_path))
            self.dp.split_recursively(self.el_img)
        else:
            self.el_img, _ = self.dp.remove_black_edge(self.dp.cv_imread(el_path))
        print(np.array(self.el_img).shape)

        self.el_click = np.zeros((self.args.ROW, self.args.COL),dtype=int)
        self.el_img = cv2.cvtColor(self.el_img,cv2.COLOR_GRAY2BGR)
        self.dp.boundries = self.dp.boundries.astype(np.int32)
        if self.show_split_result:
            for i in range(6):
                for j in range(24):
                    cv2.rectangle(
                        self.el_img, 
                        (self.dp.boundries[i][j][1], self.dp.boundries[i][j][0]),
                        (self.dp.boundries[i][j][1] + self.dp.boundries[i][j][3], self.dp.boundries[i][j][0] + self.dp.boundries[i][j][2]),
                        (51+12*i,0+12*j,255-6*i-6*j),
                        10
                    )
        
        self.load_image(self.el_img)


    def draw_EL_select(self, x,y):
        print("######### draw_EL_select ##########")
        x, y = self.dis_to_ori(x, y)
        col_select, row_select = 0, 0
        selected_top_pos, selected_left_pos, s_width, s_height = 0, 0, 0, 0

        for i in range(self.args.ROW_NUM):
            for j in range(self.args.COL_NUM):
                top_pos = self.dp.boundries[i][j][0]
                left_pos = self.dp.boundries[i][j][1]
                width = self.dp.boundries[i][j][3]
                height = self.dp.boundries[i][j][2]
                if x >= left_pos and y >= top_pos and x <= left_pos + width  and y <= top_pos + height:
                    col_select = j
                    row_select = i
                    selected_top_pos, selected_left_pos = top_pos, left_pos
                    s_width = width
                    s_height = height
                    break
        print('click in', x, y)
        print('select', row_select, col_select)
        self.el_click[row_select][col_select] = (self.el_click[row_select][col_select] + 1)%3

        self.image_copy = np.array(self.image_buffer.top())
        if self.el_click[row_select][col_select] == 0:
            color = CV_COLOR_GREEN
        elif self.el_click[row_select][col_select] == 1:
            color = CV_COLOR_RED
        if self.el_click[row_select][col_select] == 2:
            color = CV_COLOR_ORANGE
        print(f'center in {selected_left_pos}~{selected_left_pos+s_width}, {selected_top_pos}~{selected_top_pos+s_height}')
        center = (round(selected_left_pos + s_width //2), round(selected_top_pos + s_height // 2))
        print(f'center:{center}')
        cv2.circle(self.image_copy, center, 30, color, 13)
        self.image_buffer.push(np.array(self.image_copy))
        pass

    def EL_select_complete(self):
        name = os.path.split(self.el_img_path)[1]
        # 当前大图
        el_img = self.el_img

        spics = self.dp.split_result

        # 小图写入到文件夹
        print(self.el_click)
        for i in range(self.args.ROW):
            for j in range(self.args.COL):
                if self.el_click[i][j] == 0:
                    cv_imwrite(spics[i][j], self.args.NORM_DIR + '/'+name[:-4]+'_'+str(i)+'_'+str(j)+'.jpg')
                if self.el_click[i][j] == 1:
                    cv_imwrite(spics[i][j], self.args.ANOM_DIR + '/'+name[:-4]+'_'+str(i)+'_'+str(j)+'.jpg')
                if self.el_click[i][j] == 2:
                    pass

    def gray(self):
        self.image_copy = np.array(self.image_buffer.top())
        self.image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
        self.image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_GRAY2BGR)
        self.image_buffer.push(np.array(self.image_copy))

    def adaptive_threshold(self):
        self.image_copy = np.array(self.image_buffer.top())
        self.image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
        self.image_copy = cv2.adaptiveThreshold(self.image_copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
        self.image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_GRAY2BGR)
        self.image_buffer.push(np.array(self.image_copy))

    def binarization(self,win, beta = 0.9):
        self.image_copy = np.array(self.image_buffer.top())
        if len(self.image_copy.shape) == 3:
            self.image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
        if win % 2 == 0: win = win - 1
        # 边界的均值有点麻烦
        # 这里分别计算和和邻居数再相除
        kern = np.ones([win, win])
        sums = signal.correlate2d(self.image_copy, kern, 'same')
        cnts = signal.correlate2d(np.ones_like(self.image_copy), kern, 'same')
        means = sums // cnts
        # 如果直接采用均值作为阈值，背景会变花
        # 但是相邻背景颜色相差不大
        # 所以乘个系数把它们过滤掉
        self.image_copy = np.where(self.image_copy < means * beta, 0, 255)

        self.image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_GRAY2BGR)
        self.image_buffer.push(np.array(self.image_copy))

    def gaussian_blur(self):
        self.image_copy = np.array(self.image_buffer.top())
        kernelSize = (15, 15)
        self.image_copy = cv2.GaussianBlur(self.image_copy, kernelSize, 0)
        self.image_buffer.push(np.array(self.image_copy))

    def filter(self):
        self.image_copy = np.array(self.image_buffer.top())
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        self.image_copy = cv2.filter2D(self.image_copy, -1, kernel=kernel)
        self.image_buffer.push(np.array(self.image_copy))

    def roll_back(self):
        self.image_buffer.back()

    def roll_forward(self):
        self.image_buffer.forward()
    
    def set_agv(self, agv):
        self.avg_gray_val = agv
        self.dp.agv = agv
    

if __name__ == '__main__':
    ip = imageprocesser()
    ip.create_image(500, 500)
