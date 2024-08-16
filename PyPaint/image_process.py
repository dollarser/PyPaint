""" 电器设备标注相关方法 """
import cv2
import itertools
import numpy as np
import os
from typing import Tuple
from utils import cv_imread, cv_imwrite


class Rectangle:
    def __init__(self, x, y, height, width):
        self.x = round(x)
        self.y = round(y)
        self.height = round(height)
        self.width = round(width)
    
    @property
    def tl(self):
        return self.x, self.y

    @property
    def br(self):
        return self.x + self.height, self.y + self.width
    

class ImageProcessor:
    def __init__(self):
        self.split_result = np.zeros((6, 24), dtype=object)
        self.boundries = np.zeros((7,25,4)) # (x, y, height, width)
        self.markers = []
        self.emarkers = []
        self.min_width = 1
        self.small_output_dir = r'defualt'
        self.big_output_dir = r'defualt'
        self.current_pic_filename = ''
        self.current_pic_path = r''
        self.agv = 127

    def row_histogram(self, pic):
        return np.mean(pic, axis=1, dtype=np.float32)

    def col_histogram(self, pic):
        return np.mean(pic, axis=0, dtype=np.float32)

    def average_gray_value(self, pic):
        return np.mean(pic, dtype=np.float32)

    def get_big_img(self, pic):
        big_img = np.zeros(shape=(pic.shape[0] * 3, pic.shape[1] * 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                big_img[i * pic.shape[0] : (i+1) * pic.shape[0], j * pic.shape[1] : (j+1) * pic.shape[1]] = pic
        return big_img

    def remove_black_edge(self, pic, margin=10):
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
        _, pic_bin = cv2.threshold(pic, self.agv, 255, cv2.THRESH_BINARY)
        row_hist = (self.row_histogram(pic_bin) >= 190).astype(np.int32)
        col_hist = (self.col_histogram(pic_bin) >= 120).astype(np.int32) 

        left = find_consective_zero(col_hist, 300)
        right = len(col_hist) - 1 - find_consective_zero(col_hist[::-1], 300)
        top = find_consective_zero(row_hist, 500) 
        bottom = len(row_hist) - 1 - find_consective_zero(row_hist[::-1], 500)

        left, right, top, bottom = self.crop((left, right, top, bottom), (92, len(col_hist) - 1 - 92, 70, len(row_hist) - 1 - 198), margin)

        return pic[top:bottom, left:right], pic_bin[top:bottom, left:right]
    
    def crop(self, matrix: Tuple[int, int, int, int], diff_matrix: Tuple[int, int, int, int], margin: int) -> Tuple[int, int, int, int]:
        left, right, top, bottom = matrix
        diff_left, diff_right, diff_top, diff_bottom = diff_matrix
        return (max(diff_left, left - margin), 
            min(diff_right, right + margin),
            max(diff_top, top - margin),
            min(diff_bottom, bottom + margin)
            )
        
        
    def split_recursively(self, pic, row=6, col=24, output_pics=False, output_big_pics=False):
        self.markers = []
        print(self.agv)
        _, pic_bin = cv2.threshold(pic, self.agv, 255, cv2.THRESH_BINARY)
        self.find_split_line(pic, pic_bin, col, row, 0, 0, 0, 0, min_width=self.min_width, output_pics=output_pics)
        width, height = pic.shape
        self.boundries[:, -1, 1] = width - 1
        self.boundries[-1,:,0] = height - 1       
        if output_big_pics:
            self.write_big_pic(pic, pic_bin)
        return self.split_result


    def find_split_line(self, pic, pic_bin, horizontal_blocks_amount, vertical_blocks_amount, left_pos, top_pos, left_pixel_pos, top_pixel_pos, min_width=5, filename_prefix='default', folder_name='default', output_pics=False):
        def find_widest_split_line(row_hist, col_hist):
            row_max_width = 0
            row_cur_width = 0
            col_max_width = 0
            col_cur_width = 0
            row_e, col_e = -1, -1
            # group by
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

        # print(top_pos, left_pos, vertical_blocks_amount, horizontal_blocks_amount)
        if horizontal_blocks_amount < 2 and vertical_blocks_amount < 2:
            # print("single pic output..")
            self.boundries[top_pos][left_pos] = (top_pixel_pos, left_pixel_pos, *pic.shape)
            self.split_result[top_pos][left_pos] = pic
            if output_pics:
                self.write_small_pic(pic, pic_bin, left_pos, top_pos, filename_prefix=filename_prefix, folder_name=folder_name)
            return
        
        row_hist = np.where(self.row_histogram(pic_bin) < 200, 0, 1)
        col_hist = np.where(self.col_histogram(pic_bin) < 120, 0, 1)
        isCol, end, width = find_widest_split_line(row_hist, col_hist)
        start = end - min_width
        end = end - width + 1 + min_width
        _hba_1 = round(horizontal_blocks_amount * end / len(col_hist))
        _hba_2 = round(horizontal_blocks_amount * (len(col_hist) - start) / len(col_hist))
        _vba_1 = round(vertical_blocks_amount * end / len(row_hist))
        _vba_2 = round(vertical_blocks_amount * (len(row_hist) - start) / len(row_hist))
        # print(f'line width {width}')

        if width > min_width:
            if isCol:
                self.markers.append(Rectangle(end + left_pixel_pos - min_width, top_pixel_pos, len(row_hist), width))
            else:
                self.markers.append(Rectangle(left_pixel_pos, end + top_pixel_pos - min_width, width, len(col_hist)))
        if width < min_width:
            # print("split equally")
            self.split_equally(pic, pic_bin, horizontal_blocks_amount, vertical_blocks_amount, left_pos, top_pos, left_pixel_pos, top_pixel_pos, filename_prefix=filename_prefix, folder_name=folder_name, output_pics=output_pics)
        elif isCol:
            # print("divide vertical to", f'({vertical_blocks_amount},{_hba_1}) and ({vertical_blocks_amount},{_hba_2})')
            self.find_split_line(pic[:, 0:end], pic_bin[:, 0:end], _hba_1, vertical_blocks_amount, left_pos, top_pos, left_pixel_pos, top_pixel_pos, filename_prefix=filename_prefix, folder_name=folder_name, output_pics=output_pics)
            self.find_split_line(pic[:, start:], pic_bin[:, start:], _hba_2, vertical_blocks_amount, left_pos+_hba_1, top_pos, left_pixel_pos+start, top_pixel_pos, filename_prefix=filename_prefix, folder_name=folder_name, output_pics=output_pics)
        else:
            # print("divide horizontal to", f'({_vba_1},{horizontal_blocks_amount}) and ({_vba_2},{horizontal_blocks_amount})')
            self.find_split_line(pic[0:end, :], pic_bin[0:end, :], horizontal_blocks_amount, _vba_1, left_pos, top_pos, left_pixel_pos, top_pixel_pos, filename_prefix=filename_prefix, folder_name=folder_name, output_pics=output_pics)
            self.find_split_line(pic[start:, :], pic_bin[start:, :], horizontal_blocks_amount, _vba_2, left_pos, top_pos+_vba_1, left_pixel_pos, top_pixel_pos+start, filename_prefix=filename_prefix, folder_name=folder_name, output_pics=output_pics)
        

    def split_equally(self, pic, pic_bin, horizontal_blocks_amount, vertical_blocks_amount, left_pos, top_pos, left_pixel_pos, top_pixel_pos, filename_prefix='default', folder_name='default', output_pics=False):
        row_width = len(self.col_histogram(pic_bin))
        col_width = len(self.row_histogram(pic_bin))
        row_step = round(row_width / horizontal_blocks_amount)
        col_step = round(col_width / vertical_blocks_amount)

        for i, j in itertools.product(range(horizontal_blocks_amount), range(vertical_blocks_amount)): 
            left = max(0, round(i * row_step))
            right = min(row_width-1, round((i + 1) * row_step))
            top = max(0, round(j * col_step))
            bottom = min(col_width-1, round((j+1) * col_step))
            self.split_result[top_pos+j][left_pos+i] = pic[top:bottom, left:right]
            self.boundries[top_pos+j][left_pos+i][0] = top_pixel_pos+top
            self.boundries[top_pos+j][left_pos+i][1] = left_pixel_pos+left
            self.boundries[top_pos+j][left_pos+i][2] = col_step
            self.boundries[top_pos+j][left_pos+i][3] = row_step
            
            if output_pics:
                self.write_small_pic(pic[top:bottom, left:right], pic_bin[top:bottom, left:right], left_pos + i, top_pos + j, filename_prefix=filename_prefix, folder_name=folder_name)
        
        one_fourth_width = self.min_width // 4
        for i in range(left_pixel_pos + row_step - one_fourth_width, left_pixel_pos + horizontal_blocks_amount * row_step - one_fourth_width, row_step):
            self.emarkers.append(Rectangle(i, top_pixel_pos, col_width, 2 * one_fourth_width))
        for i in range(top_pixel_pos + col_step - one_fourth_width, top_pixel_pos + vertical_blocks_amount * col_step - one_fourth_width, col_step):
            self.emarkers.append(Rectangle(left_pixel_pos, i, 2 * one_fourth_width, row_width))
        

    def write_small_pic(self, pic, pic_bin, left_pos, top_pos, filename_prefix='default', folder_name='default'):
        if not os.path.exists(os.path.dirname(os.getcwd() + f'\\{folder_name}\\')):
            os.makedirs(os.path.dirname(os.getcwd() + f'\\{folder_name}\\'))
        cv_imwrite(pic, filepath=f'{folder_name}/{filename_prefix}_{top_pos}_{left_pos}.jpg')

    def write_big_pic(self, pic, pic_bin, filename_prefix='default_big', folder_name='default_big'):
        if not os.path.exists(os.path.dirname(os.getcwd() + f'\\{folder_name}\\')):
            os.makedirs(os.path.dirname(os.getcwd() + f'\\{folder_name}\\'))
        pic = cv2.cvtColor(pic_bin, cv2.COLOR_GRAY2BGR)
        rgb_green=(144,238,144)
        rgb_red=(51, 0, 255)
        for marker in self.markers:
            cv2.rectangle(
                pic,
                marker.tl,
                marker.br,
                rgb_green,
                -1
            )
        for marker in self.emarkers:
            cv2.rectangle(
                pic,
                marker.tl,
                marker.br,
                rgb_red,
                -1
            )
        cv_imwrite(pic, filepath=f'{folder_name}/{filename_prefix}.jpg')
