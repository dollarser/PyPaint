import os
import cv2
import sys
import math
import numpy as np
from numpy.lib.function_base import select
from PIL import ImageTk, Image

import tkinter as tk
from tkinter.filedialog import SaveFileDialog
import tkinter.colorchooser
import tkinter.filedialog
from tkinter.messagebox import *
from tkinter import Event, FALSE, Menu, Checkbutton, Frame, Label, Scrollbar, Canvas

sys.path.insert(0, os.getcwd())
# print(sys.path)
from PyPaint.globval import *
from PyPaint.bind_fun import *
from PyPaint.create_canvas_win import create_canvas_win
from PyPaint.imageprocesser import ImageProcesser
from cfg import get_cfg, ASSETS
from utils import imwrite, imread, listdir


class Window():
    def __init__(self, args=None):
        self.args = get_cfg(overrides=args)
        self.root = tk.Tk() 
        self.SCREEN_WIDTH = self.root.winfo_screenwidth()
        self.SCREEN_HEIGHT = self.root.winfo_screenheight()
        self.root.geometry('800x600+'
            +str(int((self.SCREEN_WIDTH-800)/2))+'+'+str(int((self.SCREEN_HEIGHT-600)/2)))
        self.root.option_add('*tearOff', FALSE)
        self.canvas = None
        # 背景绑定函数
        self.root.bind('<Motion>', motion_set_coords)
        self.init_ui()
        # 图像处理类
        self.ip = ImageProcesser(1000, 100, self.args)
        # 当前展示图像索引，防止被回收
        self.current_image = None
        self.avg_gray_val = 127
        # 主循环
        self.root.mainloop()


    def init_ui(self):
        """ 
        软件ui界面绘制
        """
        self.root['background'] = 'gainsboro'
        # 设置窗口图标
        icon = self.readTkImage(os.path.join(ASSETS, self.args.icon))
        self.root.iconphoto(False, icon)  # False表示使用默认大小
        self.root.title("画图")

        # 菜单栏
        self.menubar = Menu(self.root)
        self.root['menu'] = self.menubar
        self.menu_file = Menu(self.menubar)
        self.menu_tools = Menu(self.menubar)
        self.menu_edit = Menu(self.menubar)
        self.menu_img =  Menu(self.menubar)
        self.menubar.add_cascade(menu=self.menu_file, label='文件')
        self.menubar.add_cascade(menu=self.menu_tools, label='绘制工具')
        self.menubar.add_cascade(menu=self.menu_edit, label='编辑')
        self.menubar.add_cascade(menu=self.menu_img, label = '图像算法')

        self.menu_file.add_command(label='新建空白图片', command=self.create_canvas_dialog)
        self.menu_file.add_command(label='打开图片', command=self.load_picture)
        self.menu_file.add_separator()
        self.menu_file.add_command(label='插入图片', command = self.insert_picture)
        self.menu_file.add_separator()
        self.menu_file.add_command(label='保存',command = self.save_img)
        
        self.menu_file.add_separator()
        self.menu_file.add_command(label='EL_select', command=self.EL_select)


        self.menu_tools.add_radiobutton(label='铅笔', command = lambda : self.set_tools_mode(mode='pencil'))
        self.menu_tools.add_radiobutton(label='直线', command = lambda : self.set_tools_mode(mode='line'))
        self.menu_tools.add_radiobutton(label='矩形', command = lambda : self.set_tools_mode(mode='rectangle'))
        self.menu_tools.add_radiobutton(label='橡皮擦', command = lambda : self.set_tools_mode(mode='eraser'))
        self.menu_tools.add_separator()
        self.menu_tools.add_command(label='选择颜色',command = self.set_color)

        self.menu_edit.add_command(label='撤销',command = self.roll_back)
        self.menu_edit.add_command(label='重做',command = self.roll_forward)
        self.menu_edit.add_command(label='重置放缩比例',command = self.reset_scale)
        self.menu_edit.add_separator()
        self.menu_edit.add_command(label='清屏',command = self.clear_screen)

        self.menu_img.add_command(label="图像灰度化", command=self.showGray)
        self.menu_img.add_command(label="自适应二值化", command=self.showAdaptiveThreshold)
        self.menu_img.add_command(label="模糊", command=self.showGaussianBlur)
        self.menu_img.add_command(label="锐化", command=self.showFilter)
        # 顶部导航栏容器
        self.top_nav_bar = Frame(self.root)
        self.top_nav_bar.pack(side=tk.TOP, expand=False, fill=tk.X)
        # 分割图片
        checkbuttom = Checkbutton(self.top_nav_bar, text='显示分割结果', command = lambda: self.set_split_result_show())
        checkbuttom.select()  # 默认选择状态
        checkbuttom.pack(side=tk.LEFT, fill=tk.X, anchor=tk.N)  # 设置放置位置
        
        # 设置画笔宽度
        init_thickness = get_value('thickness')
        self.thickness = tk.Scale(self.top_nav_bar, from_=1, to=255, orient=tk.HORIZONTAL, showvalue=False, length="200px", bg='white')
        self.thickness.set(init_thickness) 
        self.thickness.pack(side=tk.LEFT, fill=tk.X, anchor=tk.N, padx=(10, 0), pady=2)
        # 创建 Label 组件来显示当前值
        self.value_label = tk.Label(self.top_nav_bar, text=f"粗细: {init_thickness}", bg='white')
        self.value_label.pack(side=tk.LEFT, fill=tk.X, anchor=tk.N, padx=0, pady=2)
        # 设置 Scale 组件的命令，以便在滑动时更新 Label
        self.thickness.config(command=self.set_thickness)
        
        # 设置拖动条
        self.gray_value_set = tk.Scale(self.root, from_=0, to=255, length="200px")
        self.gray_value_set.set(127)
        # self.gray_value_set.pack(side=tk.LEFT, fill=tk.X)
        
        # 指示条
        self.flag_frame = Frame(self.root, height=10, background='#F2F2F7')
        self.flag_frame.pack(fill='x', side='top')
        # 颜色指示 在指示条内部
        self.color_flag = Frame(self.flag_frame, width=10, height = 10, background='#F2F2F7')
        self.color_flag.pack(side='left')
        # 放缩指示 在指示条内部
        self.scale_flag = Label(self.flag_frame, text='  ')
        self.scale_flag.pack(side='left')
        # 滚动条
        self.sb_v = Scrollbar(self.root, orient='vertical')
        self.sb_v.pack(side='right', fill='y')  # 放置滚动条
        self.sb_h = Scrollbar(self.root, orient='horizontal')
        self.sb_h.pack(side='bottom', fill='x')  # 放置滚动条
        # 滚动条绑定命令
        self.sb_v.config(command=self.canvas_view_scroll_bar_v)
        self.sb_h.config(command=self.canvas_view_scroll_bar_h)
        # 画布容器
        self.canvas_frame = Frame(self.root, background='#F2F2F7')
        self.canvas_frame.pack(side=tk.TOP, expand=True, fill='both')


    def create_canvas(self, width, height, image=None):
        """ 创建画布 """
        if self.canvas is not None:
            self.canvas.destroy()
        self.MINSCALE = self.args.MINSCALE  # 图像最小尺寸
        self.MAXSCALE = self.args.MAXSCALE  # 图像最大尺寸
        self.MAXZOOM = self.args.MAXZOOM  # 图像最大缩放比例
        win_w = self.canvas_frame.winfo_width() * 0.98
        win_h = self.canvas_frame.winfo_height() * 0.98

        self.canvas = Canvas(self.canvas_frame, bg='white')
        if width > win_w:
            scale_w = win_w/width
        else:
            scale_w = 1
        if height > win_h:
            scale_h = win_h/height
        else:
            scale_h = 1
        scale = min(scale_h, scale_w)
        display_width, display_height = math.floor(width*scale),math.floor(height*scale)
        self.canvas.config(width=display_width, height=display_height)
        self.canvas.place(relx=0.5, rely=0.5, anchor='c')
        self.canvas.update()
        x, y = self.canvas.winfo_x(), self.canvas.winfo_y()
        self.canvas.place(relx=0, rely=0, x=x, y=y, anchor='nw')
        # 更新滚动条
        self.update_scroll_bar(x, y, display_width, display_height, self.canvas_frame.winfo_width(), self.canvas_frame.winfo_height())
        self.root.focus_set()
        # 绑定快捷键
        self.root.bind('<Up>', func = self.canvas_view_scroll_key)
        self.root.bind('<Down>', func = self.canvas_view_scroll_key)
        self.root.bind('<Right>', func = self.canvas_view_scroll_key)
        self.root.bind('<Left>', func = self.canvas_view_scroll_key)
        self.root.bind('<Control-z>', func = self.roll_back)
        self.root.bind('<Control-Shift-Z>', func = self.roll_forward)
        self.root.bind('<Control-s>', func=self.save_img)
        self.root.bind('<Control-r>', func=self.reset_scale)
        # 鼠标滚轮
        self.canvas.bind('<MouseWheel>', self.canvas_scaling)
        # 鼠标中键
        self.canvas.bind("<Button-2>", self.on_middle_click)  # 鼠标中键按下
        self.canvas.bind("<B2-Motion>", self.on_middle_click_drag)  # 鼠标中键按住并移动

        
        # 初始化图像比例
        self.ip.set_ori_scale(width,height)
        self.ip.set_display_scale(display_width, display_height)
        # 更新放缩比例指示条
        self.scale_flag_update()

        if image is not None:
            self.ip.load_image(image)
        if image is None:
            self.ip.create_image(width,height)
        self.current_image = self.ip.get_dis_image()
        self.canvas.create_image(0, 0, anchor = 'nw', image = self.current_image)
    
    def readTkImage(self, path):
        # icon = PhotoImage(file=path)
        # 默认方法不支持中文使用opencv
        image = imread(path)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        icon = ImageTk.PhotoImage(pil_image)
        return icon
    
    def showGray(self):
        self.ip.gray()
        self.canvas_view_update()

    def showAdaptiveThreshold(self):
        self.ip.adaptive_threshold()
        # self.ip.binarization(10)
        self.canvas_view_update()

    def showGaussianBlur(self):
        self.ip.gaussian_blur()
        self.canvas_view_update()

    def showFilter(self):
        self.ip.filter()
        self.canvas_view_update()

    def check_canvas(self):
        if self.canvas == None:
            showwarning('提示','请先创建图像')
            return False
        return True
    
    def insert_picture(self):
        """ 
        插入图片用于编辑
        TODO: 插入时支持缩放
        """
        if(not self.check_canvas()):
            return
        file_path = tkinter.filedialog.askopenfilename(title=u'选择文件')
        if file_path == '':
            return
        image = imread(file_path)
        self.tmp_inseret_image = self.ip.convert_to_display_image(image)
        pic_id = self.canvas.create_image(get_value('x'), get_value('y'), anchor = 'c', image = self.tmp_inseret_image)
        
        bind_id = self.canvas.bind('<Motion>', adaptor(motion_put_pic, pic_id=pic_id))
        self.canvas.bind('<Button-1>', adaptor(down_put_pic, bind_id = bind_id, window = self, image=imread(file_path)))

    def set_split_result_show(self):
        if self.ip.show_split_result:
            self.ip.show_split_result = False
        else:
            self.ip.show_split_result = True
        self.ip.EL_select_init(self.ip.el_img_path, self.get_agv())

    def load_picture(self):
        file_path = tkinter.filedialog.askopenfilename(title=u'选择文件')
        if file_path == '':
            return
        image = imread(file_path)
        width,height = image.shape[1],image.shape[0]

        self.create_canvas(width,height,image)

    def save_img(self, event = None):
        if not self.check_canvas():
            return
        save_path = tkinter.filedialog.asksaveasfilename(title="Please select a filename:",filetypes=[("jpg文件", ".jpg")])
        if save_path == '':
            return
        if self.ip.save_img(save_path + '.jpg'):
            showinfo('提示','保存成功')

    def create_canvas_dialog(self):
        create_canvas_win(self)

    def canvas_view_update(self):
        self.canvas.delete(tk.ALL)
        self.current_image = self.ip.get_dis_image()
        self.canvas.create_image(0,0,image=self.current_image,anchor='nw')
    
    def on_middle_click(self, event):
        # 设置初始位置，用于记录鼠标按下时的位置
        self.initial_x, self.initial_y = event.x, event.y
    
    # 通过鼠标中键移动画布
    def on_middle_click_drag(self, event: Event):
        # 获取鼠标点击时的位置
        dx, dy = event.x - self.initial_x, event.y - self.initial_y
        # 移动画布
        self.canvas.move(0, dx, dy)
        # 重新绘制画布
        self.canvas.update()
        # 返回 True 以继续处理其他绑定
        return True

    # 滚动滚轮，画布放缩事件
    def canvas_scaling(self, event:Event):
        if self.avg_gray_val != self.get_agv():
            self.avg_gray_val = self.get_agv()
            self.ip.EL_select_init(self.ip.el_img_path, self.get_agv())
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        # 放大图片
        if event.delta > 0:
            new_w = math.floor(width * 1.1)
            new_h = math.floor(height * 1.1)
            max_scale = min(self.ip.ori_width * self.MAXZOOM, self.MAXSCALE)
            if new_w > new_h:
                if new_w > max_scale:
                    new_w = max_scale
                    new_h = math.floor(height * max_scale/width)
            else:
                if new_h > max_scale:
                    new_h = max_scale
                    new_w = math.floor(width * max_scale/height)
        else:
            new_w = math.floor(width*0.9)
            new_h = math.floor(height*0.9)
            if new_w < new_h:
                if new_w < self.MINSCALE:
                    new_w = self.MINSCALE
                    new_h = math.floor(height*self.MINSCALE/width)
            else:
                if new_h < self.MINSCALE:
                    new_h = self.MINSCALE
                    new_w = math.floor(width*self.MINSCALE/height)
        self.canvas.config(width=new_w, height=new_h)
        self.update_scroll_bar(self.canvas.winfo_x(), self.canvas.winfo_y(), new_w, new_h, self.canvas_frame.winfo_width(), self.canvas_frame.winfo_height())
        self.ip.set_display_scale(new_w, new_h)
        self.scale_flag_update()
        self.canvas_view_update()
        return False

    # 更新滚动条的位置和长短
    def update_scroll_bar(self, x, y, width, height, win_w, win_h):
        if width < win_w:
            self.sb_h.set(x/win_w, (x+width)/win_w)
        else:
            self.sb_h.set(-x/width, (win_w - x)/width)

        if height < win_h:
            self.sb_v.set(y/win_h, (y+height)/win_h)
        else:
            self.sb_v.set(-y/height, (-y+win_h)/height)

    # 通过滚动条控制画布的位置，竖直方向
    def canvas_view_scroll_bar_v(self, action, loc=0, size=None):
        
        # print(action, loc, size)
        # print( self.canvas.winfo_x(), self.canvas.winfo_y(), self.canvas.winfo_height(), self.canvas.winfo_y()/self.canvas.winfo_height())
        if self.canvas is None:
            return
        height = self.canvas.winfo_height()
        win_h = self.canvas_frame.winfo_height()
        
        if action == 'scroll':
            if height < win_h:
                loc = float(loc)
                loc = self.canvas.winfo_y() - loc/abs(loc) * height * 0.05
                loc = max(0, min(loc, win_h-height))
                self.canvas.place(relx = 0, rely = 0, x = self.canvas.winfo_x(), y = math.floor(loc), anchor='nw')
                self.sb_v.set(loc/win_h, (loc + height) / win_h)
            else:
                # print(self.canvas.winfo_y(), height, win_h, loc)
                loc = float(loc)
                loc = self.canvas.winfo_y() - loc/abs(loc) * height * 0.05
                loc = min(0, max(loc, win_h-height))
                self.canvas.place(relx=0, rely=0, x=self.canvas.winfo_x(), y=math.floor(loc), anchor='nw')
                self.sb_v.set(-loc/height, (win_h-loc)/height)
            return False

        elif action == 'moveto':
            loc = float(loc)
            print(self.canvas.winfo_y(), height, win_h, loc)
        
        if height < win_h:
            self.canvas.place(relx = 0, rely = 0, x = self.canvas.winfo_x(), y = math.floor(loc * win_h), anchor='nw')
            self.sb_v.set(loc, loc + height / win_h)
        else:
            self.canvas.place(relx = 0, rely = 0, x = self.canvas.winfo_x(), y = math.floor(-loc * height), anchor='nw')
            self.sb_v.set(loc, loc + win_h / height)
        
    # 通过滚动条控制画布的位置，水平方向
    def canvas_view_scroll_bar_h(self, action, loc):
        if self.canvas is None:
            return 
        loc = float(loc)
        width = self.canvas.winfo_width()
        win_w = self.canvas_frame.winfo_width()
        if width < win_w:
            self.canvas.place(relx=0, rely=0, x=math.floor(loc*win_w), y = self.canvas.winfo_y(), anchor='nw')
            self.sb_h.set(loc,loc+width/win_w)
        else:
            self.canvas.place(relx=0, rely=0, x=math.floor(-loc*width), y = self.canvas.winfo_y(), anchor='nw')
            self.sb_h.set(loc,loc+win_w/width)

    # 通过按键控制画布的位置 
    def canvas_view_scroll_key(self, event:Event):
        win_w = self.canvas_frame.winfo_width()
        win_h = self.canvas_frame.winfo_height()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < win_w:
            minx = 0
            maxx = win_w - w
        else:
            minx = win_w - w
            maxx = 0
        if h < win_h:
            miny = 0
            maxy = win_h - h
        else:
            miny = win_h - h
            maxy = 0
        x = self.canvas.winfo_x()
        y = self.canvas.winfo_y()
        step = 10
        new_x = x
        new_y = y
        if event.keysym == 'Up':
            new_y = max(y-step,miny)
        if event.keysym == 'Down':
            new_y = min(y+step,maxy)
        if event.keysym == 'Left':
            new_x = max(x-step,minx)
        if event.keysym == 'Right':
            new_x = min(x+step,maxx)
        self.canvas.place(relx=0, rely=0, x=new_x, y=new_y, anchor='nw')
        self.update_scroll_bar(new_x,new_y,w,h,win_w,win_h)

    def scale_flag_update(self):
        self.scale_flag.config(text=str(self.ip.get_scale()))

    def reset_scale(self, event=None):
        win_w = self.canvas_frame.winfo_width()
        win_h = self.canvas_frame.winfo_height()
        ori_width = self.ip.ori_width
        ori_height = self.ip.ori_height
        self.canvas.config(width=ori_width, height=ori_height)
        self.canvas.place(relx=0, rely=0, x=math.floor(win_w/2-ori_width/2), y=math.floor(win_h/2-ori_height/2), anchor='nw')
        self.canvas.update()
        self.update_scroll_bar(self.canvas.winfo_x(), self.canvas.winfo_y(), ori_width, ori_height, self.canvas_frame.winfo_width(), self.canvas_frame.winfo_height())
        self.ip.set_display_scale(ori_width, ori_height)
        self.scale_flag_update()
        self.canvas_view_update()

    def clear_screen(self):
        if(not self.check_canvas()):
            return
        self.ip.clear_screen()
        self.canvas_view_update()

    def set_tools_mode(self, mode):
        if(not self.check_canvas()):
            return
        if mode == 'line':
            # self.tool_flag.delete(tk.ALL)
            # self.tool_flag.create_line(0,0,self.tool_flag.winfo_width(),self.tool_flag.winfo_height(),fill='black')
            self.canvas.unbind('<Button-1>')
            self.canvas.unbind('<B1-Motion>')
            self.canvas.unbind('<ButtonRelease-1>')
            self.canvas.bind('<Button-1>', adaptor(down_straight_line, window=self))
            return
        if mode == 'rectangle':
            # self.tool_flag.delete(tk.ALL)
            # self.tool_flag.create_rectangle(3,3,self.tool_flag.winfo_width()-4,self.tool_flag.winfo_height()-6,outline = 'black')
            self.canvas.unbind('<Button-1>')
            self.canvas.unbind('<B1-Motion>')
            self.canvas.unbind('<ButtonRelease-1>')
            self.canvas.bind('<Button-1>', adaptor(down_rectangle, window=self))
            return 
        if mode == 'pencil':
            self.canvas.unbind('<Button-1>')
            self.canvas.unbind('<B1-Motion>')
            self.canvas.unbind('<ButtonRelease-1>')
            self.canvas.bind('<Button-1>', adaptor(down_pencil, window=self))
        if mode == 'eraser':
            self.canvas.unbind('<Button-1>')
            self.canvas.unbind('<B1-Motion>')
            self.canvas.unbind('<ButtonRelease-1>')
            self.canvas.bind('<Button-1>', adaptor(down_eraser, window=self))

    def get_agv(self):
        return self.gray_value_set.get()
    
    def set_thickness(self, value):
        self.value_label.config(text=f"粗细: {value}")
        set_value('thickness', int(value))
    
    def EL_select(self):
        agv = self.get_agv()
        self.ip.set_agv(agv)
        self.el_files_list = listdir(self.args.EL_SDIR)
        print(self.args.EL_ST_CNT, self.el_files_list[self.args.EL_ST_CNT-1])
        el_path = self.args.EL_SDIR + '/' + self.el_files_list[self.args.EL_ST_CNT-1]
        img = self.ip.dp.cv_imread(el_path)
        print(np.array(img).shape, 0) # read in (3499， 5800)
        img, _ = self.ip.dp.remove_black_edge(img)
        print(np.array(img).shape, 1) # remove edges (3230, 5167)
        width = np.array(img).shape[1]
        height = np.array(img).shape[0]
        self.ip.dp.split_recursively(img)
        print(np.array(img).shape, 2) # split
        self.ip.dp.boundries = self.ip.dp.boundries.astype(np.int32)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        print(np.array(img).shape, 3) # colored
        if self.ip.show_split_result:
            for i in range(6):
                for j in range(24):
                    cv2.rectangle(
                        img, 
                        (self.ip.dp.boundries[i][j][1], self.ip.dp.boundries[i][j][0]),
                        (self.ip.dp.boundries[i][j][1] + self.ip.dp.boundries[i][j][3], self.ip.dp.boundries[i][j][0] + self.ip.dp.boundries[i][j][2]),
                        (51+12*i,0+12*j,255-6*i-6*j),
                        10
                    )
        
        width,height = img.shape[1],img.shape[0]
        self.create_canvas(width,height,img)
        self.ip.EL_select_init(el_path, agv)
        
        self.canvas.unbind('<Button-1>')
        self.canvas.unbind('<B1-Motion>')
        self.canvas.unbind('<ButtonRelease-1>')
        self.canvas.bind('<Button-1>', adaptor(down_EL_select, window=self))
        # 函数返回时执行
        self.root.bind('<Return>', func=self.EL_select_complete)

    def EL_select_complete(self, event:Event):
        self.ip.EL_select_complete()
        self.args.EL_ST_CNT =  self.args.EL_ST_CNT + 1
        if self.args.EL_ST_CNT >= self.el_files_list:
            return 
        print(self.args.EL_ST_CNT, self.el_files_list[self.args.EL_ST_CNT])
        el_path = self.args.EL_SDIR+'/'+self.el_files_list[self.args.EL_ST_CNT]
        img = self.ip.dp.cv_imread(el_path)
        print(np.array(img).shape, 0) # read in (3499， 5800)
        img, _ = self.ip.dp.remove_black_edge(img)
        print(np.array(img).shape, 1) # remove edges (3230, 5167)
        width = np.array(img).shape[1]
        height = np.array(img).shape[0]
        self.ip.dp.split_recursively(img)
        print(np.array(img).shape, 2) # split
        self.ip.dp.boundries = self.ip.dp.boundries.astype(np.int32)

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        print(np.array(img).shape, 3) # colored
        if self.ip.show_split_result:
            for i in range(6):
                for j in range(24):
                    cv2.rectangle(
                        img, 
                        (self.ip.dp.boundries[i][j][1], self.ip.dp.boundries[i][j][0]),
                        (self.ip.dp.boundries[i][j][1] + self.ip.dp.boundries[i][j][3], self.ip.dp.boundries[i][j][0] + self.ip.dp.boundries[i][j][2]),
                        (51+12*i,0+12*j,255-6*i-6*j),
                        10
                    )
        width,height = img.shape[1],img.shape[0]
        self.create_canvas(width,height,img)
        self.canvas.bind('<Button-1>',adaptor(down_EL_select,window=self))
        self.ip.EL_select_init(el_path, self.get_agv())


    def set_color(self):
        if(not self.check_canvas()):
            return
        color = tkinter.colorchooser.askcolor(initialcolor=get_value('color'))
        if color[0] is not None:
            set_value('color',color[1])
            set_value('color_rgb',color[0])
            self.color_flag.configure(bg = get_value('color'))
    
    def roll_back(self,event = None):
        if(not self.check_canvas()):
            return
        self.ip.roll_back()
        self.canvas_view_update()
    
    def roll_forward(self,event = None):
        if(not self.check_canvas()):
            return
        self.ip.roll_forward()
        self.canvas_view_update()



if __name__ == '__main__':
    Window()
