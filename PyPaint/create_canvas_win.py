import tkinter as tk
from tkinter.messagebox import *
from tkinter import *


class create_canvas_win():
    """ 创建新画布窗口 """
    def __init__(self, father):
        self.father = father
        self.top_win = tk.Toplevel()
        x = self.father.root.winfo_x()
        y = self.father.root.winfo_y()
        self.top_win.geometry('250x180+'+str(x+100)+'+'+str(y+100))
        self.top_win.grab_set()
        self.top_win.resizable(0,0)
        self.init_ui()
        self.top_win.bind('<Return>',func=self.ok)
        self.top_win.bind('<Escape>',func=self.cancel)
        self.top_win.mainloop()

    def init_ui(self):
        rely1,rely2,rely3 = 0.2,0.45,0.75
        relx1,relx2,relx3 = 0.2,0.5,0.75
        self.width_value = StringVar()
        self.height_value = StringVar()
        font = '14'
        Label(self.top_win,text='宽：',font=font).place(rely=rely1,relx=relx1,anchor='c')
        self.width_input = Entry(self.top_win,highlightcolor='yellow',width=8,textvariable=self.width_value)
        self.width_input.place(rely=rely1,relx=relx2,anchor='c')
        Label(self.top_win,text='px',font=font).place(rely=rely1,relx=relx3,anchor='c')
        Label(self.top_win,text='高：',font=font).place(rely=rely2,relx=relx1,anchor='c')
        self.height_input = Entry(self.top_win,highlightcolor='yellow',width=8,textvariable=self.height_value)
        self.height_input.place(rely=rely2,relx=relx2,anchor='c')
        Label(self.top_win,text='px',font=font).place(rely=rely2,relx=relx3,anchor='c')

        self.ok_btn = Button(self.top_win,text='确定',font='12',command=self.ok)
        self.ok_btn.place(rely=rely3,relx=(relx1+relx2)/2,anchor='c')
        self.cancel_btn = Button(self.top_win,text='取消',font='12',command=self.cancel)
        self.cancel_btn.place(rely=rely3,relx=(relx2+relx3)/2,anchor='c')
        
        self.width_input.focus_set()

    def keypress_adaptor(self,fun,**kwds):
        return lambda event,fun=fun,kwds = kwds:fun(event,**kwds)

    def keypress(self,event:tk.Event, text:StringVar):
        textcheck = ''.join(i for i in event.widget.get() if i in '0123456789')
        text.set(int(textcheck))

    def ok(self,event = None):
        width = self.width_input.get()
        height = self.height_input.get()
        if (self.check(width) and self.check(height)):
            self.father.create_canvas(width = int(width), height = int(height))
            self.top_win.destroy()
        else:
            if(not self.check(width)):
                self.width_value.set('')
            if(not self.check(height)):
                self.height_value.set('')
            showinfo('值错误','请输入规范的数字')
    
    def cancel(self, event = None):
        self.top_win.destroy()

    def check(self, text):
        if text == '':
            return False
        for c in text:
            if c not in '0123456789':
                return False
        if int(text) == 0:
            return False
        return True
