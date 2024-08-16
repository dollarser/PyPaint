""" 事件绑定函数 用来响应鼠标操作和快捷键 """
from PyPaint.model import Point
from PyPaint.globval import *
from tkinter import Event, Canvas
from PyPaint.imageprocesser import ImageProcesser


def motion_straight_line(event: Event, old_line, start_point):
    event.widget.coords(old_line, start_point.x, start_point.y, event.x,event.y)

def release_straight_line(event: Event, window, start_point):
    event.widget.unbind('<B1-Motion>')
    event.widget.unbind('<ButtonRelease-1>')
    ip: ImageProcesser = window.ip
    ip.draw_line(start_point.x, start_point.y, event.x, event.y, thickness=get_value('thickness'))
    window.canvas_view_update()

def down_straight_line(event: Event, window):
    # print('line down')
    start_point = Point(event.x, event.y)
    widget = event.widget
    old_line = widget.create_line(event.x, event.y, event.x+1, event.y+1,fill = get_value('color'))
    widget.bind('<B1-Motion>', adaptor(motion_straight_line, start_point=start_point, old_line=old_line))
    widget.bind('<ButtonRelease-1>', adaptor(release_straight_line, window=window, start_point=start_point))

def motion_rectangle(event: Event, old_rec, start_point):
    event.widget.coords(old_rec,start_point.x,start_point.y,event.x,event.y)

def release_rectangle(event: Event, window, start_point):
    event.widget.unbind('<B1-Motion>')
    event.widget.unbind('<ButtonRelease-1>')
    ip: ImageProcesser = window.ip
    ip.draw_rectangle(start_point.x, start_point.y, event.x, event.y, thickness=get_value('thickness'))
    window.canvas_view_update()
    
def down_rectangle(event: Event, window):
    start_point = Point(event.x, event.y)
    widget = event.widget
    old_rec = widget.create_rectangle(event.x,event.y,event.x+1,event.y+1,outline = get_value('color'))
    widget.bind('<B1-Motion>', adaptor(motion_rectangle, start_point = start_point, old_rec=old_rec))
    widget.bind('<ButtonRelease-1>', adaptor(release_rectangle, window=window, start_point=start_point))

def motion_pencil(event: Event, window):
    """ 画笔移动事件 """
    canvas: Canvas = event.widget
    x, y = event.x, event.y
    x1, y1 = get_value('oldx'), get_value('oldy')
    ip: ImageProcesser = window.ip
    if x1 != None and y1 != None:
        x1, y1 = get_value('oldx'), get_value('oldy')
        canvas.create_line(x, y, x1, y1, fill=get_value('color'), width=5, smooth=True)
        ip.draw_pencil(x, y, x1, y1, thickness=get_value('thickness'))
    set_value('oldx', x)
    set_value('oldy', y)

def release_pencil(event: Event, window):
    event.widget.unbind('<B1-Motion>')
    event.widget.unbind('<ButtonRelease-1>')
    window.ip.draw_pencil_complete()
    window.canvas_view_update()
    
def down_pencil(event: Event, window):
    set_value('oldx', event.x)
    set_value('oldy', event.y)
    canvas: Canvas = event.widget
    window.ip.clear_image_copy()
    canvas.bind('<B1-Motion>', adaptor(motion_pencil, window=window))
    canvas.bind('<ButtonRelease-1>', adaptor(release_pencil, window=window))

def motion_eraser(event: Event, window):
    canvas = event.widget
    eraser_thickness = get_value('thickness')
    canvas.create_rectangle(event.x-eraser_thickness, event.y-eraser_thickness, event.x+eraser_thickness, event.y+eraser_thickness, fill="white", outline="white")
    ip: ImageProcesser = window.ip
    ip.draw_eraser(event.x-eraser_thickness, event.y-eraser_thickness, event.x+eraser_thickness, event.y+eraser_thickness)

def release_eraser(event: Event, window):
    event.widget.unbind('<B1-Motion>')
    event.widget.unbind('<ButtonRelease-1>')
    window.ip.draw_eraser_complete()
    window.canvas_view_update()
    
def down_eraser(event: Event, window):
    canvas: Canvas = event.widget
    window.ip.clear_image_copy()
    canvas.bind('<B1-Motion>', adaptor(motion_eraser, window=window))
    canvas.bind('<ButtonRelease-1>', adaptor(release_eraser, window=window))

def motion_put_pic(event: Event, pic_id):
    canvas = event.widget
    canvas.coords(pic_id, event.x, event.y)

def down_put_pic(event: Event, bind_id,window,image):
    event.widget.unbind('<Motion>', bind_id)
    event.widget.unbind('<Button-1>')
    window.ip.insert_image(event.x, event.y, image)
    window.canvas_view_update()

def down_EL_select(event: Event, window):
    x = event.x
    y = event.y
    window.ip.draw_EL_select(x,y)
    window.canvas_view_update()

def motion_set_coords(event):
    set_value('x', event.x)
    set_value('y', event.y)

def adaptor(fun,**kwds):
    return lambda event, fun=fun, kwds = kwds:fun(event, **kwds)
