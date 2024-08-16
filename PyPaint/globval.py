""" 保存一些全局变量 """
from PyPaint.model import Stack

stack = Stack()
stack.clear()

_global_dict = {}
_global_dict['color'] = 'black'
_global_dict['color_rgb'] = (0,0,0)
_global_dict['roll_back'] = Stack()
_global_dict['x'] = 0
_global_dict['y'] = 0
_global_dict['thickness'] = 5

def set_value(name, value):
    global _global_dict
    _global_dict[name] = value

def get_value(name):
    try:
        return _global_dict[name]
    except KeyError:
        return None

def info():
    print(_global_dict)
