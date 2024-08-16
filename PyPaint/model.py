""" 一些自定义类 """

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
class Action(object):
    def __init__(self, widget, item_type, item_id):
        self.widget = widget
        self.type = item_type
        self.item_id = item_id
        
class Stack():
    def __init__(self):
        self.data = []
        self.idx = -1

    def top(self):
        if self.idx == -1:
            return None
        else:
            return self.data[self.idx]
    
    def push(self,element):
        if len(self.data) == self.idx + 1:
            self.data.append(element)
            self.idx = self.idx + 1
        else:
            self.idx = self.idx +1
            self.data[self.idx] = element
        self.data = self.data[:self.idx+1]

    def pop(self):
        if self.idx != -1:
            self.idx = self.idx - 1
            return self.data[self.idx + 1]
        else:
            return None

    def empty(self):
        if self.idx == -1:
            return True
        else:
            return False
    
    def forward(self):
        if self.idx < len(self.data) - 1:
            self.idx = self.idx + 1

    def back(self):
        if self.idx > 0:
            self.idx = self.idx - 1

    def clear(self):
        self.idx = -1
        self.data.clear()
