import webview
import json
import os
import time
 
class API:
    """ 提供前端调用接口 """
    def __init__(self):
        self.file_path = 'note.json'
    def save_note(self, note):
        with open(self.file_path, 'w') as f:
            json.dump({'note': note}, f)
        return 'Note saved successfully!'
 
    def load_note(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                return data.get('note', '')
        return ''

def change_title(window):
    """changes title every 1 seconds"""
    for i in range(1, 10):
        time.sleep(1)
        window.title = f'New Title #{i}'
        print(window.title)

def main():
    api = API()
    # 配置web视图
    webview.settings = {
        'ALLOW_DOWNLOADS': False,
        'ALLOW_FILE_URLS': True,
        'OPEN_EXTERNAL_LINKS_IN_BROWSER': True,
        'OPEN_DEVTOOLS_IN_DEBUG': True
    }
    # 创建窗口
    window: webview.Window = webview.create_window(title='Simple Notepad', url='index.html', js_api=api)
    # 使用webview打开窗口
    webview.start(change_title, args=window, gui='edgechromium')
 
if __name__ == '__main__':
    main()
