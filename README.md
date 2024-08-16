## python画图板

### 介绍

本项目基于opencv-python，支持基本的图像处理和常用的图像算法

+ 图片拼接
+ 画笔
+ 几何绘制
+ 高斯模糊
+ 边缘检测
+ 锐化

### 界面

目前界面比较简陋, 持续优化中...

![image](https://github.com/user-attachments/assets/13b67789-1f43-40fe-84a0-c404789ae765)


### 环境配置

**配置python环境**：

    没有python环境可无脑安装miniconda, 官网：https://docs.anaconda.com/miniconda/

    直接下载：https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

**安装依赖**：

配置python环境后, 在项目目录下cmd命令行执行命令：

```bash
pip install -r requirement.txt
```
### 运行

在项目目录下使用命令行: 

```bash
python main.py
```

## 其他（待更新）

### 使用pywebview版本GUI界面

文档：https://pywebview.flowrl.com/guide/

```bash
pip install pywebview
```

### 打包发布

打包程序会将虚拟环境中的所有依赖都打包进去，因此需要新建虚拟环境，只安装必要的包

文档：https://pyinstaller.org/en/stable/usage.html

```bash
pip install pyinstaller
pyinstaller --onefile main.py
pyinstaller -F -w -i 图标.ico main.py
```