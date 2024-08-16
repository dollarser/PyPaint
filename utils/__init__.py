from pathlib import Path
import re
import yaml
import os
import numpy as np
import cv2


def imwrite(file_name, image):
    # 获取文件扩展名
    ext = os.path.splitext(file_name)[1]
    # 使用 cv2.imencode 将图像编码为指定格式
    img_encoded = cv2.imencode(ext, image)[1]
    img_encoded.tofile(file_name)

def imread(img_path):
    # 中文路径使用numpy读取
    img_data = np.fromfile(img_path, dtype=np.uint8)
    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    return image


def cv_imread(filepath, flags = cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),flags=flags)

def cv_imwrite(pic, filepath, ext='.jpg'):
    cv2.imencode(ext, pic)[1].tofile(filepath)


def listdir(root_path, ext='jpg'):
    all_files = []
    files = os.listdir(root_path)
    for file in files:
        if not os.path.isdir(os.path.join(root_path, file)):   # not a dir
            if file.split('.')[-1] == ext:
                all_files.append(os.path.join(root_path, file))
        else:  # is a dir
            child_files = listdir(os.path.join(root_path, file), ext)
            all_files.extend(child_files)
    return all_files


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def remove_colorstr(input_string):
    """
    Removes ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr('blue', 'bold', 'hello world'))
        >>> 'hello world'
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)

