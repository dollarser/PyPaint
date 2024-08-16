from typing import Union, Dict
from pathlib import Path
from types import SimpleNamespace
import sys
from utils import yaml_load, colorstr


ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
ASSETS = ROOT / "assets"  # default images
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"

DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)

# Define keys for arg type checks
CFG_FLOAT_KEYS = {  # integer or float arguments, i.e. x=2 and x=2.0
    "warmup_epochs",
}
CFG_FRACTION_KEYS = {  # fractional float arguments with 0.0<=values<=1.0
    "dropout",
}
CFG_INT_KEYS = {  # integer-only arguments
    "epochs",
}
CFG_BOOL_KEYS = {  # boolean-only arguments
    "save",
}


class IterableSimpleNamespace(SimpleNamespace):
    """IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified
            'default.yaml' file.
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def cfg2dict(cfg):
    """
    Converts a configuration object to a dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration object to be converted. Can be a file path,
            a string, a dictionary, or a SimpleNamespace object.

    Returns:
        (Dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict('config.yaml')

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1='value1', param2='value2')
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({'param1': 'value1', 'param2': 'value2'})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg

def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data source. Can be a file path, dictionary, or
            SimpleNamespace object.
        overrides (Dict | None): Dictionary containing key-value pairs to override the base configuration.

    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.

    Examples:
        >>> from ultralytics.cfg import get_cfg
        >>> config = get_cfg()  # Load default configuration
        >>> config = get_cfg('path/to/config.yaml', overrides={'epochs': 50, 'batch_size': 16})

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        # 如果默认参数中没有save_dir，自定义参数的save_dir也去掉
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # 修改一些特殊的参数 project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = cfg.get("model", "").split(".")[0]

    # Type and Value checks
    check_cfg(cfg)

    # Return instance
    return IterableSimpleNamespace(**cfg)

def check_cfg(cfg, hard=True):
    """
    检查配置参数类型和值是否满足要求
    
    该函数验证参数类型和值，确保参数正确或者可以转换为正确的值。需要验证的值保存在全局遍历CFG_FLOAT_KEYS,
    CFG_FRACTION_KEYS, CFG_INT_KEYS, and CFG_BOOL_KEYS中。
    
    Args:
        cfg (Dict): 待验证的配置字典
        hard (bool): 如果是True, 类型和值不匹配抛出异常，如果是False先尝试进行类型转换

    Examples:
        >>> config = {
        ...     'epochs': 50,     # valid integer
        ...     'lr0': 0.01,      # valid float
        ...     'momentum': 1.2,  # invalid float (out of 0.0-1.0 range)
        ...     'save': 'true',   # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # corrected 'save' key

    Notes:
        - The function modifies the input dictionary in-place.
        - None values are ignored as they may be from optional arguments.
        - Fraction keys are checked to be within the range [0.0, 1.0].
    """
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. " f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. " f"'{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    检查自定义配置和基本配置字典之间的对齐情况，为不匹配的键提供错误消息。

    Args:
        base (Dict): 包含有效key的基本配置
        custom (Dict): 包含待验证key的自定义配置
        e (Exception | None): Optional error instance passed by the calling function.

    Raises:
        SystemExit: 如果配置参数不匹配，系统退出

    Examples:
        >>> base_cfg = {'epochs': 50, 'lr0': 0.01, 'batch_size': 16}
        >>> custom_cfg = {'epoch': 100, 'lr': 0.02, 'batch_size': 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     print("Mismatched keys found")
    """
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string) from e