"""
配置管理模块
提供统一的配置加载和访问接口
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    """配置类，用于加载和管理 config.yaml 中的参数"""
    
    _instance = None  # 单例
    _config_data: Dict[str, Any] = {}
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: str = "config.yaml") -> 'Config':
        """
        加载 YAML 配置文件
        
        Args:
            config_path: 配置文件路径（相对于工作目录）
        
        Returns:
            Config 实例
        """
        instance = cls()
        
        # 尝试多个位置查找配置文件
        possible_paths = [
            config_path,
            os.path.join(os.path.dirname(__file__), "..", config_path),
        ]
        
        config_file = None
        for path in possible_paths:
            if os.path.exists(path):
                config_file = path
                break
        
        if config_file is None:
            raise FileNotFoundError(
                f"配置文件未找到。已尝试的路径: {possible_paths}"
            )
        
        with open(config_file, 'r', encoding='utf-8') as f:
            instance._config_data = yaml.safe_load(f) or {}
        
        print(f"✓ 配置已加载: {config_file}")
        return instance
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键（用点号分隔）
        
        Args:
            key: 配置键，如 "paths.model_path" 或 "training.learning_rate"
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_path(self, key: str) -> str:
        """
        获取路径配置值，并确保目录存在
        
        Args:
            key: 路径键，如 "paths.model_path"
        
        Returns:
            路径字符串
        """
        path = self.get(key)
        if path is None:
            raise KeyError(f"路径配置 '{key}' 不存在")
        return str(path)
    
    def get_list(self, key: str, default: list = None) -> list:
        """
        获取列表配置值
        
        Args:
            key: 配置键
            default: 默认值
        
        Returns:
            列表
        """
        value = self.get(key, default)
        if not isinstance(value, list):
            return default or []
        return value
    
    def get_int(self, key: str, default: int = None) -> int:
        """获取整数配置值"""
        value = self.get(key, default)
        if value is None:
            return default
        return int(value)
    
    def get_float(self, key: str, default: float = None) -> float:
        """获取浮点数配置值"""
        value = self.get(key, default)
        if value is None:
            return default
        return float(value)

    # === 【新增】 ===
    def get_str(self, key: str, default: str = None) -> str:
        """获取字符串配置值"""
        value = self.get(key, default)
        if value is None:
            return default
        return str(value)
    # ==============
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔配置值"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value) if value is not None else default
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问: config["paths.model_path"]"""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """获取配置的完整字典"""
        return self._config_data.copy()
    
    def __repr__(self) -> str:
        return f"Config({self._config_data})"


def init_config(config_path: str = "config.yaml") -> Config:
    """初始化并返回配置实例"""
    return Config.load(config_path)


def get_config() -> Config:
    """获取全局配置实例"""
    return Config()