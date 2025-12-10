"""
日志工具模块
提供统一的日志配置和管理
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class LoggerManager:
    """
    日志管理器

    功能：
    1. 同时输出到控制台和文件
    2. 按日期自动切分日志文件
    3. 按模块分类日志
    4. 支持日志级别配置
    """

    _loggers = {}  # 缓存已创建的logger

    @classmethod
    def get_logger(cls,
                   name: str,
                   log_dir: str = None,
                   level: int = logging.INFO,
                   console: bool = True,
                   file: bool = True) -> logging.Logger:
        """
        获取Logger实例

        Parameters
        ----------
        name : str
            Logger名称（通常是模块名）
        log_dir : str, optional
            日志文件目录，默认为 logs/
        level : int, optional
            日志级别，默认INFO
        console : bool, optional
            是否输出到控制台，默认True
        file : bool, optional
            是否输出到文件，默认True

        Returns
        -------
        logging.Logger
            配置好的Logger实例
        """
        # 如果logger已存在，直接返回
        if name in cls._loggers:
            return cls._loggers[name]

        # 创建logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # 避免重复输出

        # 清除已有的handler（避免重复添加）
        logger.handlers.clear()

        # 日志格式
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. 控制台Handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 2. 文件Handler
        if file:
            # 默认日志目录
            if log_dir is None:
                log_dir = "/home/zhenhai1/quantitative/logs"

            # 确保目录存在
            os.makedirs(log_dir, exist_ok=True)

            # 生成日志文件名：模块名_日期.log
            today = datetime.now().strftime('%Y%m%d')
            log_file = os.path.join(log_dir, f"{name}_{today}.log")

            # 创建文件Handler（按大小切分）
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,           # 保留5个备份
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 缓存logger
        cls._loggers[name] = logger

        return logger

    @classmethod
    def set_level(cls, name: str, level: int):
        """设置指定logger的日志级别"""
        if name in cls._loggers:
            cls._loggers[name].setLevel(level)

    @classmethod
    def clear_handlers(cls, name: str):
        """清除指定logger的所有handler"""
        if name in cls._loggers:
            cls._loggers[name].handlers.clear()


# 便捷函数
def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    获取Logger的便捷函数

    使用示例：
    >>> from utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Hello World")
    """
    return LoggerManager.get_logger(name, **kwargs)


# 日志查看工具
class LogViewer:
    """日志查看工具"""

    @staticmethod
    def list_logs(log_dir: str = "/home/zhenhai1/quantitative/logs"):
        """列出所有日志文件"""
        if not os.path.exists(log_dir):
            print(f"日志目录不存在: {log_dir}")
            return []

        log_files = []
        for file in os.listdir(log_dir):
            if file.endswith('.log'):
                filepath = os.path.join(log_dir, file)
                size = os.path.getsize(filepath) / 1024  # KB
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                log_files.append({
                    'name': file,
                    'path': filepath,
                    'size': f"{size:.2f}KB",
                    'modified': mtime.strftime('%Y-%m-%d %H:%M:%S')
                })

        # 按修改时间排序
        log_files.sort(key=lambda x: x['modified'], reverse=True)

        print("=" * 80)
        print("日志文件列表")
        print("=" * 80)
        for i, log in enumerate(log_files, 1):
            print(f"{i}. {log['name']}")
            print(f"   路径: {log['path']}")
            print(f"   大小: {log['size']}")
            print(f"   修改时间: {log['modified']}")
            print()

        return log_files

    @staticmethod
    def tail_log(log_file: str, lines: int = 50):
        """查看日志文件的最后N行"""
        if not os.path.exists(log_file):
            print(f"日志文件不存在: {log_file}")
            return

        print("=" * 80)
        print(f"日志文件: {log_file}")
        print(f"最后 {lines} 行")
        print("=" * 80)

        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            for line in last_lines:
                print(line.rstrip())

    @staticmethod
    def search_log(log_file: str, keyword: str):
        """在日志文件中搜索关键词"""
        if not os.path.exists(log_file):
            print(f"日志文件不存在: {log_file}")
            return

        print("=" * 80)
        print(f"搜索: {keyword}")
        print(f"文件: {log_file}")
        print("=" * 80)

        with open(log_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if keyword.lower() in line.lower():
                    print(f"[行{i}] {line.rstrip()}")

    @staticmethod
    def view_today_log(module: str = None):
        """查看今天的日志"""
        log_dir = "/home/zhenhai1/quantitative/logs"
        today = datetime.now().strftime('%Y%m%d')

        if module:
            log_file = os.path.join(log_dir, f"{module}_{today}.log")
            if os.path.exists(log_file):
                LogViewer.tail_log(log_file, lines=100)
            else:
                print(f"未找到今天的日志: {log_file}")
        else:
            # 显示所有今天的日志
            print(f"今天的日志文件 ({today}):")
            for file in os.listdir(log_dir):
                if today in file and file.endswith('.log'):
                    filepath = os.path.join(log_dir, file)
                    print(f"\n{'='*80}")
                    LogViewer.tail_log(filepath, lines=20)


if __name__ == "__main__":
    # 测试日志系统
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            # 列出所有日志
            LogViewer.list_logs()

        elif command == "tail" and len(sys.argv) > 2:
            # 查看日志尾部
            log_file = sys.argv[2]
            lines = int(sys.argv[3]) if len(sys.argv) > 3 else 50
            LogViewer.tail_log(log_file, lines)

        elif command == "search" and len(sys.argv) > 3:
            # 搜索日志
            log_file = sys.argv[2]
            keyword = sys.argv[3]
            LogViewer.search_log(log_file, keyword)

        elif command == "today":
            # 查看今天的日志
            module = sys.argv[2] if len(sys.argv) > 2 else None
            LogViewer.view_today_log(module)

        else:
            print("未知命令")

    else:
        # 测试日志功能
        logger = get_logger("test_module")
        logger.info("这是一条测试日志")
        logger.warning("这是警告信息")
        logger.error("这是错误信息")
        print("\n日志已保存到 logs/ 目录")
        LogViewer.list_logs()