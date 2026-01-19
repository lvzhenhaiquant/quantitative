import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.filter import NameDFilter, ExpressionDFilter
from utils.data_manager import DataManager

class DataProcessor:
    """数据处理器，用于读取和处理QLib本地数据集"""
    
    def __init__(self, qlib_data_dir="/home/yunbo/project/quantitative/qlib_data/cn_data"):
        self.qlib_data_dir = qlib_data_dir
        self.RL_Data_dir= os.path.join("./RL_Data")
        self.DataManager = DataManager()
        
        self.fields_origin = ['open', 'high', 'low', 'close']
        self.qlib_fields = ["$" + field for field in  self.fields_origin]

        qlib.init(provider_uri=qlib_data_dir, region='cn')
        self.filter_list = [
            NameDFilter(name_rule_re="^(SH60|SZ00)"),  # 主板股票
            ExpressionDFilter(rule_expression="$volume>0")  # 剔除无成交量的记录
            ]
    
    def load_data(self, csi_code, start_date, end_date):
        file_name_data = os.path.join(self.RL_Data_dir,f"{csi_code}_{start_date}_{end_date}.csv")
        if os.path.exists(file_name_data):
            pd_df = pd.read_csv(file_name_data)
        else:
            pl_data = self.DataManager.load(
                    stocks=csi_code,  # DataManager会自动解析为指数成分股
                    start=start_date,
                    end=end_date,
                    fields=self.fields_origin,  # 使用您定义的fields
                    adjust=True  # 使用后复权价格
                )
            if pl_data.is_empty():
                print(f"警告: DataManager获取数据为空")
            pd_df  = pl_data.to_pandas()
            Path(self.RL_Data_dir).mkdir(parents=True, exist_ok=True)
            pd_df.to_csv(file_name_data, index=False, encoding='utf-8-sig')
        return pd_df
        
    
    def load_data_qlib(self, csi_code, start_date, end_date):
        Qlib_fields = ["$" + field for field in self.qlib_fields]
        file_name_data = os.path.join(self.RL_Data_dir,f"{csi_code}_{start_date}_{end_date}.csv")
        if os.path.exists(file_name_data):
            data = pd.read_csv(file_name_data)
        else:
            instruments = D.instruments(csi_code,filter_pipe=self.filter_list)
            data = D.features(instruments=instruments,fields=Qlib_fields,start_time=start_date,end_time=end_date)
            data.to_csv(file_name_data, index=False, encoding='utf-8-sig')
        return data
    
    def _preprocess_data(self, data):
        """
        预处理数据
        
        参数:
            data: 原始数据
            
        返回:
            预处理后的数据
        """
        # 处理缺失值
        # data = data.fillna(method='ffill')  # 前向填充
        data = data.fillna(method='bfill')  # 后向填充
        
        # 处理无穷值
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        return data
    def normalize_min_max(self, data, fields=None):
        """
        对数据进行最大最小标准化
        
        参数:
            data: 原始数据
            fields: 需要标准化的字段列表，默认为所有价格字段
            
        返回:
            标准化后的数据
        """
        if fields is None:
            fields = self.fields_origin
            
        normalized_data = data.copy()
        
        # 按日期分组进行标准化
        for field in fields:
            if field in normalized_data.columns:
                # 按日期分组
                for date, group in normalized_data.groupby('date'):
                    min_val = group[field].min()
                    max_val = group[field].max()
                    
                    # 避免除以零
                    if max_val != min_val:
                        normalized_data.loc[normalized_data['date'] == date, field] = \
                            (group[field] - min_val) / (max_val - min_val)
                    else:
                        normalized_data.loc[normalized_data['date'] == date, field] = 0.0
        
        return normalized_data
    
# 修改utils/DataLoad.py文件中的calculate_daily_volatility方法，使用百分比变化

    def calculate_daily_volatility(self, data, fields=None):
        """
        使用百分比变化计算波动率并替换原始字段值
        
        参数:
            data: 原始数据
            fields: 需要计算波动率的字段列表，默认为所有非date和stock字段
            
        返回:
            百分比变化数据替换后的新数据框
        """
        volatility_data = data.copy()
        
        # 获取所有非date和stock的字段
        if fields is None:
            fields = [col for col in volatility_data.columns if col not in ['date', 'stock']]
        
        # 按股票分组计算百分比变化
        for stock, stock_data in volatility_data.groupby('stock'):
            # 对每个字段计算百分比变化
            for field in fields:
                if field in volatility_data.columns:
                    # 计算该股票的百分比变化
                    # 使用pct_change()计算每日变化率
                    pct_change = stock_data[field].pct_change()
                    
                    # 将百分比变化值替换回原数据框
                    volatility_data.loc[stock_data.index, field] = pct_change
        
        # 处理NaN值（第一天的变化率）
        volatility_data = volatility_data.fillna(0.0)  # 第一天的变化率设为0
        
        return volatility_data
    
    def get_pool_stocks_by_date(self, pool, date):
        """
        获取指定日期的指数成分股
        
        参数:
            pool: 指数名称（如"csi1000"）
            date: 日期（格式为"YYYY-MM-DD"）
            
        返回:
            该日期的指数成分股列表
        """
        return self.DataManager.get_pool_stocks_by_date(pool=pool, date=date)
    
    def create_dataset(self, instruments, start_date="2015-01-01", end_date="2023-12-31", 
                      test_start_date="2022-01-01", horizon=20):
        """
        创建用于训练和测试的数据集
        
        参数:
            instruments: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            test_start_date: 测试集开始日期
            horizon: 预测周期
            
        返回:
            训练数据集和测试数据集
        """
        # 定义数据处理配置
        handler_config = {
            "start_time": start_date,
            "end_time": end_date,
            "fit_start_time": start_date,
            "fit_end_time": test_start_date,
            "instruments": instruments,
            "fields": {
                "feature": self.qlib_fields,
                "label": ["$close"],
            },
            "processors": {
                "feature": [
                    {"class": "StandardScaler"},
                ],
                "label": [
                    {"class": "RobustZScoreNorm"},
                ]
            },
            "learn_processors": [
                {"class": "DropnaLabel"},
            ],
        }
        
        # 创建数据处理器
        handler = DataHandlerLP(**handler_config)
        
        # 创建数据集
        dataset = DatasetH(handler)
        
        # 分割训练集和测试集
        train_set, test_set = dataset.split(test_start_date=test_start_date)
        
        return train_set, test_set
    
    def get_field_data(self, data, field):
        """
        获取指定字段的数据
        
        参数:
            data: 完整数据集
            field: 字段名
            
        返回:
            指定字段的数据
        """
        if field in data.columns:
            return data[field]
        else:
            print(f"字段{field}不存在")
            return None
    
    def calculate_return(self, data, field="CLOSE", horizon=1):
        """
        计算指定字段的收益率
        
        参数:
            data: 数据集
            field: 字段名
            horizon: 周期
            
        返回:
            收益率数据
        """
        if field not in data.columns:
            print(f"字段{field}不存在")
            return None
            
        return data[field].unstack().pct_change(horizon).stack()
    
    def get_rolling_data(self, data, field, window=5):
        """
        获取滚动窗口数据
        
        参数:
            data: 数据集
            field: 字段名
            window: 窗口大小
            
        返回:
            滚动窗口数据
        """
        if field not in data.columns:
            print(f"字段{field}不存在")
            return None
            
        return data[field].unstack().rolling(window=window).mean().stack()
