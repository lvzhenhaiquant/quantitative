import pandas as pd
from typing import List, Optional
import warnings
import json
import os
import clickhouse_connect
from datetime import date

warnings.filterwarnings('ignore')

class ClickHouseDB:
    def __init__(self):
        self.config_path = './config.json'
        config = self._load_config()
        # 配置默认值，防止config为空
        self.host = config.get('host', 'localhost') if config else 'localhost'
        self.port = config.get('port', 8123) if config else 8123
        self.database = config.get('database', 'quantitative') if config else 'quantitative'
        self.username = config.get('username', 'default') if config else 'default'
        self.password = config.get('password', '') if config else ''
        self.client = None
        self.connect_flag = self.connect()
        
    def _load_config(self):      
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                # 获取clickhouse部分的配置
                if 'clickhouse' in config_data:
                    return config_data['clickhouse']
                else:
                    return config_data
        except FileNotFoundError:
            print(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return {}
        except json.JSONDecodeError:
            print(f"配置文件 {self.config_path} 格式错误，使用默认配置")
            return {}
    
    def connect(self):
        """连接ClickHouse，确保数据库存在"""
        try:
            # 先连接到默认数据库创建目标库
            temp_client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password
            )
            temp_client.command(f"CREATE DATABASE IF NOT EXISTS `{self.database}`")
            print(f"确保数据库 `{self.database}` 存在")
            temp_client.close()
            
            # 连接到目标数据库
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                database=self.database,
                username=self.username,
                password=self.password
            )
            print(f"成功连接到ClickHouse数据库: {self.host}:{self.port}, 数据库: {self.database}")
            return True
        except Exception as e:
            print(f"连接ClickHouse数据库失败: {str(e)}")
            return False

    def save_df_to_clickhouse(self, 
                             df: pd.DataFrame, 
                             table_name: str, 
                             primary_keys: Optional[List[str]] = None, 
                             operation_type: str = 'replace'):
        """
        将DataFrame写入ClickHouse，指定主键作为排序键
        
        Args:
            df: 要写入的DataFrame
            table_name: 目标表名
            primary_keys: 主键（排序键）列表，如['ts_code', 'trade_date']
            operation_type: 操作类型: replace(替换表) / append(追加数据) / update(更新数据)
        
        Returns:
            bool: 写入是否成功
        """
        if self.client is None:
            print("数据库未连接，无法写入")
            return False
        
        try:
            # 检查表是否存在
            tables = self.client.query(f"SHOW TABLES LIKE '{table_name}'").result_rows
            table_exists = len(tables) > 0
            
            # 处理排序键：将列表转为 ClickHouse 支持的格式（如 ts_code, trade_date）
            order_by_clause = ""
            if primary_keys and len(primary_keys) > 0:
                order_by_clause = ", ".join([f"`{pk}`" for pk in primary_keys])
            else:
                order_by_clause = "tuple()"  # 无主键时使用空元组
            
            if operation_type == 'replace':
                # 替换模式：删除旧表，创建新表
                if table_exists:
                    self.client.command(f"DROP TABLE IF EXISTS `{table_name}`")
                # 推断表结构并创建
                schema = self._infer_schema_from_dataframe(df)
                create_sql = f"""
                CREATE TABLE `{table_name}` ({schema}) 
                ENGINE = MergeTree() 
                ORDER BY ({order_by_clause})
                """
                self.client.command(create_sql)
                print(f"表 `{table_name}` 创建成功（替换模式）")
            
            elif operation_type == 'append':
                # 追加模式：表不存在则创建
                if not table_exists:
                    schema = self._infer_schema_from_dataframe(df)
                    create_sql = f"""
                    CREATE TABLE `{table_name}` ({schema}) 
                    ENGINE = MergeTree() 
                    ORDER BY ({order_by_clause})
                    """
                    self.client.command(create_sql)
                    print(f"表 `{table_name}` 创建成功（追加模式）")
            
            elif operation_type == 'update':
                # 更新模式：先删后插（基于主键）
                if not table_exists:
                    schema = self._infer_schema_from_dataframe(df)
                    create_sql = f"""
                    CREATE TABLE `{table_name}` ({schema}) 
                    ENGINE = MergeTree() 
                    ORDER BY ({order_by_clause})
                    """
                    self.client.command(create_sql)
                    print(f"表 `{table_name}` 创建成功（更新模式）")
                
                # 按主键删除现有数据
                if primary_keys and len(primary_keys) > 0:
                    # print(f"按主键 {primary_keys} 删除待更新的记录...")
                    # 批量构造删除条件（避免逐行删除，提升效率）
                    pk_values = df[primary_keys].drop_duplicates()
                    delete_conditions = []
                    for _, row in pk_values.iterrows():
                        cond = []
                        for pk in primary_keys:
                            val = row[pk]
                            if isinstance(val, pd.Timestamp) or isinstance(val, str):
                                escaped_val = str(val).replace("'", "''")
                                cond.append(f"`{pk}` = '{escaped_val}'")
                            else:
                                cond.append(f"`{pk}` = '{val}'")
                        delete_conditions.append(f"({' AND '.join(cond)})")
                    
                    if delete_conditions:
                        delete_sql = f"ALTER TABLE `{table_name}` DELETE WHERE {' OR '.join(delete_conditions)}"
                        self.client.command(delete_sql)
            
            # 预处理数据：修复日期格式，避免类型错误
            processed_df = df.copy()
            for col in processed_df.columns:
                # 检查是否包含datetime.date对象
                if processed_df[col].apply(lambda x: isinstance(x, date)).any():
                    # 将datetime.date对象转换为字符串格式
                    processed_df[col] = processed_df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if isinstance(x, date) else x)
                # 日期类型转换为 ClickHouse DateTime64 支持的格式
                elif pd.api.types.is_datetime64_any_dtype(processed_df[col]):
                    # 保留完整的日期时间信息，转换为适合DateTime64的字符串格式
                    processed_df[col] = processed_df[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]  # 保留毫秒精度
                # 处理NaN/NaT：ClickHouse不支持，转为对应类型的默认值
                elif pd.api.types.is_float_dtype(processed_df[col]):
                    processed_df[col] = processed_df[col].fillna(0.0)
                elif pd.api.types.is_integer_dtype(processed_df[col]):
                    processed_df[col] = processed_df[col].fillna(0)
                elif pd.api.types.is_string_dtype(processed_df[col]):
                    processed_df[col] = processed_df[col].fillna('')
            
            # 高效写入数据（使用insert_df，自动适配类型）
            self.client.insert_df(table_name, processed_df)
            # print(f"成功插入 {len(processed_df)} 行数据到表 `{table_name}`，操作类型: {operation_type}")
            return True
        
        except Exception as e:
            print(f"操作失败: {str(e)}")
            return False
        finally:
            # 注意：不要在这里关闭连接！否则后续无法复用连接
            # self.client.close() 
            pass

    def _infer_schema_from_dataframe(self, df: pd.DataFrame) -> str:
        """
        从DataFrame推断ClickHouse表结构，适配各类数据类型
        
        Returns:
            str: 列定义字符串，如 "`ts_code` String, `trade_date` Date"
        """
        if df.empty:
            raise ValueError("DataFrame不能为空，无法推断表结构")
        
        schema_parts = []
        for col_name in df.columns:
            col_dtype = df[col_name].dtype
            col_data = df[col_name]
            
            # 精准映射pandas类型到ClickHouse类型
            if pd.api.types.is_integer_dtype(col_dtype):
                # 根据数值范围选择整数类型，避免溢出
                min_val = col_data.min() if not col_data.isna().all() else 0
                max_val = col_data.max() if not col_data.isna().all() else 0
                if pd.notna(min_val) and pd.notna(max_val):
                    if min_val >= 0:
                        if max_val <= 255:
                            ch_type = "UInt8"
                        elif max_val <= 65535:
                            ch_type = "UInt16"
                        elif max_val <= 4294967295:
                            ch_type = "UInt32"
                        else:
                            ch_type = "UInt64"
                    else:
                        if min_val >= -128 and max_val <= 127:
                            ch_type = "Int8"
                        elif min_val >= -32768 and max_val <= 32767:
                            ch_type = "Int16"
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            ch_type = "Int32"
                        else:
                            ch_type = "Int64"
                else:
                    ch_type = "Int64"
            
            elif pd.api.types.is_float_dtype(col_dtype):
                ch_type = "Float64"  # 统一使用Float64，避免精度丢失
            
            elif pd.api.types.is_bool_dtype(col_dtype):
                ch_type = "UInt8"  # ClickHouse推荐用UInt8存储布尔值
            
            elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                ch_type = "DateTime64(3)"  # 使用DateTime64(3)保留毫秒精度，而不是Date
            
            elif pd.api.types.is_timedelta64_dtype(col_dtype):
                ch_type = "Int64"  # 时间差转为毫秒数
            
            else:
                # 字符串/对象类型统一用String
                ch_type = "String"
            
            schema_parts.append(f"`{col_name}` {ch_type}")
        
        return ", ".join(schema_parts)
    
    def get_table_date_range(self, table_name, date_column='trade_date'):
            """获取表中数据的日期范围"""
            try:
                query = f"""
                    SELECT 
                        MIN({date_column}) as min_date, 
                        MAX({date_column}) as max_date 
                    FROM {table_name}
                """
                result = self.client.query(query)
                df = result.df()
                
                if df.empty or df.iloc[0]['min_date'] is None:
                    return None, None
                else:
                    min_date = df.iloc[0]['min_date']
                    max_date = df.iloc[0]['max_date']
                    return min_date, max_date
            except Exception as e:
                print(f"获取表日期范围失败: {e}")
                return None, None
        
    def table_exists(self, table_name):
            """检查表是否存在"""
            try:
                query = f"EXISTS TABLE {table_name}"
                result = self.client.query(query)
                df = result.df()
                return df.iloc[0][0] == 1
            except Exception as e:
                print(f"检查表存在性失败: {e}")
                return False

    def close(self):
        """手动关闭数据库连接"""
        if self.client:
            self.client.close()
            print("数据库连接已关闭")



