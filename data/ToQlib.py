import hashlib
from datetime import timedelta, datetime
from pathlib import Path
import pandas as pd
import os
import subprocess
import sys
import qlib
from tqdm import tqdm


class DataToQlib:
    def __init__(self, 
        qlib_data_dir_path=r"/home/zhenhai1/quantitative/qlib_data/cn_data",
        dump_bin_py_path=r"/home/zhenhai1/quantitative/venv/lib/python3.10/site-packages/qlib/scripts/dump_bin.py"
    ):
        self.save_basic_folder_path = './download_data/basic'
        self.save_adj_folder_path = './download_data/adj'
        self.save_balancesheet_folder_path = './download_data/balancesheet'
        self.save_income_folder_path = './download_data/income'
        self.save_fina_indicator_folder_path = './download_data/fina_indicator'
        self.save_basic_30min_folder_path = './download_data/basic_mins'
        self.save_daily_folder_path = './download_data/daily'
        self.save_index_path = './download_data/origin_download_index'
        self.save_index_daily_folder_path = './download_data/index_daily'

        self.qlib_index_instrument_path=f'{qlib_data_dir_path}/instruments'

        self.qlib_data_dir = qlib_data_dir_path #qlib数据存放文件夹
        self.dump_bin_py = dump_bin_py_path #dump_bin.py路径

        self.code_to_csi = {
            "000016.SH": "csi50",  # 上证50
            "000300.SH": "csi300",  # 沪深300
            "000688.SH": "csi50kechuang",  # 科创50
            "000852.SH": "csi1000",  # 中证1000
            "000905.SH": "csi500",  # 中证500
            "000906.SH": "csi800",  # 中证800
            "000985.SH": "all",  # 中证全指
            "399006.SZ": "csigg",  # 创业板指
        }

        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'

        #self.orgin_download_csv_path = './download_data/orgin_download_data/all_returns_adjusted_20220101_20251114.csv' #存储所有的股票csv

        # #每个表的字段
        # self.basic_fields = [] 自动读取所有列
        # self.balancesheet_fields = []
        # self.income_fields = []
        # self.fina_indicator_fields = []

    # -----------2222 方法二，单线程------------
    # 单线程版本，易理解
    def start_to_qlib_single_thread(self):
        self.process(self.save_index_daily_folder_path, ['ts_code', 'trade_date'], 'day', '1、指数日线行情')#放前面，不然会覆盖
        # self.process(self.save_balancesheet_folder_path ,['ts_code', 'ann_date'],'6mon','2、负债')
        # self.process(self.save_income_folder_path,['ts_code', 'ann_date'],'6mon','3、利润')  # ← 已完成
        # self.process(self.save_fina_indicator_folder_path, ['ts_code', 'ann_date'],'6mon','4、财务')
        # self.process(self.save_basic_folder_path, ['ts_code', 'trade_date'], 'day','5、基础行情')  # ← 已转换,注释掉避免重复
        # self.process(self.save_adj_folder_path, ['ts_code', 'trade_date'], 'day','6、复权因子')  # ← 已转换,注释掉避免重复
        # self.process(self.save_daily_folder_path, ['ts_code', 'trade_date'], 'day', '7、日线行情')  # ← 已转换,注释掉避免重复
        # self.process(self.save_basic_30min_folder_path, ['ts_code', 'time'], '30min','5、30分钟级别数据')


        print("全部 process 执行完毕")

    def process(self, csv_folder_path, required_columns, frequency, desc_str="qlib 转换"):
        # 使用pathlib处理路径
        folder_path = Path(csv_folder_path)
        if not folder_path.exists():
            print(f"错误：文件夹不存在 - {folder_path}")
            return
        if not folder_path.is_dir():
            print(f"错误：需要传入目录路径，而不是文件路径 - {folder_path}")
            return
        csv_file_list = list(folder_path.glob("*.parquet"))
        if len(csv_file_list) == 0:
            print(f"找到 {len(csv_file_list)} 个股票数据文件,已退出")
            return
        print(f"{desc_str}找到 {len(csv_file_list)} 个股票数据文件,{csv_folder_path}开始处理...")
        # 从第一个文件中检测可转换字段（假设所有文件结构相同）
        try:
            sample_file = csv_file_list[0]
            df_sample = pd.read_parquet(sample_file,engine='pyarrow')
            all_columns = list(df_sample.columns)
            convertible_fields = []
            for col in all_columns:
                try:
                    # 尝试将列转换为数值类型
                    pd.to_numeric(df_sample[col].dropna().head(1000))
                    convertible_fields.append(col)
                except (ValueError, TypeError):
                    # 无法转换为数值，跳过
                    continue
            # 排除必需的列（如股票代码、日期）
            convertible_fields = list(set(convertible_fields) - set(required_columns))
            fields_str = ",".join(convertible_fields) if convertible_fields else ""
            if not fields_str:
                print("错误：没有检测到可用的数值字段")
                return
        except Exception as e:
            print(f"读取样本文件出错: {e}")
            return
        # 构建命令 - 现在传入目录路径而不是单个文件
        try:
            qlib_data_dir_path = Path(self.qlib_data_dir)
            dump_bin_py_path = Path(self.dump_bin_py)
            if not dump_bin_py_path.exists():
                print(f"错误：dump_bin.py脚本不存在 - {dump_bin_py_path}")
                return
            qlib_dir = Path(qlib.__file__).parent.parent if 'qlib' in sys.modules else Path(".")
            # "dump_all": DumpDataAll, "dump_fix": DumpDataFix, "dump_update": DumpDataUpdate
            cmd = [
                sys.executable,
                str(dump_bin_py_path.absolute()),
                "dump_all",  # 使用dump_all处理整个目录
                "--data_path", str(folder_path.absolute()),  # 传入目录路径
                "--qlib_dir", str(qlib_data_dir_path.absolute()),
                "--freq", frequency,
                "--date_field_name", required_columns[1],
                "--symbol_field_name", required_columns[0],
                "--include_fields", fields_str,
                "--file_suffix", ".parquet"
            ]
            print(f"执行命令: {' '.join(cmd)}")
            # 执行转换命令
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=qlib_dir,encoding='utf-8', errors='ignore')
            if result.returncode == 0:
                print("转换成功！")
                result = result.stdout.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
                if result.strip():
                    print("输出信息:", result)
            else:
                print("转换失败！")
                result = result.stderr.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
                print("错误信息:",result)
        except Exception as e:
            print(f"处理目录 {csv_folder_path} 时出错: {e}")

    def start_toqlib_index(self):
        code_to_csi = self.code_to_csi
        if not os.path.exists(self.save_index_path):
            print(f"指数Parquet目录不存在：{self.save_index_path}")
            return
        os.makedirs(self.qlib_index_instrument_path, exist_ok=True)
        index_files_list = list(Path(self.save_index_path).glob('index_basic*.parquet'))
        if not index_files_list:
            print(f"{self.save_index_path} 下无parquet文件")
            return
        # 遍历处理每个指数文件
        for index_file in  tqdm(index_files_list, desc='qlib指数转换'):
            # 1. 解析文件名：提取前9位指数代码（如000300.SH）
            file_name = index_file.name
            index_code = file_name[len('index_basic_'):len("index_basic_0000_0000")] if len(file_name) >= len("index_basic_0000_0000") else ""

            # 检查是否有CSI命名映射
            if index_code not in code_to_csi:
                print(f"\n文件{file_name}的指数代码{index_code}无CSI命名映射，跳过")
                continue
            csi_name = code_to_csi[index_code]  # 直接获取CSI命名（如csi300）
            # 2. 读取并预处理Parquet数据
            result_df = pd.read_parquet(index_file, engine='pyarrow')
            if result_df.empty:
                print(f"\n{index_code}（{csi_name}）文件{file_name}无数据，跳过")
                continue
            if 'con_code' not in result_df.columns or 'trade_date' not in result_df.columns:
                print(f"\n{index_code}（{csi_name}）文件{file_name}缺失con_code/trade_date列，跳过")
                continue

            # 统一日期格式，过滤无效数据
            result_df['trade_date'] = pd.to_datetime(result_df['trade_date'], format='%Y%m%d', errors='coerce')
            result_df = result_df.dropna(subset=['trade_date', 'con_code'])
            if result_df.empty:
                print(f"\n{index_code}（{csi_name}）无有效trade_date/con_code数据，跳过")
                continue

            # 3. 对比成分股集合，获取变化日期
            sorted_dates = sorted(result_df['trade_date'].unique())
            if not sorted_dates:
                print(f"\n{index_code}（{csi_name}）无有效日期，跳过")
                continue
            change_dates = [sorted_dates[0]]
            prev_stocks = set(result_df[result_df['trade_date'] == sorted_dates[0]]['con_code'])
            for curr_date in sorted_dates[1:]:
                curr_stocks = set(result_df[result_df['trade_date'] == curr_date]['con_code'])
                if curr_stocks != prev_stocks:
                    change_dates.append(curr_date)
                    prev_stocks = curr_stocks
            change_dates = pd.Series(change_dates).sort_values()

            # 4. 生成start_date/end_date + 格式化股票代码
            total_end = result_df['trade_date'].max() #选择0
            # total_end = datetime.now()  # 选择1：最后一个end_date设为今天
            final_result = []
            for i in range(len(change_dates)):
                start_date = change_dates.iloc[i].strftime('%Y-%m-%d')
                # 计算结束日期
                end_date = total_end.strftime('%Y-%m-%d') if i == len(change_dates) - 1 else (change_dates.iloc[i + 1] - timedelta(days=1)).strftime('%Y-%m-%d')
                # 格式化股票代码（如000001.SZ → SZ000001）
                curr_stocks = result_df[result_df['trade_date'] == change_dates.iloc[i]]['con_code'].unique()
                for con_code in curr_stocks:
                    if '.' in con_code:
                        code_part, market_part = con_code.split('.', 1)
                        format_code = f"{market_part}{code_part.zfill(6)}"
                    else:
                        format_code = con_code
                    final_result.append({'code': format_code,'start_date': start_date,'end_date': end_date})

            # 5. 去重并保存文件
            final_df = pd.DataFrame(final_result).drop_duplicates()
            if final_df.empty:
                print(f"\n{index_code}（{csi_name}）无有效最终数据，跳过保存")
                continue
            # TXT文件：QLib路径，命名=csiXXX.txt（如csi300.txt）
            txt_path = os.path.join(self.qlib_index_instrument_path, f"{csi_name}.txt")
            final_df.to_csv(txt_path, index=False, header=False, sep='\t', encoding='utf-8')
            #print(f"{csi_name}转换成功")
