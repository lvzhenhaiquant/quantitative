import datetime
import pandas as pd
import DownLoadData
import ToQlib
from datetime import timedelta
import datetime
import time


# 使用说明
# origin_down_load里面放的是原始下载数据
# 能够直接转换给qlib的股票，格式为SH000000（除了）

if __name__ == '__main__':
    start_date_str = '20150101'  # 过去5年：从2019年12月开始
    end_date_str = '20251217'    # 到2024年12月（今天）
    
    # 二、从Tushare拉取数据到本地处理
    token = 'a79f284e5d10967dacb6531a3c755a701ca79341ff0c60d59f1fcbf1'
    # 1、初始化
    DownLoadData = DownLoadData.DownloadDataFromTushare_Baostock(token)
    # 2、下载"中证1000"基础行情数据
    # DownLoadData.download_tushare_basic(start_date_str,end_date_str)  # 已完成
    # 3、下载"中证1000"财务数据
    # DownLoadData.download_tushare_finance(start_date_str,end_date_str)  # 已完成
    # 4、下载"中证1000"半小时级行情数据
    # DownLoadData.download_baostock_basic_mins(start_date_str,end_date_str)


    # 5、下载全A股基础行情数据
    # DownLoadData.download_tushare_A_basic(start_date_str,end_date_str)
    # 6、下载全A财务数据
    # DownLoadData.download_tushare_A_finance(start_date_str,end_date_str)
    # 7、下载全A股基础半小时级行情数据
    # DownLoadData.download_baostock_A_basic_mins(start_date_str, end_date_str)
    # 8、下载指数成分股
    # DownLoadData.download_index(start_date_str,end_date_str)
    # 9、下载指数日线行情
    # DownLoadData.download_index_daily(start_date_str,end_date_str)
    # 10、下载申万分类数据
    # DownLoadData.download_tushare_shenwan_classify()
    # 11、下载申万指数日线行情
    # DownLoadData.download_tushare_shenwan_daily(start_date_str,end_date_str)


    # 三、增量基础行情更新 (与2~4互斥使用)
    # 1、增量更新“中证1000”基础行情数据
    # DownLoadData.updates_tushare_basic(start_date_str,end_date_str)
    # 2、增量更新“中证1000”财务数据
    # DownLoadData.updates_tushare_finance(start_date_str,end_date_str)
    # 3、增量更新“中证1000”半小时级行情数据
    # DownLoadData.updates_baostock_basic_mins(start_date_str, end_date_str)
    

    # 4、增量下载全A基础行情数据
    # DownLoadData.updates_tushare_A_basic(start_date_str, end_date_str)
    # 5、增量下载全A财务数据
    # DownLoadData.updates_tushare_A_finance(start_date_str, end_date_str)
    # 6、增量更新全A半小时级数据
    # DownLoadData.updates_tushare_A_basic_mins(start_date_str, end_date_str)
    # 7、增量更新指数成分股
    # DownLoadData.update_index(start_date_str, end_date_str)
    # 8、增量更新指数日线行情
    # DownLoadData.update_index_daily(start_date_str, end_date_str)
    # 9、增量下载申万日线行情
    # start_date_str = '20150101'  # 过去5年：从2019年12月开始
    # end_date_str = '20201205'    # 到2024年12月（今天）
    # DownLoadData.update_tushare_shenwan_daily(start_date_str, end_date_str)
    # # 10、下载申万分类数据
    # DownLoadData.download_tushare_shenwan_classify()


    # 四、将数据转为Qlib格式
    # 1 初始化 （配置路径和参数）
    qlib_data_dir_path = "/home/yunbo/project/quantitative/qlib_data/cn_data"  # QLib数据存放目录
    dump_bin_py_path = "/home/yunbo/software/venv/lib/python3.10/site-packages/qlib/scripts/dump_bin.py"  # dump_bin.py脚本路径
    ToQlib = ToQlib.DataToQlib(qlib_data_dir_path=qlib_data_dir_path,dump_bin_py_path=dump_bin_py_path)
    # # # 2 开始转换
    ToQlib.start_to_qlib_single_thread()  # 转换指数日线行情
    # # 指数转化
    ToQlib.start_toqlib_index()  # 转换指数成分股到instruments (已完成)
    # # 3 测试校验
    # ToQlib.check_qlib_data()