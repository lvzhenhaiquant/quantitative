import datetime
import time
import os
import pandas as pd
import os
import shutil
# 假设这两个是你的自定义模块
import DownLoadData
import ToQlib
from datetime import timedelta


# ===================== 断点持久化工具函数 =====================
def save_breakpoint(date_str):
    """保存断点到本地文件（记录最后一次成功下载的结束日期）"""
    try:
        with open(BREAKPOINT_FILE, 'w', encoding='utf-8') as f:
            f.write(date_str)
        print(f"📌 {MODULE_NAME}：断点已保存 | 最后成功日期：{date_str}")
    except Exception as e:
        print(f"⚠️ {MODULE_NAME}：断点保存失败 | 错误：{str(e)}")

def load_breakpoint():
    """从本地文件加载断点（重启后恢复）"""
    if not os.path.exists(BREAKPOINT_FILE):
        print(f"📝 {MODULE_NAME}：未找到断点文件,返回None,使用指定日期")
        return None  # 首次运行返回初始日期
    
    try:
        with open(BREAKPOINT_FILE, 'r', encoding='utf-8') as f:
            date_str = f.read().strip()
        # 校验日期格式是否合法
        datetime.datetime.strptime(date_str, '%Y%m%d')
        print(f"🔄 {MODULE_NAME}：已加载断点 | 最后成功日期：{date_str}")
        return date_str
    except Exception as e:
        print(f"⚠️ {MODULE_NAME}：断点文件损坏，使用初始日期 {INIT_START_DATE} | 错误：{str(e)}")
        return INIT_START_DATE



def backup_qlib_data():
    source = '/home/yunbo/project/quantitative/qlib_data'
    destination = '/home/yunbo/下载/quantitative_old/qlib_data'
    temp_backup = destination + "_temp"

    # 如果目标文件夹存在，先重命名为临时文件夹
    if os.path.exists(destination):
        try:
            os.rename(destination, temp_backup)
            print(f"✅ 已将目标文件夹重命名为临时文件夹：{temp_backup}")
        except Exception as e:
            print(f"❌ 无法重命名目标文件夹：{e}")
            return

    # 复制源文件夹到目标位置
    try:
        shutil.copytree(source, destination)
        print(f"✅ 备份完成：{source} -> {destination}")
        # 如果复制成功，删除临时文件夹
        if os.path.exists(temp_backup):
            shutil.rmtree(temp_backup)
            print(f"✅ 已删除临时文件夹：{temp_backup}")
    except Exception as e:
        print(f"❌ 备份失败：{e}")
        # 如果复制失败，恢复临时文件夹
        if os.path.exists(temp_backup):
            os.rename(temp_backup, destination)
            print(f"✅ 已恢复原目标文件夹：{destination}")

# ===================== 数据下载核心函数 =====================
def download_stock_data(start_date_str, end_date_str):
    """封装数据下载逻辑"""
    try:
        # 二、从Tushare拉取数据到本地处理
        # 1、初始化
        downloader  = DownLoadData.DownloadDataFromTushare_Baostock(TUSHARE_TOKEN)
        # 2、下载"中证1000"基础行情数据
        # downloader.download_tushare_basic(start_date_str,end_date_str)  # 已完成
        # 3、下载"中证1000"财务数据
        # downloader.download_tushare_finance(start_date_str,end_date_str)  # 已完成
        # 4、下载"中证1000"半小时级行情数据
        # downloader.download_baostock_basic_mins(start_date_str,end_date_str)


        # 5、下载全A股基础行情数据
        # downloader.download_tushare_A_basic(start_date_str,end_date_str)
        # 6、下载全A财务数据
        # downloader.download_tushare_A_finance(start_date_str,end_date_str)
        # 7、下载全A股基础半小时级行情数据
        # downloader.download_baostock_A_basic_mins(start_date_str, end_date_str)
        # 8、下载指数成分股
        # downloader.download_index(start_date_str,end_date_str)
        # 9、下载指数日线行情
        # downloader.download_index_daily(start_date_str,end_date_str)
        # 10、下载申万分类数据
        # downloader.download_tushare_shenwan_classify()
        # 11、下载申万指数日线行情
        # downloader.download_tushare_shenwan_daily(start_date_str,end_date_str)


        # 三、增量基础行情更新 (与2~4互斥使用)
        # 1、增量更新“中证1000”基础行情数据
        # downloader.updates_tushare_basic(start_date_str,end_date_str)
        # 2、增量更新“中证1000”财务数据
        # downloader.updates_tushare_finance(start_date_str,end_date_str)
        # 3、增量更新“中证1000”半小时级行情数据
        # downloader.updates_baostock_basic_mins(start_date_str, end_date_str)
        

        # 4、增量下载全A基础行情数据
        downloader.updates_tushare_A_basic(start_date_str, end_date_str)
        # 5、增量下载全A财务数据
        downloader.updates_tushare_A_finance(start_date_str, end_date_str)
        # 6、增量更新全A半小时级数据
        downloader.updates_tushare_A_basic_mins(start_date_str, end_date_str)
        # 7、增量更新指数成分股
        downloader.update_index(start_date_str, end_date_str)
        # 8、增量更新指数日线行情
        downloader.update_index_daily(start_date_str, end_date_str)
        # 9、增量下载申万日线行情
        downloader.update_tushare_shenwan_daily(start_date_str, end_date_str)
        # 10、下载申万分类数据
        downloader.download_tushare_shenwan_classify()
        # 11、增量更新申万指数成分股
        DownLoadData.updates_tushare_shenwan_constituent_stock(start_date_str, end_date_str)

        # 四、将数据转为Qlib格式
        # 1 初始化 （配置路径和参数）
        qlib_data_dir_path = "/home/yunbo/project/quantitative/qlib_data/cn_data"  # QLib数据存放目录
        dump_bin_py_path = "/home/yunbo/software/venv/lib/python3.10/site-packages/qlib/scripts/dump_bin.py"  # dump_bin.py脚本路径
        qlib_min_data_dir_path="/home/yunbo/project/quantitative/qlib_data/cn_data_60min"
        toqlib = ToQlib.DataToQlib(qlib_data_dir_path=qlib_data_dir_path,qlib_min_data_dir_path=qlib_min_data_dir_path,dump_bin_py_path=dump_bin_py_path)
        # # # 2 开始转换
        toqlib.start_to_qlib_single_thread()  # 转换指数日线行情
        # # 指数转化
        toqlib.start_toqlib_index()  # 转换指数成分股到instruments (已完成)
        print(f"✅ {MODULE_NAME}：数据下载完成 | 时间范围：{start_date_str} ~ {end_date_str}")
        return True
    except Exception as e:
        print(f"❌ {MODULE_NAME}：数据下载失败 | 错误：{str(e)} | 时间范围：{start_date_str} ~ {end_date_str}")
        return False

def get_date_str(date_obj):
    """日期转YYYYMMDD字符串"""
    return date_obj.strftime('%Y%m%d')


# ===================== 核心配置（可修改） =====================
TUSHARE_TOKEN = 'a79f284e5d10967dacb6531a3c755a701ca79341ff0c60d59f1fcbf1'
RUN_HOUR = 18  # 每日运行时间（18点）
RETRY_INTERVAL = 60  # 失败重试间隔（秒）
BREAKPOINT_FILE = "stock_data_breakpoint.txt"  # 断点保存文件路径
MODULE_NAME = "股票数据下载程序（断点续传版）"
INIT_START_DATE = datetime.datetime.strptime('20251227', '%Y%m%d').date()  # 初始起始日期
init_end_date = datetime.datetime.strptime('20251228', '%Y%m%d').date()     # 初始结束日期

#使用说明
#如果stock_data_breakpoint.txt文件不存在，程序将从INIT_START_DATE到init_end_date下载数据
#如果stock_data_breakpoint.txt文件存在，程序将读取从上次断点日期到today的日期下载数据
#~/software/venv/bin/python3.10 /home/yunbo/project/quantitative/data/main_auto_execute.py #每天自动运行

# ===================== 主程序 =====================
def main():
    while True:
        try:
            # 1. 获取基础时间变量
            today = datetime.datetime.now().date()
            now = datetime.datetime.now()
            last_breakpoint_str = load_breakpoint()

            # 2. 确定本次下载的起止日期
            if last_breakpoint_str is None:
                # 无断点：使用初始化起止日期
                current_start_date = INIT_START_DATE
                current_end_date = init_end_date
            else:
                # 有断点：从上次结束日期开始，到当天结束
                last_success_end_date = datetime.datetime.strptime(last_breakpoint_str, '%Y%m%d').date()
                current_start_date = last_success_end_date
                current_end_date = today
            print(f"\n🔍 本次计划下载日期范围：{get_date_str(current_start_date)} ~ {get_date_str(current_end_date)}")
            # 3. 边界判断：起始日期 > 结束日期 → 等待次日运行
            if current_start_date > current_end_date:
                next_run_date = today + datetime.timedelta(days=1)
                next_run_time = datetime.datetime.combine(next_run_date, datetime.time(RUN_HOUR, 0, 0))
                wait_seconds = (next_run_time - now).total_seconds()
                print(f"\n📅 起始日期 {get_date_str(current_start_date)} 超过结束日期 {get_date_str(current_end_date)}")
                print(f"⏳ 等待次日{RUN_HOUR}点运行，需等待 {wait_seconds / 3600:.2f} 小时")
                time.sleep(wait_seconds)
                continue

            # 4. 等待到指定运行时间（仅处理当天数据时）
            if current_end_date == today:
                run_time_today = datetime.datetime.combine(today, datetime.time(RUN_HOUR, 0, 0))
                if now < run_time_today:
                    # 当前时间还没到指定运行时间，等待
                    wait_seconds = (run_time_today - now).total_seconds()
                    print(f"\n🕒 当前时间 {now.strftime('%H:%M:%S')}，需等待 {wait_seconds / 3600:.2f} 小时至{RUN_HOUR}点")
                    time.sleep(wait_seconds)
                    # 等待后更新now和today（避免时间偏差）
                    # now = datetime.datetime.now()
                    # today = now.date()

            # 5. 执行下载
            start_str = get_date_str(current_start_date)
            end_str = get_date_str(current_end_date)
            print(f"\n>>> 开始下载：{start_str} ~ {end_str} <<<")
            download_stock_data(start_str, end_str)  # 执行实际下载
            backup_qlib_data()# 备份qlib_data数据

            # 6. 更新断点（保存本次结束日期）
            save_breakpoint(end_str)

            # 7. 计算下次运行时间（次日指定小时）
            if now.hour >= RUN_HOUR:
                # 如果当前时间已经过了今天的18点运行时间，安排到明天
                next_run_date = today + datetime.timedelta(days=1)
                
            else:
                # 否则安排到今天18点后
                next_run_date = today + datetime.timedelta(days=0)
            next_run_time = datetime.datetime.combine(next_run_date, datetime.time(RUN_HOUR, 0, 0))
            wait_seconds = (next_run_time - datetime.datetime.now()).total_seconds()#
            wait_seconds = max(0, wait_seconds)#如果当日结束时间超过了18点，则立即运行，等待时间为0，不能为负数
            print(f"\n✅ 本次下载完成，断点已更新为 {end_str}")
            print(f"📅 下次运行时间：{next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏳ 需等待 {wait_seconds / 3600:.2f} 小时")
            
            # 等待到下次运行时间
            time.sleep(wait_seconds)

        except Exception as e:
            print(f"\n❌ 程序异常：{str(e)}")
            print(f"⏳ {RETRY_INTERVAL}秒后重试...")
            time.sleep(RETRY_INTERVAL)  # 10分钟后重试
            continue


if __name__ == '__main__':
    main()