import numpy as np
import pandas as pd

from DataLoad import DataProcessor
from calculator import FactorCalculator

# ===================== 1. 复刻报告Token库（完全沿用报告设置）=====================
class FactorTokenLibrary:
    """因子生成的Token库，严格对应报告中的字段、常数、算子定义"""
    def __init__(self):
        self.data_processor = DataProcessor()
        # 字段Token
        self.FIELDS = self.data_processor.fields_origin  # ['open', 'high', 'low', 'close']
        # 常数Token
        self.CONSTANTS = [-250, -120, -60, -30, -10, -5, -2, -1, -0.5, -1, 0.5, 1, 2, 5, 10, 30, 60 ,120, 250]
        # 算子Token（直接从FactorCalculator获取）
        self.OPERATORS = FactorCalculator.SUPPORTED_OPERATORS
        # 起止符Token
        self.BEG = "BEG"
        self.SEP = "SEP"
        self.token_max_length = 15

         # 1. 因子大类（枚举式定义，覆盖常见量化因子类型）
        self.FACTOR_TYPES = {
            # 基础类型
            'PRICE_VOLUME': '量价因子',          # 基于open/high/low/close的因子
            'VOLUME': '成交量因子',              # 扩展：若有volume字段时启用
            'VOLATILITY': '波动率因子',          # 基于标准差/方差的因子
            'MOMENTUM': '动量因子',              # 基于Ref/Delta/Mean的趋势因子
            'MEAN_REVERSION': '均值回归因子',    # 基于反向Delta/EMA的反转因子
            'CORRELATION': '相关性因子',         # 基于Cov/Corr的配对因子
            # 扩展类型（可按需添加）
            'TECHNICAL': '技术指标因子',         # 如MA/RSI/MACD（复合算子组合）
            'CUSTOM': '自定义因子'               # 非标准类型
        }

    @property
    def all_tokens(self):
        """所有Token的集合（用于强化学习动作空间）"""
        operators_flat = [op for op_type in self.OPERATORS.values() for op in op_type]
        constants_str = [str(c) for c in self.CONSTANTS]  # 常数转为字符串形式便于Token化
        result =[self.BEG, self.SEP] + self.FIELDS + constants_str + operators_flat
        return result



class RPNEncoder:
    """将传统因子表达式转为逆波兰表达式Token序列，严格遵循报告编码逻辑"""
    def __init__(self, token_lib: FactorTokenLibrary):
        self.token_lib = token_lib
        self.max_length = token_lib.token_max_length  # 因子最大长度（报告超参数：MAX_EXPR_LENGTH=15）
    
    def decode(self, tokens: list, return_type: str = "dict"): 
        """
        将从Token序列解码为表达式
        :param tokens: 逆波兰token序列（如['BEG', 'OPEN', '1', 'Add', 'Log', 'CLOSE', 'Div']）
        :param return_type: 返回类型，"dict"=AST字典，"string"=可执行的字符串表达式
        :return: 对应类型的因子表达式
        """
        # 第一步：过滤掉BEG/SEP，保留核心逆波兰token
        filtered_tokens = []
        for token in tokens:
            if token == self.token_lib.BEG:
                continue
            if token == self.token_lib.SEP:
                break
            filtered_tokens.append(token)
        if not filtered_tokens:
            return "" if return_type == "string" else {}
        
        # 第二步：用栈构建AST字典
        stack = []
        for token in filtered_tokens:
            if token in self.token_lib.FIELDS:
                # 字段：入栈（字典/字符串占位）
                stack.append({
                    "type": "field", 
                    "name": token,
                    "str": token  # 字符串表示
                })
            elif token in [str(c) for c in self.token_lib.CONSTANTS]:
                # 常量：入栈
                stack.append({
                    "type": "constant", 
                    "value": token,
                    "str": token  # 字符串表示
                })
            elif token in self.token_lib.OPERATORS["unary"]:
                # 一元运算符（Log/Abs等）
                if len(stack) < 1:
                    raise ValueError("Invalid RPN: 一元运算符缺少操作数")
                operand = stack.pop()
                # 构建AST字典
                ast_node = {
                    "type": "operation",
                    "operator": token,
                    "operands": [operand],
                    # 构建字符串：Log(operand)
                    "str": f"{token}({operand['str']})"
                }
                stack.append(ast_node)
            elif token in self.token_lib.OPERATORS["binary"]:
                # 二元运算符（+/-/*/÷等）
                if len(stack) < 2:
                    raise ValueError("Invalid RPN: 二元运算符缺少操作数")
                right = stack.pop()
                left = stack.pop()
                # 构建AST字典
                ast_node = {
                    "type": "operation",
                    "operator": token,
                    "operands": [left, right],
                    # 构建字符串：Add(left, right)（函数调用形式，不再是中缀表达式）
                    "str": f"{token}({left['str']}, {right['str']})"
                }
                stack.append(ast_node)
            elif token in self.token_lib.OPERATORS["rolling"]:
                # 滚动运算符（如MA/RollingMean）
                if len(stack) < 2:
                    raise ValueError("Invalid RPN: 滚动运算符缺少操作数")
                window = stack.pop()  # 第二个操作数是窗口大小
                operand = stack.pop()  # 第一个操作数是数据字段
                # 构建AST字典
                ast_node = {
                    "type": "operation",
                    "operator": token,
                    "operands": [operand, window],  # 保存两个操作数：数据字段和窗口大小
                    "str": f"{token}({operand['str']}, {window['str']})"
                }
                stack.append(ast_node)
            elif token in self.token_lib.OPERATORS["pair_rolling"]:
                # 成对滚动运算符（如RollingCorr）
                if len(stack) < 3:
                    raise ValueError("Invalid RPN: 成对滚动运算符缺少操作数")
                window = stack.pop()  # 第三个操作数是窗口大小
                right = stack.pop()   # 第二个操作数是第二个数据字段
                left = stack.pop()    # 第一个操作数是第一个数据字段
                # 构建AST字典
                ast_node = {
                    "type": "operation",
                    "operator": token,
                    "operands": [left, right, window],  # 保存三个操作数：两个数据字段和窗口大小
                    "str": f"{token}({left['str']}, {right['str']}, {window['str']})"
                }
                stack.append(ast_node)
            else:
                raise ValueError(f"未知Token: {token}")
        
        if len(stack) != 1:
            raise ValueError("Invalid RPN: 栈最终应只有一个元素")
        # 根据return_type返回对应结果
        #表达式string
        expr_str = stack[0]["str"].strip()
        if expr_str.startswith("(") and expr_str.endswith(")"):
            expr_str = expr_str[1:-1]
        print(expr_str)
        
        if return_type == "string":
            return expr_str
        # 返回AST字典，移除临时的str字段
        def remove_str_field(node):
            if "str" in node:
                del node["str"]
            if node["type"] == "operation":
                for op in node["operands"]:
                    remove_str_field(op)
            return node
        return remove_str_field(stack[0])  

    def _is_valid_expression(self, tokens):
        """逆波兰表达式（后缀表达式）的栈式语法规则
        操作数入栈，运算符按类型消耗对应数量的操作数并生成新结果入栈，最终栈中仅留 1 个结果则表达式合法。
        """
        # 语法检查
        stack = []
        for token in tokens:
            if token == self.token_lib.BEG:
                continue
            if token == self.token_lib.SEP:
                break
            if token in self.token_lib.FIELDS or token in [str(c) for c in self.token_lib.CONSTANTS]:
                stack.append("operand")
            elif token in self.token_lib.OPERATORS["unary"]:
                if len(stack) < 1:
                    return False
                stack.pop()
                stack.append("operand")
            elif token in self.token_lib.OPERATORS["binary"] or token in self.token_lib.OPERATORS["rolling"]:
                # 二元操作符和滚动操作符都需要2个操作数
                if len(stack) < 2:
                    return False
                stack.pop()
                stack.pop()
                stack.append("operand")
            elif token in self.token_lib.OPERATORS["pair_rolling"]:
                # 配对滚动操作符需要3个操作数（两个数据字段和一个窗口大小）
                if len(stack) < 3:
                    return False
                stack.pop()
                stack.pop()
                stack.pop()
                stack.append("operand")
        # 有效表达式应该有且只有一个结果
        return len(stack) == 1


    