import numpy as np
import pandas as pd

from DataLoad import DataProcessor

# ===================== 1. 复刻报告Token库（完全沿用报告设置）=====================
class FactorTokenLibrary:
    """因子生成的Token库，严格对应报告中的字段、常数、算子定义"""
    def __init__(self):
        self.data_processor = DataProcessor()
        # 字段Token
        self.FIELDS = self.data_processor.fields_origin  # ['open', 'high', 'low', 'close']
        # 常数Token（13个）
        self.CONSTANTS = [-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        # 算子Token（4类）
        self.OPERATORS = {
            "unary": ["Abs", "Log"],  # 一元操作符
            "binary": ["Add", "Sub", "Mul", "Div", "Greater", "Less"],  # 二元操作符
            "rolling": ["Ref", "Mean", "Sum", "Std", "Var", "Max", "Min", "Med", "Mad", "Delta", "WMA", "EMA"],  # 滚动操作符
            "pair_rolling": ["Cov", "Corr"]  # 配对滚动操作符
        }
        # 起止符Token
        self.BEG = "BEG"
        self.SEP = "SEP"
    
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
        self.max_length = 15  # 因子最大长度（报告超参数：MAX_EXPR_LENGTH=15）
        # 运算符映射表
        self.op_map = {
                "Abs": "abs", "Log": "log", "Add": "+", "Sub": "-", "Mul": "*",
                "Div": "/", "Greater": ">", "Less": "<", "Ref": "ref", "Mean": "mean",
                "Sum": "sum", "Std": "std", "Var": "var", "Max": "max", "Min": "min",
                "Med": "median", "Mad": "mad", "Delta": "delta", "WMA": "wma", "EMA": "ema",
                "Cov": "cov", "Corr": "corr"
            }

        self.qlib_op_map = {
                    "Abs": "Abs", "Log": "Log", "Add": "Add", "Sub": "Sub", "Mul": "Mul",
                    "Div": "Div", "Greater": "Greater", "Less": "Less", "Ref": "Ref", "Mean": "MA",
                    "Sum": "Sum", "Std": "Std", "Var": "Var", "Max": "Max", "Min": "Min",
                    "Med": "Median", "Mad": "Mad", "Delta": "Delta", "WMA": "WMA", "EMA": "EMA",
                    "Cov": "Cov", "Corr": "RollingCorr"}
    
    def decode(self, tokens: list, return_type: str = "dict"): 
        """
        将从Token序列解码为表达式
        :param tokens: 逆波兰token序列（如['BEG', 'OPEN', '1.0', 'Add', 'Log', 'CLOSE', 'Div']）
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
        # 运算符→字符串映射（适配QLib/普通数学表达式）
        op_map = self.op_map
        
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
                    "str": f"{op_map.get(token, token)}({operand['str']})"
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
                    # 构建字符串：(left op right)（加括号保证运算顺序）
                    "str": f"({left['str']} {op_map.get(token, token)} {right['str']})"
                }
                stack.append(ast_node)
            elif token in self.token_lib.OPERATORS["rolling"]:
                # 滚动运算符（如MA/RollingMean）
                if len(stack) < 1:
                    raise ValueError("Invalid RPN: 滚动运算符缺少操作数")
                operand = stack.pop()
                ast_node = {
                    "type": "operation",
                    "operator": token,
                    "operands": [operand],
                    "str": f"{op_map.get(token, token)}({operand['str']})"
                }
                stack.append(ast_node)
            elif token in self.token_lib.OPERATORS["pair_rolling"]:
                # 成对滚动运算符（如RollingCorr）
                if len(stack) < 2:
                    raise ValueError("Invalid RPN: 成对滚动运算符缺少操作数")
                right = stack.pop()
                left = stack.pop()
                ast_node = {
                    "type": "operation",
                    "operator": token,
                    "operands": [left, right],
                    "str": f"{op_map.get(token, token)}({left['str']}, {right['str']})"
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
    
    
    def _recursive_encode(self, node, token_sequence):
        """递归后序遍历：先处理子节点（字段/常数/子表达式），再处理当前算子"""
        if isinstance(node, str) and node in self.token_lib.FIELDS:
            # 字段节点：直接加入序列
            token_sequence.append(node)
        elif isinstance(node, (int, float)) and node in self.token_lib.CONSTANTS:
            # 常数节点：转为字符串加入序列
            token_sequence.append(str(node))
        elif isinstance(node, dict) and "op" in node:
            # 算子节点：先递归处理所有参数，再加入算子
            for arg in node["args"]:
                self._recursive_encode(arg, token_sequence)
            token_sequence.append(node["op"])


    
    def _is_valid_expression(self, tokens):
        """逆波兰表达式（后缀表达式）的栈式语法规则
        操作数入栈，运算符按类型消耗对应数量的操作数并生成新结果入栈，最终栈中仅留 1 个结果则表达式合法。
        """
        # 简单的语法检查
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
            elif token in self.token_lib.OPERATORS["binary"] or token in self.token_lib.OPERATORS["pair_rolling"]:
                if len(stack) < 2:
                    return False
                stack.pop()
                stack.pop()
                stack.append("operand")
            elif token in self.token_lib.OPERATORS["rolling"]:
                if len(stack) < 1:
                    return False
                stack.pop()
                stack.append("operand")
        # 有效表达式应该有且只有一个结果
        return len(stack) == 1

