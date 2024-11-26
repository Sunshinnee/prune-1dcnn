import tensorflow as tf
import numpy as np
from openpyxl import Workbook
import os

def save_weights_and_biases_to_excel(tflite_model_path, file_path, file_name):
    """
    提取 TFLite 模型的权重和偏置，保存到指定路径的 Excel 文件中。

    参数：
        tflite_model_path (str): TFLite 模型文件路径。
        file_path (str): 保存 Excel 文件的目录路径。
        file_name (str): 保存的 Excel 文件名（包含 .xlsx 后缀）。

    返回：
        None
    """
    # 检查保存路径是否存在，不存在则创建
    os.makedirs(file_path, exist_ok=True)
    output_excel_path = os.path.join(file_path, file_name)

    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # 获取张量详细信息
    tensor_details = interpreter.get_tensor_details()

    # 创建一个新的 Excel 工作簿
    wb = Workbook()
    ws = wb.active
    ws.title = "Weights and Biases"

    # 写入标题行
    ws.append(["Tensor Name", "Type", "Shape", "Data"])

    # 遍历张量，提取权重和偏置
    for tensor in tensor_details:
        tensor_name = tensor['name']
        tensor_data = interpreter.tensor(tensor['index'])()
        tensor_shape = tensor_data.shape
        tensor_type = tensor_data.dtype

        # 仅提取包含权重或偏置的张量
        if 'weight' in tensor_name.lower() or 'bias' in tensor_name.lower():
            # 将张量信息写入 Excel
            ws.append([
                tensor_name,  # 张量名称
                str(tensor_type),  # 数据类型
                str(tensor_shape),  # 张量形状
                np.array2string(tensor_data, separator=', ')  # 张量数据
            ])

    # 保存 Excel 文件到指定路径
    wb.save(output_excel_path)
    print(f"权重和偏置已保存到: {output_excel_path}")
