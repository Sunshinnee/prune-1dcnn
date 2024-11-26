import tempfile
import tensorflow as tf
import pandas as pd

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)
def convert_to_sparse(model):
    """将剪枝后的模型转换为稀疏格式"""
    # 获取模型权重
    weights = model.get_weights()

    # 将每一层的权重转换为稀疏矩阵（仅在有零值时才有效）
    sparse_weights = []
    for weight in weights:
        if len(weight.shape) > 1:  # 如果是矩阵或更高维度（如2D权重矩阵）
            sparse_weight = tf.sparse.from_dense(weight)
            sparse_weights.append(sparse_weight)
        else:
            sparse_weights.append(weight)  # 一维权重不进行转换
    return sparse_weights

# 保存模型权重到 Excel
def save_weights_to_excel(model, filename="model_weights.xlsx"):
    # 创建一个字典来存储权重
    weights_dict = {}

    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()  # 获取权重和偏置
        if weights:  # 如果该层有权重
            weights_dict[f"Layer_{i}_weights"] = weights[0].flatten().tolist()  # 展平权重矩阵
            weights_dict[f"Layer_{i}_bias"] = weights[1].tolist()  # 偏置为一维

    # 将字典转换为 DataFrame
    max_length = max(len(v) for v in weights_dict.values())
    for k, v in weights_dict.items():
        if len(v) < max_length:
            weights_dict[k] += [None] * (max_length - len(v))  # 补齐列长度

    df = pd.DataFrame(weights_dict)
    # 保存为 Excel 文件
    df.to_excel(filename, index=False)
    print(f"Weights saved to {filename}")