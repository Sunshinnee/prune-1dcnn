import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def plotHeatMap(Y_test, Y_pred, save_dir = None, filename = "heatmap.png"):
    # 如果Y_pred是概率分布，取最大概率的类别
    Y_pred_class = np.argmax(Y_pred, axis=1)

    # 如果Y_test是字符串类别标签，则转换为整数标签
    label_encoder = LabelEncoder()
    Y_test_encoded = label_encoder.fit_transform(Y_test)

    # 计算混淆矩阵
    con_mat = confusion_matrix(Y_test_encoded, Y_pred_class)
    # 绘图
    categories = ['N', 'A', 'V', 'L', 'R'] # 对应 Normal, Atrial, Ventricular, Left block, Right block
    plt.figure(figsize=(6, 5))
    sns.heatmap(con_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=categories, yticklabels=categories)

    # 设置坐标轴标签
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')


    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)  # 动态生成完整路径
    plt.savefig(save_path, bbox_inches='tight')  # 保存图表
    print(f"Heatmap saved to {save_path}")
    plt.close()  # 关闭图表，避免影响后续绘图


def plotAccuracy(history, save_dir = None, filename = "accuracy.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()


    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Accuracy saved to {save_path}")
    plt.close()



def plot_loss_curve(history, save_dir = None, filename = "loss_curve.png"):
    """
    绘制训练和验证损失曲线。

    参数:
        history (tensorflow.keras.callbacks.History): 训练过程中返回的History对象。
    """
    # 提取训练损失和验证损失
    train_loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])

    # 绘制训练损失和验证损失曲线
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')

    # 如果有验证损失，则绘制
    if val_loss:
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')

    # 添加标题和标签
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")
    plt.close()


def calculate_performance_metrics(Y_test, Y_pred, num_classes=5):
    # 如果 Y_pred 是概率分布，将其转换为类别标签
    if Y_pred.ndim > 1:  # 如果是多维数组（例如 one-hot 编码或者概率分布）
        Y_pred_class = np.argmax(Y_pred, axis=1)  # 获取最大概率对应的类别标签
    else:
        Y_pred_class = Y_pred  # 如果已经是类别标签

    # 确保 Y_test 也是整数标签
    label_encoder = LabelEncoder()
    Y_test_encoded = label_encoder.fit_transform(Y_test) if Y_test.ndim > 1 else Y_test  # 转换标签为整数编码
    # 计算混淆矩阵
    cm = confusion_matrix(Y_test_encoded, Y_pred_class)

    # 准确率 (Accuracy)
    accuracy = np.trace(cm) / np.sum(cm)  # 正确预测的样本数 / 总样本数

    # 敏感度 (Sensitivity) / 召回率 (Recall) for each class
    sensitivity = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        sensitivity.append(tp / (tp + fn) if (tp + fn) != 0 else 0)

    # 特异性 (Specificity) for each class
    specificity = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) != 0 else 0)

    # 阳性预测值 (Positive Predictive Value, PPV) for each class
    ppv = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        ppv.append(tp / (tp + fp) if (tp + fp) != 0 else 0)

    # 宏平均 (Macro-Average) for each metric
    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    macro_ppv = np.mean(ppv)

    return accuracy, sensitivity, specificity, ppv, macro_sensitivity, macro_specificity, macro_ppv


def print_acc_sparsity(sparsity, accuracy, save_dir=None, filename="acc_sparsity.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(sparsity, accuracy, marker='o', color='b', label='Accuracy')

    # 设置标题和轴标签
    plt.title(f"Accuracy vs Sparsity")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # 标注每个点的值
    for x, y in zip(sparsity, accuracy):
        plt.text(x, y, f"({x:.1f}%, {y:.2f})", fontsize=9, ha='right', va='bottom')

    # 如果提供了保存路径，则保存图像

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{filename} saved saved to {save_path}")
    plt.close()




def print_performance_metrics(y_true, y_pred, num_classes=5):
    accuracy, sensitivity, specificity, ppv, macro_sensitivity, macro_specificity, macro_ppv = calculate_performance_metrics(y_true, y_pred, num_classes)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    for i in range(num_classes):
        print(f"Class {i} - Sensitivity: {sensitivity[i] * 100:.2f}%")
        print(f"Class {i} - Specificity: {specificity[i] * 100:.2f}%")
        print(f"Class {i} - PPV: {ppv[i] * 100:.2f}%")

    print(f"Macro Sensitivity: {macro_sensitivity * 100:.2f}%")
    print(f"Macro Specificity: {macro_specificity * 100:.2f}%")
    print(f"Macro PPV: {macro_ppv * 100:.2f}%")