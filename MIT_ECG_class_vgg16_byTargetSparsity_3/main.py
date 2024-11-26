import wfdb
import pywt
import numpy as np
import train_and_purne_with_sparsity as tps
from calculate_result import print_acc_sparsity
import tensorflow as tf


tf.config.set_visible_devices([], 'GPU')
# 测试集在数据集中所占的比例
RATIO = 0.2
batch_size = 128
epochs = 100
R_WEIGHT = 0
database_path = "D:/purning/mit-bih-arrhythmia-database-1.0.0/"#MIT-BIH数据集的路径，根据项目路径自行修改

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data,record_ids):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    # 读取MLII导联的数据
    record = wfdb.rdrecord(database_path + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(database_path + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为320的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            # Rclass[i] 是标签
            lable = ecgClassSet.index(Rclass[i])
            # 基于经验值，基于R峰向前取110个点，向后取210个点
            x_train = rdata[Rlocation[i] - 110:Rlocation[i] + 210]

            if isinstance(x_train, np.ndarray) and x_train.dtype in [np.float32, np.float64]:  # 确保是数值类型
                X_data.append(x_train)
            else:
                print(f"Invalid x_train: {x_train}")

            if isinstance(lable, int):  # 确保是整数类型
                Y_data.append(lable)
            else:
                print(f"Invalid lable: {lable}")

            number_int = int(number)
            record_ids.append(number_int)  # 将对应的记录编号加入到 record_ids 中

            i += 1
        except ValueError:
            i += 1

    return

# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []#特征数据
    lableSet = []#对应的标签数据
    record_ids = []  # 存储心电图记录的编号
    num_images = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet,record_ids)
    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 320)#将特征数据 320 个为一行重新排序为一个二维数组
    lableSet = np.array(lableSet).reshape(-1, 1)#将数据标签 1 个为一行重新排序为一个二维数组（列向量）
    record_ids = np.array(record_ids).reshape(-1, 1)  # 将编号转为列向量
    train_ds = np.hstack((dataSet, lableSet,record_ids))#将特征数据和标签堆叠为一个整体数组trian_ds
    np.random.shuffle(train_ds)#随机化样本序列，有助于模型训练
    # 数据集及其标签集
    X = train_ds[:, :320].reshape(-1, 320, 1)#取前320列（这里就是随机化样本之后的所有特征数据），将特征数据形状调整为（样本数，320,1）用于一维卷积神经网络的输入特征图
    Y = train_ds[:, 320]#取第321列（多有样本的标签）
    ids = train_ds[:, 321]  # 取第322列（所有样本的原始编号）
    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    # 设定测试集的大小 RATIO是测试集在数据集中所占的比例
    test_length = int(RATIO * len(shuffle_index))
    # 测试集的长度
    test_index = shuffle_index[:test_length]
    # 训练集的长度
    train_index = shuffle_index[test_length:]
    num_images = len(train_ds)
    X_test, Y_test, ids_test = X[test_index], Y[test_index], ids[test_index]
    X_train, Y_train, ids_train = X[train_index], Y[train_index], ids[train_index]
    return X_train, Y_train, X_test, Y_test, ids_train, ids_test, num_images


def main():
    # 数据加载
    X_train, Y_train, X_test, Y_test, ids_train, ids_test, images = loadData()
    print("X_train shape:", X_train.shape)
    print("测试集的原始记录编号:", ids_test)

    # 设置稀疏度列表
    sparsity_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    NoPruneAcc_Array = []
    PruneAcc_Array = []
    #TestAcc_Array = []


    # 循环执行训练与剪枝
    for sparsity in sparsity_values:
        NoPruneAcc, PruneAcc =tps.train_and_prune_with_sparsity(X_train, Y_train, X_test, Y_test, images, epochs, batch_size, sparsity, RATIO, R_WEIGHT)
        NoPruneAcc_Array.append(NoPruneAcc)
        PruneAcc_Array.append(PruneAcc)
        #TestAcc_Array.append(TestAcc)

    print_acc_sparsity(sparsity_values,NoPruneAcc_Array,save_dir=f"./model",filename="NoPrune_AccBySparsity.png")
    print_acc_sparsity(sparsity_values, PruneAcc_Array, save_dir=f"./model", filename="Pruned_AccBySparsity.png")




if __name__ == "__main__":
    main()




