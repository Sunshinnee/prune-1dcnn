import tensorflow as tf

# 构建VGG16-CNN模型
def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(320, 1)),
        #第一个卷积层， 4 个 5*1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=5, strides=1, padding='SAME', activation='relu'),
        #第二个卷积层， 4 个 5*1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=5, strides=1, padding='SAME', activation='relu'),
        #第一个池化层， 最大池化， 2*1 卷积核
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='SAME'),
        #第三个卷积层， 8 个 5*1 卷积核
        tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='SAME', activation='relu'),
        #第四个卷积层， 8 个 5*1 卷积核
        tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='SAME', activation='relu'),
        #第二个池化层，最大池化， 2*1 卷积核
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='SAME'),
        #第五个卷积层， 16 个 5*1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='SAME', activation='relu'),
        #第六个卷积层， 16 个 5*1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='SAME', activation='relu'),
        # 第三个池化层，最大池化， 2*1 卷积核
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='SAME'),
        # 第七个卷积层， 32 个 5*1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='SAME', activation='relu'),
        # 第八个卷积层， 32 个 5*1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='SAME', activation='relu'),
        # 第四个池化层，最大池化， 2*1 卷积核
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='SAME'),
        # 第五个池化层，全局平均池化
        tf.keras.layers.AvgPool1D( padding='SAME'),
        # 打平层,方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点 转换成128个节点
        tf.keras.layers.Dense(20,activation='relu'),
        # Dropout层,dropout = 0.2
        #tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newModel