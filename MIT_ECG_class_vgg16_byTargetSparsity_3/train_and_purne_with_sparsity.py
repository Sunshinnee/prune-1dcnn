import os
import tempfile
import numpy as np
import tensorflow as tf
import evaluate_model
import purning_function
from openpyxl import Workbook
import quantizationFunc
from purning_function import get_gzipped_model_size
import calculate_result
import buildModel as bm
import tensorflow_model_optimization as tfmot
from SparsityAccuracyCallback import SparsityAccuracyCallback


def train_and_prune_with_sparsity(X_train, Y_train, X_test, Y_test, images, epochs, batch_size, final_sparsity,RATIO, R_WEIGHT):
    """
    执行一次训练和剪枝，稀疏度由 final_sparsity 设置。
    """
    print(f"Starting training and pruning with final_sparsity={final_sparsity}...")
    # 构建模型
    model = bm.buildModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 基础模型训练
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=RATIO)
    loss, no_pruning_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Model without pruning in {final_sparsity:.1f} target sparsity test accuracy: {no_pruning_acc:.4f}")

    #保存baseline模型
    save_dir = f"./model/TargetSparsity_{final_sparsity:.1f}"
    os.makedirs(save_dir, exist_ok=True)

    keras_file = os.path.join(save_dir, f"TargetSparsity_{final_sparsity:.1f}_baseline_model.h5")
    model_no_pruning_file = os.path.join(save_dir, f"TargetSparsity_{final_sparsity:.1f}_model_no_pruning.h5")

    tf.keras.models.save_model(model, keras_file, include_optimizer=False)
    print(f'Saved baseline model in {final_sparsity:.1f} target sparsity to:', keras_file)

    model.save(model_no_pruning_file)
    print(f'Saved model without pruning in {final_sparsity:.1f} target sparsity to:', model_no_pruning_file)

    #baseline预测
    Y_pred = model.predict(X_test)
    print(Y_pred)
    os.makedirs(f"./model/TargetSparsity_{final_sparsity:.1f}/weight_in_{final_sparsity:.1f}TargetSparsity", exist_ok=True)
    # 训练完模型后保存权重
    purning_function.save_weights_to_excel(model, f"model/TargetSparsity_{final_sparsity:.1f}/weight_in_{final_sparsity:.1f}TargetSparsity/noPrune_model_weights_{final_sparsity:.1f}TargetSparsity.xlsx")
    os.makedirs(f"./model/TargetSparsity_{final_sparsity:.1f}/ResultFig_in_{final_sparsity:.1f}TargetSparsity", exist_ok=True)
    # 绘制混淆矩阵热力图
    calculate_result.plotHeatMap(Y_test, Y_pred, f"./model/TargetSparsity_{final_sparsity:.1f}/ResultFig_in_{final_sparsity:.1f}TargetSparsity", f"baseline_heatmap_{final_sparsity:.1f}TargetSparsity.png")
    # 绘制训练和验证精度变化图表
    calculate_result.plotAccuracy(history,f"./model/TargetSparsity_{final_sparsity:.1f}/ResultFig_in_{final_sparsity:.1f}TargetSparsity", f"baseline_accuracy_{final_sparsity:.1f}TargetSparsity.png")
    # 绘制损失函数曲线和训练的变化图表
    calculate_result.plot_loss_curve(history,f"./model/TargetSparsity_{final_sparsity:.1f}/ResultFig_in_{final_sparsity:.1f}TargetSparsity", f"baseline_loss_{final_sparsity:.1f}TargetSparsity.png")
    # 从准确率acc、敏感度sen、特异性spec、阳性预测值ppv等角度分析模型性能
    print("Before pruning, calculate the acc, sen, spec, ppv of the model_no_puring:")
    calculate_result.print_performance_metrics(Y_test, Y_pred,num_classes=5)

    # 剪枝设置
    print(f"start pruning for {final_sparsity:.1f} target sparsity model...")
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    end_step = np.ceil(images / batch_size).astype(np.int32) * epochs
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.05,
            final_sparsity=final_sparsity,
            begin_step=0,
            end_step=end_step
        )
    }
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_for_pruning.summary()

    # 剪枝后的训练
    logdir = tempfile.mkdtemp()
    # 定义回调
    accuracy_file = f"./model/TargetSparsity_{final_sparsity:.1f}/weight_in_{final_sparsity:.1f}TargetSparsity/{final_sparsity:.1f}TargetSparsity_accuracy_epoch.xlsx"
    loss_file = f"./model/TargetSparsity_{final_sparsity:.1f}/weight_in_{final_sparsity:.1f}TargetSparsity/{final_sparsity:.1f}TargetSparsity_loss_epoch.xlsx"
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        SparsityAccuracyCallback(model_for_pruning, final_sparsity, accuracy_file, loss_file),
    ]
    purning_history = model_for_pruning.fit(
        X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=RATIO, callbacks=callbacks
    )
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    loss, pruning_acc = model_for_pruning.evaluate(X_test, Y_test, verbose=0)
    print(f"Model no pruning (final_sparsity={final_sparsity}) test accuracy: {no_pruning_acc: .4f}")
    print(f"Model with pruning (final_sparsity={final_sparsity}) test accuracy: {pruning_acc:.4f}")
    print("logdir:", logdir)

    # 保存模型

    os.makedirs(save_dir, exist_ok=True)
    pruned_keras_file = os.path.join(save_dir, f"TargetSparsity_{final_sparsity:.1f}_pruned_keras_file.h5")
    model_for_pruning_file = os.path.join(save_dir, f"TargetSparsity_{final_sparsity:.1f}_model_for_pruning.h5")
    pruned_tflite_file = os.path.join(save_dir, f"TargetSparsity_{final_sparsity:.1f}_pruned_model.tflite")
    pruned_tflite_file_fp16 = os.path.join(save_dir, f"TargetSparsity_{final_sparsity:.1f}_pruned_model_fp16.tflite")
    #pruned_tflite_file_int8 = os.path.join(save_dir, f"TargetSparsity_{final_sparsity:.1f}_pruned_model_int8.tflite")


    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved baseline model to:', pruned_keras_file)

    model_for_pruning.save(model_for_pruning_file)
    print('Saved model without pruning to:', model_for_pruning_file)

    '''
    # int8量化
    #为 TFLite 创建一个可压缩模型
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    # 设置代表性数据集
    converter_int8.representative_dataset = representative_dataset_gen
    converter_int8.target_spec.supported_types = [tf.int8]
    pruned_tflite_model_int8 = converter_int8.convert()
    with open(pruned_tflite_file_int8, 'wb') as f:
        f.write(pruned_tflite_model_int8)

    print('Saved pruned TFLite model to:', pruned_tflite_file_int8)

    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
    print("Size of gzipped pruned TFlite model(int8): %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file_int8)))
    '''
    #fp16量化

    #为 TFLite 创建一个可压缩模型
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    #converter_fp16._model = pruned_tflite_model
    pruned_tflite_model_fp16 = converter_fp16.convert()

    with open(pruned_tflite_file_fp16, 'wb') as f:
        f.write(pruned_tflite_model_fp16)

    print('Saved pruned TFLite model to:', pruned_tflite_file_fp16)

    # 第二阶段：对量化后的模型应用剪枝（稀疏化优化）
    # 先从量化后的TFLite模型加载
    #interpreter = tf.lite.Interpreter(model_content=pruned_tflite_model_fp16)
    #interpreter.allocate_tensors()

    # 获取量化后的模型张量信息
    #tensor_details = interpreter.get_tensor_details()

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    converter._model = pruned_tflite_model_fp16  # 将量化后的模型作为输入
    pruned_tflite_model = converter.convert()
    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)

    print('Saved pruned TFLite model to:', pruned_tflite_file)

    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
    print("Size of gzipped pruned TFlite model(int8): %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))


    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
    print("Size of gzipped pruned TFlite model(fp16): %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file_fp16)))

    # 剪枝后的预测
    Y_pred_purning = model_for_pruning.predict(X_test)
    print(Y_pred_purning)
    os.makedirs(f"./model/TargetSparsity_{final_sparsity:.1f}/weight_in_{final_sparsity:.1f}TargetSparsity",
                exist_ok=True)
    # 训练完模型后保存权重
    purning_function.save_weights_to_excel(model_for_pruning, f"model/TargetSparsity_{final_sparsity:.1f}/weight_in_{final_sparsity:.1f}TargetSparsity/Prune_model_weights_{final_sparsity:.1f}TargetSparsity.xlsx")
    # 保存int8和fp16量化后的权重
    #quantizationFunc.save_weights_and_biases_to_excel(pruned_tflite_model_int8,f"model/TargetSparsity_{final_sparsity:.1f}/weight_in_{final_sparsity:.1f}TargetSparsity",f"int8_Prune_model_weights_{final_sparsity:.1f}TargetSparsity.xlsx")
    #quantizationFunc.save_weights_and_biases_to_excel(pruned_tflite_file_fp16,f"model/TargetSparsity_{final_sparsity:.1f}/weight_in_{final_sparsity:.1f}TargetSparsity",f"fp16_Prune_model_weights_{final_sparsity:.1f}TargetSparsity.xlsx")
    # 绘制混淆矩阵热力图
    calculate_result.plotHeatMap(Y_test, Y_pred_purning, f"./model/TargetSparsity_{final_sparsity:.1f}/ResultFig_in_{final_sparsity:.1f}TargetSparsity", f"prune_heatmap_{final_sparsity:.1f}TargetSparsity.png")
    # 绘制训练和验证精度变化图表
    calculate_result.plotAccuracy(purning_history,f"./model/TargetSparsity_{final_sparsity:.1f}/ResultFig_in_{final_sparsity:.1f}TargetSparsity", f"prune_accuracy_{final_sparsity:.1f}TargetSparsity.png")
    # 绘制损失函数曲线和训练的变化图表
    calculate_result.plot_loss_curve(purning_history,f"./model/TargetSparsity_{final_sparsity:.1f}/ResultFig_in_{final_sparsity:.1f}TargetSparsity", f"prune_loss_{final_sparsity:.1f}TargetSparsity.png")
    # 绘制 Validation Accuracy 随 Sparsity 变化 和 Validation Loss 随 Sparsity 变化 的图
    #sparsity_accuracy_callback = callbacks[-1]  # 获取最后一个回调，即 SparsityAccuracyCallback
    #sparsity_accuracy_callback.plot_results(save_dir=f"./model/TargetSparsity_{final_sparsity:.1f}/ResultFig_in_{final_sparsity:.1f}TargetSparsity",filename=f"AccuracyBySparsity_{final_sparsity:.1f}TargetSparsity.png")

    print("After pruning, calculate the acc, sen, spec, ppv of the model_for_puring:")
    calculate_result.print_performance_metrics(Y_test, Y_pred,num_classes=5)
    if (R_WEIGHT):
        with open(f'./model/TargetSparsity_{final_sparsity:.1f}/TargetSparsity_{final_sparsity:.1f}_pruned_model_fp16.tflite', 'rb') as f:
            model_data = f.read()

        interpreter = tf.lite.Interpreter(model_content=model_data)
        input_details = interpreter.get_input_details()
        print("Input Details Before Resizing:", input_details)
        new_input_shape = [1, 320, 1]  # 根据模型的要求调整输入形状
        interpreter.resize_tensor_input(input_details[0]['index'], new_input_shape, strict=True)
        interpreter.allocate_tensors()

        test_accuracy = evaluate_model(interpreter, X_test, Y_test)
        print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
        print('Pruned TF test accuracy:', pruning_acc)

    return no_pruning_acc, pruning_acc #, test_accuracy


def representative_dataset_gen():
    """
    生成代表性数据集，确保输入为 [batch_size, input_length] 的一维数据。
    """
    for _ in range(100):  # 100是代表性样本数 320是输入张量的长度
        yield [np.random.random((1, 320)).astype(np.float32)]