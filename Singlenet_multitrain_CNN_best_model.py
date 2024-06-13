import os
import cv2
import numpy as np
import csv
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import gc

def build_model(input_shape, nclasses, convolutional_layers, filters_size, filters_numbers, pool_size, weight_decay, dropout, optimizer):
    L2_norm = tf.keras.regularizers.l2(weight_decay)

    model_input = Input(shape=input_shape)
    model_output = Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(2, 2), padding="valid", kernel_regularizer=L2_norm, activation='relu',
                          data_format='channels_last')(model_input)
    model_output = BatchNormalization()(model_output)
    model_output = MaxPooling2D(pool_size=(pool_size))(model_output)

    for i in range(1, convolutional_layers):
        model_output = Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), padding="same", kernel_regularizer=L2_norm, activation='relu',
                              data_format='channels_last')(model_output)
        model_output = BatchNormalization()(model_output)
        if i != convolutional_layers - 1:
            model_output = MaxPooling2D(pool_size=(pool_size))(model_output)

    model_output = GlobalAveragePooling2D(data_format='channels_last')(model_output)
    model_output = Flatten()(model_output)
    model_output = Dense(1024)(model_output)
    model_output = Dropout(dropout)(model_output)
    model_output = Dense(512)(model_output)
    model_output = Dropout(dropout)(model_output)
    model_output = Dense(nclasses, activation='softmax', name='id')(model_output)

    model = Model(inputs=model_input, outputs=model_output)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def lectura_datos(data_dir, bd):
    IMG_SIZE = 128 
    test_data = []
    nclases = 0
    class_map = {} 
    for subdir in sorted(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            class_name = subdir.split('_')[0]
            if class_name not in class_map:
                class_map[class_name] = nclases
                nclases += 1
            class_label = class_map[class_name]
            for img_name in sorted(os.listdir(subdir_path)):
                if img_name.endswith('.png'):
                    img_path = os.path.join(subdir_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    test_data.append((img, class_label))
    return test_data, nclases, IMG_SIZE


def procesamiento_datos(test_data, nclases, IMG_SIZE):
    test_samples = []
    test_labels = []
    for s, l in test_data:
        test_samples.append(s)
        test_labels.append(l)
    test_samples = np.array(test_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_labels = np.array(test_labels)
    meanTest = np.mean(test_samples, axis=0)
    test_samples = test_samples - meanTest
    test_samples = test_samples / 255
    test_samples = test_samples.reshape(test_samples.shape[0], IMG_SIZE, IMG_SIZE, 1)
    return test_samples, test_labels

# Hiperparámetros a considerar salida.txt
# dropout_values = [0.3, 0.4, 0.5, 0.6, 0.7]
# weight_decay_values = [1e-4, 1e-5, 1e-6, 1e-7]
# learning_rate_values = [0.001, 0.0001, 0.00001]
# batch_size_values = [8, 16, 32, 64, 128]

#salida2.txt
# dropout_values = [0.4, 0.5, 0.6]
# weight_decay_values = [1e-4, 1e-5, 1e-6]
# learning_rate_values = [0.001, 0.0001, 0.00001]
# batch_size_values = [16, 32, 64]

if __name__ == "__main__":

    dropout_values = [0.4, 0.5, 0.6]
    weight_decay_values = [1e-4, 1e-5, 1e-6]
    learning_rate_values = [0.0001]
    batch_size_values = [16, 32, 64]


    bd_ruta = {'Unet': '/home/alumnos/mgonzalez/magister/images/unet_labeled/'}
    contT1a = 0
    contT1b = 0
    contT1c = 0
    contT2 = 0
    contT3 = 0
    # Leer datos
    for bd in bd_ruta:
        test_data, nclases, IMG_SIZE = lectura_datos(bd_ruta[bd], bd)
        test_samples, test_labels = procesamiento_datos(test_data, nclases, IMG_SIZE)
        print(len(test_samples))
        print(test_labels[10:])

        for i in test_labels:
            if i == 0:
                contT1a+=1
            elif i == 1:
                contT1b+=1
            elif i == 2:
                contT1c+=1
            elif i == 3:
                contT2+=1
            else:
                contT3+=1
        print("numero de T1a: ", contT1a)
        print("numero de T1b ", contT1b)
        print("numero de T1c: ", contT1c)
        print("numero de T2: ", contT2)
        print("numero de T3: ", contT3)
        exit()


        input_shape = test_samples.shape[1:]
        nclasses = nclases
        convolutional_layers = 5
        filters_size = [[7, 7], [5, 5], [3, 3], [3, 3], [2, 2]]
        filters_numbers = [96, 192, 284, 512, 1024]
        pool_size = (2, 2)


        with open('resultados_experimentos2.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dropout', 'Weight_Decay', 'Learning_Rate', 'Batch_Size', 'Train_Accuracy', 'Val_Accuracy', 'Train_Loss', 'Val_Loss', 'Tiempo_Entrenamiento'])
            total_time = time.time()
            iteraciones = 0
            # Iterar sobre los valores de hiperparámetros
            for dropout in dropout_values:
                for weight_decay in weight_decay_values:
                    for learning_rate in learning_rate_values:
                        for batch_size in batch_size_values:
                            optimizer = Adam(learning_rate=learning_rate)
                            
                            train_samples, val_samples, train_labels, val_labels = train_test_split(
                                test_samples, test_labels, test_size=0.2, stratify=test_labels)

                            model = build_model(input_shape, nclasses, convolutional_layers, filters_size, filters_numbers, pool_size, weight_decay, dropout, optimizer)

                            # Medir el tiempo de entrenamiento
                            start_time = time.time()
                            history = model.fit(train_samples, to_categorical(train_labels, nclasses),
                                                batch_size=batch_size,
                                                epochs=10,
                                                validation_data=(val_samples, to_categorical(val_labels, nclasses)))
                            end_time = time.time()
                            training_time = end_time - start_time

                            final_train_accuracy = history.history['accuracy'][-1]
                            final_val_accuracy = history.history['val_accuracy'][-1]
                            final_train_loss = history.history['loss'][-1]
                            final_val_loss = history.history['val_loss'][-1]

                            print("Modelo N°: ", iteraciones)
                            print("Dropout: ", dropout)
                            print("Weight decay: ", weight_decay)
                            print("Learning rate: ", learning_rate)
                            print("Batch size: ", batch_size)
                            print(f'Final training accuracy: {final_train_accuracy}')
                            print(f'Final validation accuracy: {final_val_accuracy}')
                            print(f'Final training loss: {final_train_loss}')
                            print(f'Final validation loss: {final_val_loss}')
                            iteraciones+=1

                            writer.writerow([dropout, weight_decay, learning_rate, batch_size,
                                            final_train_accuracy, final_val_accuracy, final_train_loss, final_val_loss, training_time])
                            file.flush()

                            del model, train_samples, val_samples, train_labels, val_labels
                            tf.keras.backend.clear_session()
                            gc.collect()


#Buscar la mejor precision en el csv
#calcular el numero de muestras por clase
#sacar f1-score para cada clase
