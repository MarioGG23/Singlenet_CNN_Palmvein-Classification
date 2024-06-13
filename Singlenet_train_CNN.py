import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


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

    model_output = Flatten()(model_output)  # Capa de flatten añadida

    model_output = Dense(1024)(model_output)  # 512, 1024, 2048, 4096
    model_output = Dropout(dropout)(model_output)

    model_output = Dense(512)(model_output)
    model_output = Dropout(dropout)(model_output)

    model_output = Dense(nclasses, activation='softmax', name='id')(model_output)

    model = Model(inputs=model_input, outputs=model_output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model

def lectura_datos(data_dir, bd):
    IMG_SIZE = 128 
    test_data = []
    nclases = 0
    class_map = {} 

    for subdir in sorted(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            class_name = subdir.split('_')[0]  # Obtener el nombre de la clase
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
    mode = 'all'
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

def k_means(test_labels, flattened_vectors_test):

    n_clusters = len(np.unique(test_labels))
    print(n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(flattened_vectors_test)

    predicted_labels = kmeans.labels_

    # Evaluar la coincidencia con las etiquetas originales usando Adjusted Rand Index (ARI)
    ari_score = adjusted_rand_score(test_labels, predicted_labels)

    print(f'Adjusted Rand Index (ARI) Score: {ari_score}')

    return predicted_labels

def calculate_accuracy(true_labels, predicted_labels):
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    print("matriz de confusion: ", confusion_mat)
    # Usar el método de asignación lineal para encontrar la mejor correspondencia
    row_ind, col_ind = linear_sum_assignment(-confusion_mat)
    print("row_ind: ", row_ind)
    print("col_ind: ", col_ind)
    accuracy = confusion_mat[row_ind, col_ind].sum() / confusion_mat.sum()
    
    return accuracy

def optimal_number_of_clusters_elbow(flattened_vectors_test, max_k=10, filename='elbow_plot.png'):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(flattened_vectors_test)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title('Metodo codo para k optimo')
    plt.xlabel('Numero de clusters')
    plt.ylabel('Inercia')
    
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    bd_ruta = {'Unet': '/home/alumnos/mgonzalez/magister/images/unet_labeled/'}

    for bd in bd_ruta:
        test_data, nclases, IMG_SIZE = lectura_datos(bd_ruta[bd], bd) #ojo con nclases
        test_samples, test_labels = procesamiento_datos(test_data, nclases, IMG_SIZE)

        input_shape = test_samples.shape[1:]
        nclasses = nclases
        convolutional_layers = 5
        filters_size = [[7, 7], [5, 5], [3, 3], [3, 3], [2, 2]]
        filters_numbers = [96, 192, 284, 512, 1024]
        pool_size = (2, 2)
        weight_decay = 1e-5
        dropout = 0.5
        optimizer = Adam(learning_rate=0.0001)

        train_samples, val_samples, train_labels, val_labels = train_test_split(
            test_samples, test_labels, test_size=0.2, stratify=test_labels)

        model = build_model(input_shape, nclasses, convolutional_layers, filters_size, filters_numbers, pool_size, weight_decay, dropout, optimizer)

        ##Entrenar el modelo
        history = model.fit(train_samples, to_categorical(train_labels, nclasses),
                            batch_size=32,
                            epochs=10,
                            validation_data=(val_samples, to_categorical(val_labels, nclasses)))

        final_train_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        print(f'Final training accuracy: {final_train_accuracy}')
        print(f'Final validation accuracy: {final_val_accuracy}')

        model.save('model.keras')

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Listar todas las capas del modelo con sus índices y nombres
        # for i, layer in enumerate(model.layers):
        #     print(f"Índice: {i}, Nombre de la capa: {layer.name}, Tipo de capa: {layer.__class__.__name__}")

        # penultimate_layer_name = model.layers[-2].name
        # print(f"La penúltima capa es: {penultimate_layer_name}")

        # Crear un nuevo modelo para obtener los flatten
        # flattened_vectors = model.layers[-2].output #4
        # intermediate_model = Model(inputs=model.input, outputs=flattened_vectors)
        # intermediate_model.save('intermediate_model.h5')
        # intermediate_model = load_model('intermediate_model.h5')
        # flattened_vectors_test = intermediate_model.predict(test_samples)


        # optimal_number_of_clusters_elbow(flattened_vectors_test, max_k=10, filename='elbow_plot.png')
        # kmeans_labels = k_means(test_labels, flattened_vectors_test)

        # #kmeans_labels = kmeans.labels_
        # accuracy = calculate_accuracy(test_labels, kmeans_labels)

        # print(f"Porcentaje de aciertos: {accuracy * 100:.2f}%")


        # print(kmeans_labels)
        # print(len(kmeans_labels))

        # print(len(test_labels))
        # print(test_labels)
        # cont = 0
        # for i in range(len(kmeans_labels)):
        #     if kmeans_labels[i] == test_labels[i]:
        #         cont+=1
        # porcentaje = cont/len(kmeans_labels)
        # print("porcentaje: ", porcentaje)

        del model, test_samples, test_labels, test_data
        tf.keras.backend.clear_session()
        break


# Final training accuracy: 0.9910312294960022
# Final validation accuracy: 0.9380000233650208

#Hiperparámetros a considerar: dropout = [0.3, 0.4, 0.5, 0.6, 0.7], weight_decay = [1e-4, 1e-5, 1e-6, 1e-7], learning_rate = [0.001, 0.0001, 0.00001], batch_size =[8, 16, 32, 64, 128] 