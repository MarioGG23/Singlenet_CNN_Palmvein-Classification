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

def lectura_datos(data_dir):
    IMG_SIZE = 128  # Tamaño de las imágenes
    test_data = []
    class_map_individuals = {}  # Mapeo de nombres de subdirectorios a etiquetas numéricas
    class_map_categories = {}  # Mapeo de categorías a etiquetas numéricas
    n_individuals = 0
    n_categories = 0

    # Función de ordenación personalizada
    def sort_key(subdir):
        category, number = subdir.split('_')
        return int(number)

    # Ordenar subdirectorios usando la función personalizada
    for subdir in sorted(os.listdir(data_dir), key=sort_key):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            category = subdir.split('_')[0]
            sujeto = subdir.split('_')[1]  # Obtener la categoría del sujeto
            individual = subdir  # Nombre del subdirectorio es el identificador único del individuo

            if category not in class_map_categories:
                class_map_categories[category] = n_categories
                n_categories += 1

            if individual not in class_map_individuals:
                class_map_individuals[individual] = n_individuals
                n_individuals += 1

            individual_label = class_map_individuals[individual]
            category_label = class_map_categories[category]

            for img_name in sorted(os.listdir(subdir_path)):
                if img_name.endswith('.png'):
                    img_path = os.path.join(subdir_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Leer la imagen en escala de grises
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Redimensionar la imagen
                    test_data.append((img, sujeto, category))

    return test_data, n_individuals, n_categories, IMG_SIZE

def procesamiento_datos(test_data, IMG_SIZE):
    test_samples = []
    individual_labels = []
    category_labels = []

    for s, individual_label, category_label in test_data:
        test_samples.append(s)
        individual_labels.append(individual_label)
        category_labels.append(category_label)

    test_samples = np.array(test_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    individual_labels = np.array(individual_labels)
    category_labels = np.array(category_labels)

    meanTest = np.mean(test_samples, axis=0)
    test_samples = test_samples - meanTest
    test_samples = test_samples / 255


    return test_samples, individual_labels, category_labels

if __name__ == "__main__":
    data_dir = '/home/alumnos/mgonzalez/magister/images/unet_labeled/'

    # Leer y procesar datos
    test_data, n_individuals, n_categories, IMG_SIZE = lectura_datos(data_dir)
    test_samples, individual_labels, category_labels = procesamiento_datos(test_data, IMG_SIZE)

    input_shape = test_samples.shape[1:]
    nclasses = n_individuals  # Número de individuos únicos
    convolutional_layers = 5
    filters_size = [[7, 7], [5, 5], [3, 3], [3, 3], [2, 2]]
    filters_numbers = [96, 192, 284, 512, 1024]
    pool_size = (2, 2)
    weight_decay = 1e-5
    dropout = 0.65
    optimizer = Adam(learning_rate=0.01)

    model = build_model(input_shape, nclasses, convolutional_layers, filters_size, filters_numbers, pool_size, weight_decay, dropout, optimizer)

    model.fit(test_samples, to_categorical(individual_labels, nclasses), batch_size=32, epochs=10, validation_split=0.2)

    flattened_vectors = model.layers[-2].output
    intermediate_model = Model(inputs=model.input, outputs=flattened_vectors)

    flattened_vectors_test = intermediate_model.predict(test_samples)

    # n_clusters = len(np.unique(category_labels))
    # print(n_clusters)

    # # Aplicar K-means
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # kmeans.fit(flattened_vectors_test)

    # # Obtener las etiquetas predichas por K-means
    # predicted_labels = kmeans.labels_
    # print(predicted_labels)

    # # Evaluar la coincidencia con las etiquetas originales usando Adjusted Rand Index (ARI)
    # ari_score = adjusted_rand_score(category_labels, predicted_labels)
    # print(f'Adjusted Rand Index (ARI) Score: {ari_score}')

    # # Ajustar DBSCAN
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # dbscan.fit(flattened_vectors_test)

    # # Obtener las etiquetas predichas por DBSCAN
    # predicted_labels_dbscan = dbscan.labels_

    # # Evaluar con ARI
    # ari_score_dbscan = adjusted_rand_score(category_labels, predicted_labels_dbscan)
    # print(f'Adjusted Rand Index (ARI) Score for DBSCAN: {ari_score_dbscan}')

    del model, intermediate_model, test_samples, individual_labels, test_data
    tf.keras.backend.clear_session()
