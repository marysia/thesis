# source: https://www.kaggle.com/mumech/data-science-bowl-2017/loading-and-processing-the-sample-images/discussion

import keras

# building the network

nn = keras.models.Sequential([
        keras.layers.convolutional.Convolution3D(32, 3, 3, 3, border_mode='same', activation='relu', input_shape=train_features.shape[1:], dim_ordering='th'),
        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
        keras.layers.convolutional.Convolution3D(32, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
        keras.layers.convolutional.Convolution3D(64, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
        keras.layers.convolutional.Convolution3D(64, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
        keras.layers.convolutional.Convolution3D(128, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
        keras.layers.convolutional.Convolution3D(256, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
        keras.layers.convolutional.AveragePooling3D((4, 4, 4), dim_ordering='th'),
        keras.layers.core.Flatten(),
        keras.layers.core.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.core.Dense(1, activation='sigmoid')
    ])
nn.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Finally train the CNN
# nn.fit(train_features, train_labels, batch_size=1, validation_split=0.1, nb_epoch=1)
