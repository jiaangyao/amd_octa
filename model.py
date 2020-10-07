import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, Conv2D, AvgPool2D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, ReLU, concatenate, MaxPool2D, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping


# TODO: maybe there is some better way of passing in the sample size....
# Everything in the same function
def get_model(str_model, cfg):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), ' Physical GPUs, ', len(logical_gpus), " Logical GPUs")

        except RuntimeError as e:
            print(e)

    if str_model == 'arch_001':
        angiography_inputs = Input(shape=cfg.sample_size[0])
        structure_inputs = Input(shape=cfg.sample_size[0])
        bscan_inputs = Input(shape=cfg.sample_size[1])

        # this is information extracted from structural information
        x = Conv3D(16, kernel_size=(20, 20, 1), activation='relu', kernel_initializer='he_uniform')(structure_inputs)
        x = MaxPooling3D(pool_size=(5, 5, 1), strides=(2, 2, 1))(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Conv3D(32, kernel_size=(10, 10, 1), activation='relu', kernel_initializer='he_uniform')(x)
        x = MaxPooling3D(pool_size=(5, 5, 1), strides=(2, 2, 1))(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Conv3D(64, kernel_size=(10, 10, 1), activation='relu', kernel_initializer='he_uniform')(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Flatten()(x)
        structure_features = Dropout(0.1)(x)

        # this is the information extracted from bscan information
        x = Conv2D(6, kernel_size=(10, 10), strides=1, padding='valid', kernel_initializer='he_uniform')(bscan_inputs)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(4, 4), strides=2)(x)
        x = Dropout(0.01)(x)

        x = Conv2D(16, kernel_size=(8, 8), strides=1, padding='valid', kernel_initializer='he_uniform')(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(4, 4), strides=2)(x)
        x = Dropout(0.03)(x)

        x = Conv2D(64, kernel_size=(5, 5), strides=1, padding='valid', kernel_initializer='he_uniform')(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(4, 4), strides=2)(x)
        x = Dropout(0.05)(x)

        x = Flatten()(x)
        bscan_features = Dropout(0.1)(x)

        # finally this is information from the angiography data
        x = Conv3D(32, kernel_size=(15, 15, 1), activation='relu', kernel_initializer='he_uniform')(angiography_inputs)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Conv3D(64, kernel_size=(10, 10, 1), activation='relu', kernel_initializer='he_uniform')(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Conv3D(64, kernel_size=(10, 10, 1), activation='relu', kernel_initializer='he_uniform')(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        # Dense layer
        x = Flatten()(x)
        angiography_features = Dropout(0.1)(x)

        x = Dense(128, kernel_initializer='he_uniform')(angiography_features)
        angiography_dense = Dropout(0.1)(x)

        aux_info_combined = concatenate([structure_features, bscan_features])
        x = Dense(32, kernel_initializer='he_uniform')(aux_info_combined)
        aux_dense = Dropout(0.3)(x)

        x_combined = concatenate([angiography_dense, aux_dense])
        x = Dense(128, kernel_initializer='he_uniform')(x_combined)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        combined_dense = Dropout(0.3)(x)

        # skip connection
        added = Add()([combined_dense, angiography_dense])

        x = Dense(64, kernel_initializer='he_uniform')(added)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)

        y = Dense(cfg.num_classes, activation='softmax')(x)

        model = Model(inputs=[angiography_inputs, structure_inputs, bscan_inputs], outputs=y)
        model.summary()

        model.compile(optimizer=RMSprop(lr=cfg.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    else:
        raise NotImplementedError('Specified architecture is not implemented')


# Separate network implementation
# TODO: actually thinking back this might not be the best idea for implementing this thing...
def structure_conv3d(str_model, cfg):
    if str_model == 'arch_001':
        inputs = Input(shape=cfg.sample_size[0])

        x = Conv3D(32, kernel_size=(20, 20, 1), activation='relu', kernel_initializer='he_uniform')(inputs)
        x = MaxPooling3D(pool_size=(5, 5, 1), strides=2)(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Conv3D(64, kernel_size=(10, 10, 1), activation='relu', kernel_initializer='he_uniform')(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=2)(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Flatten()(x)
        x = Dropout(0.1)(x)

        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        y = Dense(1, activation='sigmoid')(x)

        structure_conv3d = Model(inputs=inputs, outputs=y, name='structure_conv3d')
        structure_conv3d.summary()

        return structure_conv3d
    else:
        raise NotImplementedError('Specified architecture is not implemented')


def angiography_conv3d(str_model, cfg):
    if str_model == 'arch_001':
        inputs = Input(shape=cfg.sample_size[0])

        x = Conv3D(32, kernel_size=(12, 12, 1), activation='relu', kernel_initializer='he_uniform')(inputs)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=2)(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Conv3D(64, kernel_size=(10, 10, 1), activation='relu', kernel_initializer='he_uniform')(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=2)(x)
        x = BatchNormalization(center=True, scale=True)(x)
        x = Dropout(0.05)(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        y = Dense(1, activation='sigmoid')(x)

        angiography_conv3d = Model(inputs=inputs, outputs=y, name='angiography_conv3d')
        angiography_conv3d.summary()

        return angiography_conv3d
    else:
        raise NotImplementedError('Specified architecture is not implemented')


def bscan_conv2d(str_model, cfg):
    if str_model == 'arch_001':
        inputs = Input(shape=cfg.sample_size[1])

        x = Conv2D(6, kernel_size=(10, 10), strides=1, padding='valid', kernel_initializer='he_uniform')(inputs)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = AvgPool2D(pool_size=(2, 2), strides=2)(x)
        x = Dropout(0.01)(x)

        x = Conv2D(16, kernel_size=(8, 8), strides=1, padding='valid', kernel_initializer='he_uniform')(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = AvgPool2D(pool_size=(2, 2), strides=2)(x)
        x = Dropout(0.03)(x)

        x = Conv2D(120, kernel_size=(5, 5), strides=1, padding='valid', kernel_initializer='he_uniform')(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = AvgPool2D(pool_size=(2, 2), strides=2)(x)
        x = Dropout(0.05)(x)

        x = Flatten()(x)
        x = Dropout(0.1)(x)

        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dropout(0.3)(x)

        x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dropout(0.3)(x)

        y = Dense(1, activation='sigmoid')(x)

        angiography_conv3d = Model(inputs=inputs, outputs=y, name='angiography_conv3d')
        angiography_conv3d.summary()

        return angiography_conv3d
    else:
        raise NotImplementedError('Specified architecture is not implemented')


# TODO: check this implementation later
def get_callbacks(cfg):
    es = EarlyStopping(monitor='val_loss', min_delta=cfg.es_min_delta, patience=cfg.es_patience,
                       restore_best_weights=True)

    return [es]
