import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# TODO: maybe there is some better way of passing in the sample size....
# TODO: consider switching to Functional API if we want more complex models
def get_model(str_model, cfg):
    if str_model == 'arch_001':
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',
                         input_shape=cfg.sample_size))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(BatchNormalization(center=True, scale=True))
        model.add(Dropout(0.5))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(BatchNormalization(center=True, scale=True))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=cfg.lr),
                      metrics=['accuracy'])
        model.summary()

        return model

    else:
        raise NotImplementedError('Specified architecture is not implemented')


# TODO: check this implementation later
def get_callbacks(cfg):
    es = EarlyStopping(monitor='val_loss', min_delta=cfg.es_min_delta, patience=cfg.es_patience,
                       restore_best_weights=True)

    return [es]
