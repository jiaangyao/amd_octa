import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, Conv2D, AvgPool2D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, LeakyReLU, ReLU, concatenate, MaxPool2D, Add, GlobalAveragePooling3D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l1, l2
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

    if str_model == 'arch_002':
        """
        Test 1: 10 repeated runs
        cfg.lr = 5e-5
        cfg.lam = 1e-5
        cfg.downscale_size = [350, 350]
        
        number of parameters: 330k
        
        Train, valid, test acc: [86.8+5.4, 78.1+7.7, 68.1+7.8] (avg + standard error)
        
        Test 2: 5 repeated runs
        cfg.lr = 5e-5
        cfg.lam = 1e-5
        cfg.downscale_size = [512, 512]
        
        number of parameters: 893k
        
        Train, valid, test acc: [42.8+15.4, 41.0+9.1, 40.6+13.9] (avg + standard error)
        """

        angiography_inputs = Input(shape=cfg.sample_size[0])
        structure_inputs = Input(shape=cfg.sample_size[0])
        bscan_inputs = Input(shape=cfg.sample_size[1])

        x = Conv3D(5, kernel_size=(40, 40, 2), kernel_initializer='he_uniform',
                   kernel_regularizer=l1(cfg.lam))(angiography_inputs)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        x = Conv3D(10, kernel_size=(20, 20, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(20, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(30, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(40, kernel_size=(4, 4, 1), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Dense layer
        x = Flatten()(x)
        x = Dropout(0.05)(x)

        x = Dense(64)(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(32)(x)
        x = LeakyReLU()(x)

        y = Dense(cfg.num_classes, activation='softmax')(x)

        model = Model(inputs=[angiography_inputs, structure_inputs, bscan_inputs], outputs=y)
        model.summary()

        model.compile(optimizer=RMSprop(lr=cfg.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    if str_model == 'arch_003':
        """
        Test 1: 10 repeated runs
        cfg.lr = 5e-5
        cfg.lam = 1e-5
        cfg.downscale_size = [512, 512]

        number of parameters : 880k

        Train, valid, test acc: [90.6+6.3, 68.5+7.0, 68.8+9.6] (avg + standard error)
        
        lr is too high?
        think of doing global average pooling 
        """

        angiography_inputs = Input(shape=cfg.sample_size[0])
        structure_inputs = Input(shape=cfg.sample_size[0])
        bscan_inputs = Input(shape=cfg.sample_size[1])

        x = Conv3D(5, kernel_size=(40, 40, 2), kernel_initializer='he_uniform',
                   kernel_regularizer=l1(cfg.lam))(angiography_inputs)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        x = Conv3D(10, kernel_size=(30, 30, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(20, kernel_size=(20, 20, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(30, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Dense layer
        x = Flatten()(x)
        x = Dropout(0.05)(x)

        x = Dense(64, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(16, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)

        y = Dense(cfg.num_classes, activation='softmax')(x)

        model = Model(inputs=[angiography_inputs, structure_inputs, bscan_inputs], outputs=y)
        model.summary()

        model.compile(optimizer=RMSprop(lr=cfg.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    if str_model == 'arch_004':
        """
        Test 1: 10 repeated runs
        cfg.lr = 5e-5
        cfg.lam = 1e-5
        cfg.downscale_size = [512, 512]

        number of parameters : 694k

        Train, valid, test acc: [81.5+5.9, 74.1+1.3, 68.5+4.7] (avg + standard error)

        lr is too high?
        think of doing global average pooling 
        """

        angiography_inputs = Input(shape=cfg.sample_size[0])
        structure_inputs = Input(shape=cfg.sample_size[0])
        bscan_inputs = Input(shape=cfg.sample_size[1])

        x = Conv3D(5, kernel_size=(40, 40, 2), kernel_initializer='he_uniform',
                   kernel_regularizer=l1(cfg.lam))(angiography_inputs)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        x = Conv3D(10, kernel_size=(30, 30, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(20, kernel_size=(20, 20, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(30, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(40, kernel_size=(5, 5, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(50, kernel_size=(3, 3, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(1, 1, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        # Dense layer
        x = Flatten()(x)
        x = Dropout(0.05)(x)

        x = Dense(64, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(16, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)

        y = Dense(cfg.num_classes, activation='softmax')(x)

        model = Model(inputs=[angiography_inputs, structure_inputs, bscan_inputs], outputs=y)
        model.summary()

        model.compile(optimizer=RMSprop(lr=cfg.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    if str_model == 'arch_005':
        """
        Test 1: 10 repeated runs
        cfg.lr = 5e-5
        cfg.lam = 1e-5
        cfg.downscale_size = [512, 512]

        number of parameters : 694k

        Train, valid, test acc: [79.0+4.6, 70.4+8.0, 63.9+8.2] (avg + standard error)

        lr is too high?
        think of doing global average pooling 
        """

        angiography_inputs = Input(shape=cfg.sample_size[0])
        structure_inputs = Input(shape=cfg.sample_size[0])
        bscan_inputs = Input(shape=cfg.sample_size[1])

        x = Conv3D(5, kernel_size=(40, 40, 2), kernel_initializer='he_uniform',
                   kernel_regularizer=l1(cfg.lam))(angiography_inputs)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        x = Conv3D(10, kernel_size=(30, 30, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(20, kernel_size=(20, 20, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(30, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(40, kernel_size=(5, 5, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(50, kernel_size=(3, 3, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        # Dense layer
        x = Flatten()(x)
        x = Dropout(0.05)(x)

        x = Dense(64, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(16, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)

        y = Dense(cfg.num_classes, activation='softmax')(x)

        model = Model(inputs=[angiography_inputs, structure_inputs, bscan_inputs], outputs=y)
        model.summary()

        model.compile(optimizer=RMSprop(lr=cfg.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    if str_model == 'arch_006':
        """
        Test 1: 10 repeated runs
        cfg.lr = 5e-5
        cfg.lam = 1e-5
        cfg.downscale_size = [512, 512]

        number of parameters: 912k

        Train, valid, test acc: [79.0+4.6, 70.4+8.0, 63.9+8.2] (avg + standard error)

        lr is too high?
        think of doing global average pooling 
        """

        angiography_inputs = Input(shape=cfg.sample_size[0])
        structure_inputs = Input(shape=cfg.sample_size[0])
        bscan_inputs = Input(shape=cfg.sample_size[1])

        x = Conv3D(5, kernel_size=(40, 40, 2), kernel_initializer='he_uniform',
                   kernel_regularizer=l1(cfg.lam))(angiography_inputs)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        x = Conv3D(10, kernel_size=(30, 30, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(20, kernel_size=(20, 20, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(30, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(40, kernel_size=(5, 5, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(50, kernel_size=(3, 3, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x_angio = Flatten()(x)

        x = Conv3D(5, kernel_size=(40, 40, 2), kernel_initializer='he_uniform',
                   kernel_regularizer=l1(cfg.lam))(structure_inputs)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        x = Conv3D(5, kernel_size=(30, 30, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(10, kernel_size=(20, 20, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(20, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(30, kernel_size=(10, 10, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x_struct = Flatten()(x)

        x = concatenate([x_angio, x_struct])
        x = Dropout(0.05)(x)

        x = Dense(64, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(16, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)

        y = Dense(cfg.num_classes, activation='softmax')(x)

        model = Model(inputs=[angiography_inputs, structure_inputs, bscan_inputs], outputs=y)
        model.summary()

        model.compile(optimizer=RMSprop(lr=cfg.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    if str_model == 'arch_007':
        """
        Test 1: 10 repeated runs
        cfg.lr = 5e-5
        cfg.lam = 1e-5
        cfg.downscale_size = [512, 512]

        number of parameters : 642k

        Train, valid, test acc: [79.0+4.6, 70.4+8.0, 63.9+8.2] (avg + standard error)

        lr is too high?
        think of doing global average pooling 
        """

        angiography_inputs = Input(shape=cfg.sample_size[0])
        structure_inputs = Input(shape=cfg.sample_size[0])
        bscan_inputs = Input(shape=cfg.sample_size[1])

        x = Conv3D(5, kernel_size=(40, 40, 2), kernel_initializer='he_uniform',
                   kernel_regularizer=l1(cfg.lam))(angiography_inputs)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        x = Conv3D(10, kernel_size=(30, 30, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(20, kernel_size=(20, 20, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(30, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(40, kernel_size=(5, 5, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(50, kernel_size=(3, 3, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x_angio = GlobalAveragePooling3D()(x)

        x = Conv3D(5, kernel_size=(40, 40, 2), kernel_initializer='he_uniform',
                   kernel_regularizer=l1(cfg.lam))(structure_inputs)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        x = Conv3D(5, kernel_size=(30, 30, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(10, kernel_size=(20, 20, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(20, kernel_size=(10, 10, 2), kernel_initializer='he_uniform', kernel_regularizer=l1(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(30, kernel_size=(10, 10, 1), kernel_initializer='he_uniform', kernel_regularizer=l2(cfg.lam))(x)
        x = ReLU()(x)
        x = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x_struct = GlobalAveragePooling3D()(x)

        x = concatenate([x_angio, x_struct])

        x = Dense(64, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(16, kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)

        y = Dense(cfg.num_classes, activation='softmax')(x)

        model = Model(inputs=[angiography_inputs, structure_inputs, bscan_inputs], outputs=y)
        model.summary()

        model.compile(optimizer=RMSprop(lr=cfg.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    else:
        raise NotImplementedError('Specified architecture is not implemented')


# TODO: check this implementation later
def get_callbacks(cfg):
    es = EarlyStopping(monitor='val_loss', min_delta=cfg.es_min_delta, patience=cfg.es_patience,
                       restore_best_weights=True, verbose=1)

    return [es]
