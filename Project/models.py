import tensorflow as tf
import keras
from keras import layers, models
import Config

# ============================================================
# BASE CNN BUILDER
# ============================================================

def CNN(x,use_bn = False, use_dropout = False,dropout_rate = Config.dropout_rate ,use_l2 = False,l2_factor=Config.l2_lambda):
    kernel_reg = keras.regularizers.l2(l2_factor) if use_l2 else None

    #-----Conv_Layer_1------
    x = layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=kernel_reg)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if use_dropout: x = layers.SpatialDropout2D(dropout_rate * 0.5)(x)
    x = layers.MaxPooling2D()(x)

    #-----Conv_Layer_2------
    x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=kernel_reg)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if use_dropout: x = layers.SpatialDropout2D(dropout_rate * 0.75)(x)
    x = layers.MaxPooling2D()(x)

    #-----Conv_Layer_3------
    x = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=kernel_reg)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if use_dropout: x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.MaxPooling2D()(x)

    #-----Conv_Layer_4:(for GRAD-CAM )------
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=kernel_reg, name='last_conv')(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if use_dropout: x = layers.SpatialDropout2D(dropout_rate)(x)
    
    x = layers.GlobalAveragePooling2D()(x)

    #-----Dense_Layer------
    x = layers.Dense(64,kernel_regularizer=kernel_reg)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if use_dropout: x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(Config.num_classes, activation='softmax')(x)

    return output


# ============================================================
# AUGMENTATION PIPELINE (toggleable)
# ============================================================

def data_aug():
    return keras.Sequential([
            keras.layers.RandomFlip('horizontal'),
            keras.layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.1)
        ])

# ============================================================
# ALL 6 MODEL VARIANTS (Explicitly Defined)
# ============================================================

VARIENTS = {
    # M1: No Regularisation,Augmentation,Dropouts
    "M1_Baseline":(
        {'use_bn':False, 'use_dropout':False},
        False
    ),

    # M2: BatchNorm only (No Regularisation,Augmentation,Dropouts)
    "M2_BN_Only":(
        {'use_bn':True, 'use_dropout':False},
        False
    ),

    # M3: Dropouts only (No Regularisation,Augmentation)
    "M3_Dropout_Only":(
        {'use_bn':False, 'use_dropout':True},
        False
    ),

    # M4: BatchNorm + Dropouts (No Regularisation,Augmentation)
    "M4_BN_Dropouts":(
        {'use_bn':True, 'use_dropout':True},
        False
    ),

    # M5: BN + Dropout + Augmentation (best practice)
    "V5_BN_Dropout_Aug": (
        {"use_bn": True, "use_dropout": True},
        True
    ),

    # M6: Heavy regularization (over-regularization probe)
    "M6_Heavy_Reg": (
        {"use_bn": True, "use_dropout": True, 'dropout_rate':Config.heavy_dropout_rate,'use_l2':True, 'l2_factor':Config.heavy_l2_lambda, },
        True
    ),
} 

def build_model(name):
    kwargs, use_aug = VARIENTS[name]

    inputs = keras.Input(shape=(Config.img_size, Config.img_size, Config.channel))
    x = inputs

    if use_aug:
        aug_layer = data_aug()
        x = aug_layer(x)

    output = CNN(x, **kwargs)
    model = models.Model(inputs, output)
    return model