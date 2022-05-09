import tensorflow as tf

from tensorflow import keras
from keras import mixed_precision
from keras import layers
from keras.layers import BatchNormalization


class Model:
    def __init__(self, image_dimensions, num_classes, image_channels):
        self.image_dimensions = image_dimensions
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.model = None

    def create_model(self, type, distribution_strategy):
        if distribution_strategy == "mirrored":
            distribution_strategy = tf.distribute.MirroredStrategy()
        else:
            distribution_strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with distribution_strategy.scope():

            input_shape = (self.image_dimensions[0], self.image_dimensions[1], self.image_channels)
            input_layer = tf.keras.layers.Input(input_shape)

            if type == "transfer_learning":
                base_model = tf.keras.applications.ResNet101V2(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(self.image_dimensions[0], self.image_dimensions[1], self.image_channels),
                    pooling="avg",
                    classifier_activation=None, # only used when you are including the top
                )
                base_model.trainable = False
                base_layers = base_model(input_layer, training=False)
            else:
                # Base
                # base_layers = layers.experimental.preprocessing.Rescaling(1./255, name='bl_1')(input_layer)
                base_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='bl_2')(input_layer)
                base_layers = layers.MaxPooling2D(name='bl_3')(base_layers)
                base_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='bl_4')(base_layers)
                base_layers = layers.MaxPooling2D(name='bl_5')(base_layers)
                base_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='bl_6')(base_layers)
                base_layers = layers.MaxPooling2D(name='bl_7')(base_layers)
                base_layers = layers.Flatten(name='bl_8')(base_layers)

            # Classifier branch
            classifier_branch = layers.Dense(128, activation='relu', name='cl_1')(base_layers)
            classifier_branch = layers.Dense(self.num_classes, name='cl_head')(classifier_branch)
            # logisitic regression for each possible class

            # Localizer branch
            locator_branch = layers.Dense(128, activation='relu', name='bb_1')(base_layers)
            locator_branch = layers.Dense(64, activation='relu', name='bb_2')(locator_branch)
            locator_branch = layers.Dense(32, activation='relu', name='bb_3')(locator_branch)
            locator_branch = layers.Dense(4, activation='sigmoid', name='bb_head')(locator_branch)
            # output 4 floats, MSE loss metric

            model = tf.keras.Model(input_layer, outputs=[classifier_branch,locator_branch])
        self.model = model
        return model

    def add_loss_function(self, initial_learning_rate, decay_steps, decay_rate):
        """Compiles the model object with 2 loss functions and metrics"""
        losses = { 
            "cl_head":tf.keras.losses.SparseCategoricalCrossentropy(),        
            "bb_head":tf.keras.losses.MSE
        }

        metrics = { 
            "cl_head": "accuracy",
            "bb_head": "mean_squared_error"
        }

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(loss=losses, optimizer=opt, metrics=metrics)

    def get_model(self):
        return self.model
