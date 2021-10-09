import re

import tensorflow as tf
from efficientnet import tfkeras
from tensorflow.keras import Model

from src.factory import Factory
from src.log import myLogger


class ModelFactory(Factory[Model]):
    def __init__(
        self,
        model_type: str,
        img_size: int,
        img_channel: int,
        do_rate: float,
        weights: str,
        logger: myLogger,
    ):
        super().__init__(
            model_type=model_type,
            img_size=img_size,
            img_channel=img_channel,
            do_rate=do_rate,
            weights=weights,
            logger=logger,
        )

    def _create(
        self,
    ) -> Model:
        inputs = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 3))
        if re.match("EfficientNetB[0-7]", self.model_type):
            model_layer = getattr(tfkeras, self.model_type)(
                input_shape=(self.img_size, self.img_size, self.img_channel),
                weights=self.weights,
                include_top=False,
            )
        else:
            raise NotImplementedError

        x = model_layer(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dropout(self.do_rate)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        return model
