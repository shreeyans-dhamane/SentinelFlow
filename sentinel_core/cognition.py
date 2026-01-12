import tensorflow as tf
from tensorflow.keras import layers
from .interfaces import BaseModule

class TemporalReasoningUnit(BaseModule):
    def __init__(self, feature_dim=1280):
        super().__init__("CognitiveUnit")
        self.feature_dim = feature_dim
        self.model = None

    def initialize(self) -> bool:
        inputs = layers.Input(shape=(None, self.feature_dim))
        
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
        x = layers.Dropout(0.4)(x) 
        x = layers.Bidirectional(layers.LSTM(64))(x)
        
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.Model(inputs, output)
        return True

    def process(self, features):
        if features is None: return 0.0
        # Returns raw probability (0.0 to 1.0)
        risk_probability = self.model(features, training=False)
        return float(risk_probability.numpy()[0][0])

    def shutdown(self):
        del self.model
