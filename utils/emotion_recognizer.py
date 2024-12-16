import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Input
from tensorflow.keras.layers import GlobalAveragePooling2D

class EmotionRecognizer:
    def __init__(self, input_shape, sequence_length, num_classes):
        self.input_shape = input_shape  # Shape of each frame (height, width, channels)
        self.sequence_length = sequence_length  # Number of frames in each sequence
        self.num_classes = num_classes  # Number of emotion classes

        self.model = self._build_model()

    def _build_model(self):
        """
        Builds a CNN-LSTM model for emotion recognition.
        """
        # CNN for feature extraction
        cnn_base = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            GlobalAveragePooling2D()
        ])

        # Wrap CNN with TimeDistributed to handle sequences of frames
        input_layer = Input(shape=(self.sequence_length,) + self.input_shape)
        cnn_out = TimeDistributed(cnn_base)(input_layer)

        # LSTM for temporal analysis
        lstm_out = LSTM(128, return_sequences=False, dropout=0.5)(cnn_out)

        # Fully connected output layer
        output_layer = Dense(self.num_classes, activation='softmax')(lstm_out)

        # Build the final model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_generator, val_generator, epochs=10):
        """
        Trains the CNN-LSTM model.
        """
        self.model.fit(train_generator, validation_data=val_generator, epochs=epochs)

    def predict(self, sequence):
        """
        Predicts emotions for a sequence of frames.
        """
        return self.model.predict(np.expand_dims(sequence, axis=0))
