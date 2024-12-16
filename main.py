from utils.data_loader import DataLoader
from utils.emotion_recognizer import EmotionRecognizer

# Parameters
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 48, 48, 3
SEQUENCE_LENGTH = 10
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 10

# Paths
train_dir = "data/images/train"
val_dir = "data/images/validation"
model_save_path = "models/cnn_lstm_model.h5"

# Classes for emotions
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load data
data_loader = DataLoader(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
train_data, val_data = data_loader.load_data(train_dir, val_dir)

# Build and train the model
emotion_recognizer = EmotionRecognizer(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
    sequence_length=SEQUENCE_LENGTH,
    num_classes=NUM_CLASSES
)
emotion_recognizer.train(train_data, val_data, epochs=EPOCHS)

# Save the trained model
emotion_recognizer.model.save(model_save_path)
