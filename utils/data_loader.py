from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, img_height, img_width, batch_size):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def load_data(self, train_dir, val_dir):
        """
        Loads and preprocesses the training and validation datasets.
        """
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        val_data = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        return train_data, val_data
