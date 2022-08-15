import tensorflow as tf


class DIRECTORY:
    ROOT_DIR = 'dataset'
    TRAIN_DIR = 'dataset/training'
    VALIDATION_DIR = 'dataset/validation'


SPLIT_SIZE = .9  # chia tỉ lệ train/ test(validation)
IMAGE_EXTENSION = '.png'

# tf.keras.optimizers. để tìm thử các optimizers khác
OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate=0.001)
# tf.keras.losses. để tìm thử các losses khác, lưu ý binary chỉ dành cho phân loại 2 đối tượng, bài mình có 10 đối tượng từ 0 tới 9
LOSS = tf.keras.losses.CategoricalCrossentropy()


class DataPreparation:
    # tạo thêm ảnh bằng cách
    rotation_range: float = 40  # xoay ảnh cũ tối đa là 40 độ
    width_shift_range: float = 0.2  # di chuyển ảnh sang một bên, đồng nghĩa sẽ tạo dải màu đen ở chỗ bị thiếu
    height_shift_range: float = 0.2  # di chuyển ảnh sang một bên, đồng nghĩa sẽ tạo dải màu đen ở chỗ bị thiếu
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True  # mirror ảnh theo chiều dọc
    vertical_flip: bool = True  # mirror ảnh theo chiều ngang

    SIZE_IMAGE: tuple = (28, 28)

    """
    batch_size là để chia nhỏ ra dữ liệu huấn luyện, khi train chúng ta sẽ đưa hết toàn bộ dữ liệu lên memory, có thể gây quá tải
    """
    BATCH_SIZE_TRAIN = 1000
    BATCH_SIZE_VALID = 100


class TrainConfig:
    """
    epoch là số vòng lặp khi huấn luyện
    - Không có nghiên cứu cụ thể bao nhiêu epoch là đủ, mình dựa vào kết quả qua từng epoch để đánh giá
        + accuracy tăng cực chậm hoặc hầu như không tăng -> không cần train nữa
        + accuracy, loss không ổn định (lên rồi xuống lên rồi xuống) -> không cần train nữa
        + ...
    """
    epochs = 10
    """
        verbose là cách hiển thị terminal khi train
        - 1 là chi tiết (thời gian hoàn thành 1 epoch, etc và kết quả epoch)
        - 2 là không chi tiết (chỉ kết quả epoc)
        """
    verbose = 1


ACTIVATION = {
    "ReLU": "relu",
    "tanh": "tanh",
    "softmax": "softmax",
    "sigmoid": "sigmoid",
}


def convolutional_model():
    """
    Anh có thể sửa mục model = này
    """
    model = tf.keras.models.Sequential([
        # Phần mạng tích chập (trích xuất đặc trưng ảnh)
        # tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),

        # Phần mạng neuron bình thường
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=['accuracy'])

    return model


class myCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")

            # Stop training once the above condition is met
            self.model.stop_training = True
