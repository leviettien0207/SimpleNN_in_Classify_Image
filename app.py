import config
import directory_helper
import constant
import os
import matplotlib.pyplot as plt
from data_handle import split_data, train_val_generators
from data_kaggle_api_download import download_dataset
from tensorflow.keras.preprocessing.image import load_img

"""
Tải dữ liệu (nếu sử dụng API kaggle để tải)
(đối với trường hợp muốn sử dụng dữ liệu mới, git có dữ liệu mẫu sử dụng rồi)
"""
# print('downloading dataset')
# download_dataset()

"""
Phân chia tập dữ liệu training, validation (nếu chưa được chưa)
(đối với trường hợp muốn sử dụng dữ liệu mới, git có dữ liệu mẫu sử dụng rồi)
"""
# # Tạo folder train và validation, và các folder đánh số từ 0 -> 9 bên trong
# try:
#     print('start create folders and subfolders')
#     directory_helper.create_train_val_dirs(root_path=config.ROOT_DIR)
#     print('end create folders and subfolders')
# except FileExistsError:
#     print("You should not be seeing this since the upper directory is removed beforehand")
#
# # Copy ảnh từ thư mục dataset gốc vào các đường dẫn
# print('start copy images')
# split_data(constant.ZERO_SOURCE_DIR, constant.TRAINING_ZERO_SOURCE_DIR, constant.VALIDATION_ZERO_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.ONE_SOURCE_DIR, constant.TRAINING_ONE_SOURCE_DIR, constant.VALIDATION_ONE_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.TWO_SOURCE_DIR, constant.TRAINING_TWO_SOURCE_DIR, constant.VALIDATION_TWO_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.THREE_SOURCE_DIR, constant.TRAINING_THREE_SOURCE_DIR, constant.VALIDATION_THREE_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.FOUR_SOURCE_DIR, constant.TRAINING_FOUR_SOURCE_DIR, constant.VALIDATION_FOUR_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.FIVE_SOURCE_DIR, constant.TRAINING_FIVE_SOURCE_DIR, constant.VALIDATION_FIVE_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.SIX_SOURCE_DIR, constant.TRAINING_SIX_SOURCE_DIR, constant.VALIDATION_SIX_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.SEVEN_SOURCE_DIR, constant.TRAINING_SEVEN_SOURCE_DIR, constant.VALIDATION_SEVEN_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.EIGHT_SOURCE_DIR, constant.TRAINING_EIGHT_SOURCE_DIR, constant.VALIDATION_EIGHT_SOURCE_DIR, config.SPLIT_SIZE)
# split_data(constant.NINE_SOURCE_DIR, constant.TRAINING_NINE_SOURCE_DIR, constant.VALIDATION_NINE_SOURCE_DIR, config.SPLIT_SIZE)
# print('end copy images')

# # Test thử cái ảnh xem sao
# print("Sample image:")
# plt.imshow(load_img(f"{os.path.join(constant.ONE_SOURCE_DIR, os.listdir(constant.ONE_SOURCE_DIR)[5])}"))
# plt.show()
# # Ảnh sẽ bị đen xì

"""
Đẩy tập dữ liệu training, validation lên memory(xem readme để biết cấu trúc dataset cần như thế nào)
"""

train_generator, validation_generator = train_val_generators(config.DIRECTORY.TRAIN_DIR,
                                                             config.DIRECTORY.VALIDATION_DIR)

"""
Get the untrained model
"""
# Get the untrained model
model = config.convolutional_model()
callbacks = config.myCallback()
# Train the model
# Note that this may take some time.
history = model.fit(train_generator,
                    epochs=config.TrainConfig.epochs,
                    verbose=config.TrainConfig.verbose,
                    validation_data=validation_generator,
                    callbacks=[callbacks])

"""
Plot đồ thị accuracy và loss
"""
# Plot the chart for accuracy and loss on both training and validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
