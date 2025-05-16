import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображений
images = np.array([np.load(f"images/car_{i}.npy") for i in range(9)])
print("Размерность массива изображений:", images.shape)

# Сумма всех пикселей во всех изображениях
total_sum = np.sum(images)
print("Сумма всех пикселей:", total_sum)

# Сумма пикселей по каждому изображению
sum_per_image = np.sum(images, axis=(1, 2))
print("Сумма по изображениям:", sum_per_image)

# Индекс изображения с максимальной суммой
max_idx = np.argmax(sum_per_image)
print("Индекс изображения с max суммой:", max_idx)

# Среднее изображение по всем
mean_image = np.mean(images, axis=0)

# Отображение среднего изображения
mean_image_clipped = np.clip(mean_image, 0, 255)
plt.imshow(mean_image_clipped.astype(np.uint8))
plt.axis('off')
plt.title("Среднее изображение")
plt.show()

# Стандартное отклонение по всем пикселям
std_dev = np.std(images)
print("Стандартное отклонение:", std_dev)

# Нормализация изображений
normalized_images = (images - mean_image) / std_dev
print("Размерность нормализованных изображений:", normalized_images.shape)

# обрезка фото
h, w = images.shape[1], images.shape[2]
if h >= 300 and w >= 400:
    cropped_images = images[:, 200:300, 280:400]
    print("Размерность обрезанных изображений:", cropped_images.shape)
else:
    print(f"Изображения слишком малы для обрезки: ({h}, {w})")
