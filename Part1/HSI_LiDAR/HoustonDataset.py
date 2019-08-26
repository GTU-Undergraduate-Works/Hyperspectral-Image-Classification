hsi_image_file_path = "drive/Undergraduate_Project/HoustonDataset/HSI_DATA.tif"
lidar_image_file_path = "drive/Undergraduate_Project/HoustonDataset/LiDAR_DATA.tif"
train_file_path = "drive/Undergraduate_Project/HoustonDataset/train.txt"
test_file_path = "drive/Undergraduate_Project/HoustonDataset/test.txt"

from osgeo import gdal
import numpy as np
from sklearn.decomposition import PCA

class Houston:

    def __init__(self):
        raster = gdal.Open(hsi_image_file_path)
        self.hsi_raw_data = np.array(raster.ReadAsArray())
        raster = gdal.Open(lidar_image_file_path)
        self.lidar_raw_data = np.array(raster.ReadAsArray())

        HEIGHT = self.hsi_raw_data.shape[1]
        WIDTH = self.hsi_raw_data.shape[2]
        self.train_pixels = self.get_pixels(train_file_path)
        self.test_pixels = self.get_pixels(test_file_path)


        hsi_train_data = []
        hsi_test_data = []
        lidar_train_data = []
        lidar_test_data = []
        train_labels = []
        test_labels = []

        for i in range(HEIGHT):
            for j in range(WIDTH):
                if self.train_pixels[i, j] != 0:
                    hsi_train_data.append(self.hsi_raw_data[:, i, j])
                    lidar_train_data.append(self.lidar_raw_data[i, j])
                    train_labels.append(self.train_pixels[i, j])
                if self.test_pixels[i, j] != 0:
                    hsi_test_data.append(self.hsi_raw_data[:, i, j])
                    lidar_test_data.append(self.lidar_raw_data[i, j])
                    test_labels.append(self.test_pixels[i, j])


        self.hsi_train_data = np.array(hsi_train_data)
        self.hsi_test_data = np.array(hsi_test_data)
        self.lidar_train_data = np.array(lidar_train_data)
        self.lidar_test_data = np.array(lidar_test_data)
        self.train_labels = np.array(train_labels)
        self.test_labels = np.array(test_labels)

        self.one_hot_train = self.convert_to_one_hot(self.train_labels)
        self.one_hot_test = self.convert_to_one_hot(self.test_labels)

    def get_hsi_data(self):
        return self.hsi_raw_data

    def get_lidar_data(self):
        return self.lidar_raw_data

    def get_hsi_train_data(self):
        return self.hsi_train_data

    def get_hsi_test_data(self):
        return self.hsi_test_data

    def get_lidar_train_data(self):
        return self.lidar_train_data

    def get_lidar_test_data(self):
        return self.lidar_test_data


    def get_train_labels(self):
        return self.train_labels

    def get_test_labels(self):
        return self.test_labels


    def get_train_as_one_hot(self):
        return self.one_hot_train

    def get_test_as_one_hot(self):
        return self.one_hot_test

    def get_pixels(self, filename):

        file = open(filename)
        triplets = file.read().split()
        for i in range(0, len(triplets)):
            triplets[i] = triplets[i].split(",")
        array = np.array(triplets, dtype=int)
        file.close()
        return array

    def get_train_pixels(self):
        return self.train_pixels

    def get_test_pixels(self):
        return self.test_pixels


    def convert_to_one_hot(self, vector, num_classes=None):
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        vector = vector-1

        if num_classes is None:
            num_classes = np.max(vector) + 1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)

        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)


    def HSI_PCA(self, n_components=2):
        NUM_BANDS = self.hsi_raw_data.shape[0]
        HEIGHT = self.hsi_raw_data.shape[1]
        WIDTH = self.hsi_raw_data.shape[2]
        hsi_data_2d = self.hsi_raw_data.transpose(1,2,0).reshape((HEIGHT*WIDTH), NUM_BANDS)
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(hsi_data_2d)
        principalComponents = np.array(principalComponents).transpose(1, 0).reshape(n_components, HEIGHT, WIDTH)
        return principalComponents


    def get_patches(self, patch_size, Train=True, PCA=False, LiDAR=False, n_components=2):

        if PCA:
            image_data = self.HSI_PCA(n_components=n_components)
        else:
            image_data = self.hsi_raw_data

        if LiDAR:
            lidar_data = self.get_lidar_data()
            image_data = np.concatenate([image_data, lidar_data[None, ...]], axis=0)

        HEIGHT = image_data.shape[1]
        WIDTH = image_data.shape[2]
        offset = int(patch_size / 2)
        train_patches = []
        if Train:
            data = self.train_pixels
        else:
            data = self.test_pixels
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if data[i, j] != 0:
                    row_low = max(0, i - offset)
                    row_high = min(HEIGHT - 1, i + offset)
                    if row_low == 0:
                        row_high = row_low + patch_size - 1
                    if row_high == HEIGHT - 1:
                        row_low = row_high - patch_size + 1

                    col_low = max(0, j - offset)
                    col_high = min(WIDTH - 1, j + offset)
                    if col_low == 0:
                        col_high = col_low + patch_size - 1
                    if col_high == WIDTH - 1:
                        col_low = col_high - patch_size + 1

                    train_patches.append(image_data[0:, row_low:row_high + 1, col_low:col_high + 1])
        return np.array(train_patches)








