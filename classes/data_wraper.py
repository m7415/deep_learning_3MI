from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, file_paths):
        self.file_paths = file_paths

        self.map1_list = np.zeros((0, 512, 512))
        self.coeff1_list = np.zeros((0))
        self.mask1_list = np.zeros((0, 512, 512))

        self.map2_list = np.zeros((0, 512, 512))
        self.coeff2_list = np.zeros((0))
        self.mask2_list = np.zeros((0, 512, 512))

        self.map3_list = np.zeros((0, 512, 512))
        self.coeff3_list = np.zeros((0))
        self.mask3_list = np.zeros((0, 512, 512))

        self.combined_list = np.zeros((0, 512, 512))
        self.azimut_list = np.zeros((0))
        self.elevation_list = np.zeros((0))

        self.map1_list_sv = None
        self.map2_list_sv = None
        self.map3_list_sv = None
        self.combined_list_sv = None

        self.crop_size = None

        for file_path in self.file_paths:
            self.add_data(file_path)

        self.min_max_values = np.zeros((len(self.map1_list),2))

        self.map1_list_sv = self.map1_list.copy()
        self.map2_list_sv = self.map2_list.copy()
        self.map3_list_sv = self.map3_list.copy()
        self.combined_list_sv = self.combined_list.copy()
    
    def add_data(self, file_path):
        file = loadmat(file_path)
        old_lenght = len(self.map1_list)
        new_lenght = old_lenght + len(file['Map1_outForCeline'][0])

        # extend the length of the lists
        self.map1_list = np.resize(self.map1_list, (new_lenght, 512, 512))
        self.coeff1_list = np.resize(self.coeff1_list, (new_lenght))
        self.mask1_list = np.resize(self.mask1_list, (new_lenght, 512, 512))

        self.map2_list = np.resize(self.map2_list, (new_lenght, 512, 512))
        self.coeff2_list = np.resize(self.coeff2_list, (new_lenght))
        self.mask2_list = np.resize(self.mask2_list, (new_lenght, 512, 512))

        self.map3_list = np.resize(self.map3_list, (new_lenght, 512, 512))
        self.coeff3_list = np.resize(self.coeff3_list, (new_lenght))
        self.mask3_list = np.resize(self.mask3_list, (new_lenght, 512, 512))

        self.combined_list = np.resize(self.combined_list, (new_lenght, 512, 512))
        self.azimut_list = np.resize(self.azimut_list, (new_lenght))
        self.elevation_list = np.resize(self.elevation_list, (new_lenght))

        # add the data
        outlier_index = []
        for i in range(old_lenght, new_lenght):
            self.map1_list[i] = file['Map1_outForCeline'][0][i - old_lenght]
            self.coeff1_list[i] = file['coef1_outForCeline'][0][i - old_lenght]
            self.mask1_list[i] = file['mask1_outForceline'][0][i - old_lenght]

            self.map2_list[i] = file['Map2_outForCeline'][0][i - old_lenght]
            self.coeff2_list[i] = file['coef2_outForCeline'][0][i - old_lenght]
            self.mask2_list[i] = file['mask2_outForceline'][0][i - old_lenght]

            self.map3_list[i] = file['Map3_outForCeline'][0][i - old_lenght]
            self.coeff3_list[i] = file['coef3_outForCeline'][0][i - old_lenght]
            self.mask3_list[i] = file['mask3_outForceline'][0][i - old_lenght]

            self.apply_coeff(i)

            self.combined_list[i] = self.compute_combined(i)
            #file['Acombine_outForCeline'][0][i - old_lenght]
            self.azimut_list[i] = file['OUT_Fa_celine'][0][i - old_lenght]
            self.elevation_list[i] = file['OUT_Fb_celine'][0][i - old_lenght]

            """ _, _, sl = self.compute_stray_light(self.combined_list[i])
            if sl > 1:
                outlier_index.append(i) """
            if np.max(self.mask1_list[i]) == 1.0:
                outlier_index.append(i)
        
        self.map1_list = np.delete(self.map1_list, outlier_index, axis=0)
        self.coeff1_list = np.delete(self.coeff1_list, outlier_index, axis=0)
        self.mask1_list = np.delete(self.mask1_list, outlier_index, axis=0)

        self.map2_list = np.delete(self.map2_list, outlier_index, axis=0)
        self.coeff2_list = np.delete(self.coeff2_list, outlier_index, axis=0)
        self.mask2_list = np.delete(self.mask2_list, outlier_index, axis=0)

        self.map3_list = np.delete(self.map3_list, outlier_index, axis=0)
        self.coeff3_list = np.delete(self.coeff3_list, outlier_index, axis=0)
        self.mask3_list = np.delete(self.mask3_list, outlier_index, axis=0)

        self.combined_list = np.delete(self.combined_list, outlier_index, axis=0)
        self.azimut_list = np.delete(self.azimut_list, outlier_index, axis=0)
        self.elevation_list = np.delete(self.elevation_list, outlier_index, axis=0)

        new_lenght = len(self.map1_list)

        print('Added data from ' + file_path)
        print('New lenght: ' + str(new_lenght))
        
    
    def apply_coeff(self, i):
        self.map1_list[i] = self.map1_list[i] * self.coeff1_list[i]
        self.map2_list[i] = self.map2_list[i] * self.coeff2_list[i]
        self.map3_list[i] = self.map3_list[i] * self.coeff3_list[i]
    
    def compute_combined(self, i):
        map3 = self.map3_list[i] * (1 - self.mask3_list[i])
        map2 = self.map2_list[i] * (1 - self.mask2_list[i]) * self.mask3_list[i]
        map1 = self.map1_list[i] * (1 - self.mask1_list[i]) * self.mask2_list[i] * self.mask3_list[i]
        return map1 + map2 + map3

    def preprocess_data(self):
        def pretreatment(x):
            x = np.abs(x)
            x[x == 0] = np.min(x[x != 0])
            x = np.log10(x)
            return x

        for i in range(len(self.map1_list)):
            self.map1_list[i] = pretreatment(self.map1_list[i])
            self.map2_list[i] = pretreatment(self.map2_list[i])
            self.map3_list[i] = pretreatment(self.map3_list[i])
            self.combined_list[i] = pretreatment(self.combined_list[i])
            min_val = np.min([np.min(self.map1_list[i]), np.min(self.map2_list[i]), np.min(self.map3_list[i]), np.min(self.combined_list[i])])
            max_val = np.max([np.max(self.map1_list[i]), np.max(self.map2_list[i]), np.max(self.map3_list[i]), np.max(self.combined_list[i])])
            self.min_max_values[i] = (min_val, max_val)
            self.map1_list[i] = (self.map1_list[i] - min_val) / (max_val - min_val)
            self.map2_list[i] = (self.map2_list[i] - min_val) / (max_val - min_val)
            self.map3_list[i] = (self.map3_list[i] - min_val) / (max_val - min_val)
            self.combined_list[i] = (self.combined_list[i] - min_val) / (max_val - min_val)

    def rev_preprocess(self, map, i):
        min_val, max_val = self.min_max_values[i]
        map = map * (max_val - min_val) + min_val
        map = np.power(10, map)
        return map
    
    def embed_map(self, pred, i):
        pred = self.rev_preprocess(pred, i)
        x, y = self.get_center(self.map1_list_sv[i])
        box = np.array([x - self.crop_size / 2, y - self.crop_size / 2, x + self.crop_size / 2, y + self.crop_size / 2])
        box = box.astype(int)
        box = np.clip(box, 0, 512)
        map3_cropped = self.crop_map(self.map3_list_sv[i], box, self.crop_size)
        pred_fused = pred * self.mask3_list[i] + map3_cropped * (1 - self.mask3_list[i])
        map = self.map3_list_sv[i]
        map[box[0]:box[2], box[1]:box[3]] = pred_fused[0:(box[2] - box[0]), 0:(box[3] - box[1])]
        return map
    
    def compute_stray_light(self, map):
        i, j = self.get_center(map)
        nominal = np.sum(map[i:i+2, j:j+2])
        map = map / nominal
        map[i:i+2, j:j+2] = 0
        return i, j, np.sum(map)
    
    def crop_data(self, crop_size):
        self.crop_size = crop_size

        map1_list_cropped = np.zeros((len(self.map1_list), crop_size, crop_size))
        mask1_list_cropped = np.zeros((len(self.mask1_list), crop_size, crop_size))
        map2_list_cropped = np.zeros((len(self.map2_list), crop_size, crop_size))
        mask2_list_cropped = np.zeros((len(self.mask2_list), crop_size, crop_size))
        map3_list_cropped = np.zeros((len(self.map3_list), crop_size, crop_size))
        mask3_list_cropped = np.zeros((len(self.mask3_list), crop_size, crop_size))
        combined_list_cropped = np.zeros((len(self.combined_list), crop_size, crop_size))
        for i in range(len(self.map1_list)):
            x, y = self.get_center(self.map1_list[i])
            box = np.array([x - crop_size / 2, y - crop_size / 2, x + crop_size / 2, y + crop_size / 2])
            box = box.astype(int)
            box = np.clip(box, 0, 512)
            map1_list_cropped[i] = self.crop_map(self.map1_list[i], box, crop_size)
            mask1_list_cropped[i] = self.crop_map(self.mask1_list[i], box, crop_size)
            map2_list_cropped[i] = self.crop_map(self.map2_list[i], box, crop_size)
            mask2_list_cropped[i] = self.crop_map(self.mask2_list[i], box, crop_size)
            map3_list_cropped[i] = self.crop_map(self.map3_list[i], box, crop_size)
            mask3_list_cropped[i] = self.crop_map(self.mask3_list[i], box, crop_size)
            combined_list_cropped[i] = self.crop_map(self.combined_list[i], box, crop_size)
            
        self.map1_list = map1_list_cropped
        self.mask1_list = mask1_list_cropped
        self.map2_list = map2_list_cropped
        self.mask2_list = mask2_list_cropped
        self.map3_list = map3_list_cropped
        self.mask3_list = mask3_list_cropped
        self.combined_list = combined_list_cropped

    def get_center(self, map):
        max_sum = 0
        nom_i = 0
        nom_j = 0
        max = np.max(map)
        coord = np.where(map == max)
        i_m = coord[0][0]
        j_m = coord[1][0]
        for i in range(i_m - 2, i_m + 2):
            for j in range(j_m - 2, j_m + 2):
                if i < 0 or j < 0 or i >= 512 or j >= 512:
                    continue
                sum = np.sum(map[i:i+2, j:j+2])
                if sum > max_sum:
                    max_sum = sum
                    nom_i = i
                    nom_j = j
        return (nom_i, nom_j)

    def crop_map(self, image, box, crop_size):
        image = image[box[0]:box[2], box[1]:box[3]]
        pad = np.ones((crop_size, crop_size)) * np.min(image)
        pad[0:image.shape[0], 0:image.shape[1]] = image
        return pad