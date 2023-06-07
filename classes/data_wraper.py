from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, file_paths, save_maps=False):
        self.file_paths = file_paths
        self.save_maps = save_maps

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

        self.map1_list_sv = None
        self.map2_list_sv = None
        self.map3_list_sv = None
        self.combined_list_sv = None

        for file_path in self.file_paths:
            self.add_data(file_path)
                
        self.apply_coeff()

        if self.save_maps:
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

        # add the data
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

            self.combined_list[i] = file['Acombine_outForCeline'][0][i - old_lenght]
            self.azimut_list[i] = file['OUT_Fa_celine'][0][i - old_lenght]


        print('Added data from ' + file_path)
        print('New lenght: ' + str(new_lenght))
        
    
    def apply_coeff(self):
        self.map1_list = self.map1_list * self.coeff1_list[:, None, None]
        self.map2_list = self.map2_list * self.coeff2_list[:, None, None]
        self.map3_list = self.map3_list * self.coeff3_list[:, None, None]

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
            max_val = np.max([np.max(self.map1_list[i]), np.max(self.map2_list[i]), np.max(self.map3_list[i]), np.max(self.combined_list[i])])
            min_val = np.min([np.min(self.map1_list[i]), np.min(self.map2_list[i]), np.min(self.map3_list[i]), np.min(self.combined_list[i])])
            self.map1_list[i] = (self.map1_list[i] - min_val) / (max_val - min_val)
            self.map2_list[i] = (self.map2_list[i] - min_val) / (max_val - min_val)
            self.map3_list[i] = (self.map3_list[i] - min_val) / (max_val - min_val)
            self.combined_list[i] = (self.combined_list[i] - min_val) / (max_val - min_val)

        """ min_val = np.min([np.min(self.map1_list), np.min(self.map2_list), np.min(self.map3_list), np.min(self.combined_list)])
        max_val = np.max([np.max(self.map1_list), np.max(self.map2_list), np.max(self.map3_list), np.max(self.combined_list)])
        self.map1_list = (self.map1_list - min_val) / (max_val - min_val)
        self.map2_list = (self.map2_list - min_val) / (max_val - min_val)
        self.map3_list = (self.map3_list - min_val) / (max_val - min_val)
        self.combined_list = (self.combined_list - min_val) / (max_val - min_val) """

        if self.save_maps:
            self.map1_list_sv = self.map1_list.copy()
            self.map2_list_sv = self.map2_list.copy()
            self.map3_list_sv = self.map3_list.copy()
            self.combined_list_sv = self.combined_list.copy()

    
    def crop_data(self, crop_size):
        map1_list_cropped = np.zeros((len(self.map1_list), crop_size, crop_size))
        mask1_list_cropped = np.zeros((len(self.mask1_list), crop_size, crop_size))
        map2_list_cropped = np.zeros((len(self.map2_list), crop_size, crop_size))
        mask2_list_cropped = np.zeros((len(self.mask2_list), crop_size, crop_size))
        map3_list_cropped = np.zeros((len(self.map3_list), crop_size, crop_size))
        mask3_list_cropped = np.zeros((len(self.mask3_list), crop_size, crop_size))
        combined_list_cropped = np.zeros((len(self.combined_list), crop_size, crop_size))
        for i in range(len(self.map1_list)):
            center = self.get_center(i)
            box = np.array([center[0] - crop_size / 2, center[1] - crop_size / 2, center[0] + crop_size / 2, center[1] + crop_size / 2])
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

    def get_center(self, i):
        map = self.map1_list[i]
        max_value = np.max(map)
        max_index = np.where(map == max_value)
        return max_index[0][0], max_index[1][0]

    def crop_map(self, image, box, crop_size):
        image = image[box[0]:box[2], box[1]:box[3]]
        pad = np.ones((crop_size, crop_size)) * np.min(image)
        pad[0:image.shape[0], 0:image.shape[1]] = image
        return pad