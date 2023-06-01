from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, file_paths):
        self.nb_data = 0
        self.files = []
        for file_path in file_paths:
            file = loadmat(file_path)
            self.files.append(file)
            self.nb_data += len(file['Map1_outForCeline'][0])

        self.map1_list = np.zeros((self.nb_data, 512, 512))
        self.coeff1_list = np.zeros((self.nb_data))
        self.mask1_list = np.zeros((self.nb_data, 512, 512))

        self.map2_list = np.zeros((self.nb_data, 512, 512))
        self.coeff2_list = np.zeros((self.nb_data))
        self.mask2_list = np.zeros((self.nb_data, 512, 512))

        self.map3_list = np.zeros((self.nb_data, 512, 512))
        self.coeff3_list = np.zeros((self.nb_data))
        self.mask3_list = np.zeros((self.nb_data, 512, 512))

        self.combined_list = np.zeros((self.nb_data, 512, 512))
        self.azimut_list = np.zeros((self.nb_data))

        self.add_data()

        self.files = []
                
        self.apply_coeff()
    
    def add_data(self):
        i = 0
        for file in self.files:
            map1 = file['Map1_outForCeline'][0]
            coeff1 = file['coef1_outForCeline'][0]
            mask1 = file['mask1_outForceline'][0]
            map2 = file['Map2_outForCeline'][0]
            coeff2 = file['coef2_outForCeline'][0]
            mask2 = file['mask2_outForceline'][0]
            map3 = file['Map3_outForCeline'][0]
            coeff3 = file['coef3_outForCeline'][0]
            mask3 = file['mask3_outForceline'][0]
            combined = file['Acombine_outForCeline'][0]
            azimut = file['OUT_Fa_celine'][0]
            for j in range(len(file['Map1_outForCeline'][0])):
                self.map1_list[i] = map1[j]
                self.coeff1_list[i] = coeff1[j]
                self.mask1_list[i] = mask1[j]
                self.map2_list[i] = map2[j]
                self.coeff2_list[i] = coeff2[j]
                self.mask2_list[i] = mask2[j]
                self.map3_list[i] = map3[j]
                self.coeff3_list[i] = coeff3[j]
                self.mask3_list[i] = mask3[j]
                self.combined_list[i] = combined[j]
                self.azimut_list[i] = azimut[j]
                i += 1
    
    def apply_coeff(self):
        self.map1_list = self.map1_list * self.coeff1_list[:, None, None]
        self.map2_list = self.map2_list * self.coeff2_list[:, None, None]
        self.map3_list = self.map3_list * self.coeff3_list[:, None, None]

    def preprocess_data(self):
        def pretreatment(x):
            x = np.abs(x)
            x_nonzero = x[x != 0]
            min_val = np.min(x_nonzero)
            x[x == 0] = min_val
            x = np.log10(x)
            return x

        self.map1_list = pretreatment(self.map1_list)
        self.map2_list = pretreatment(self.map2_list)
        self.map3_list = pretreatment(self.map3_list)
        self.combined_list = pretreatment(self.combined_list)

        min_val = np.min([np.min(self.map1_list), np.min(self.map2_list), np.min(self.map3_list), np.min(self.combined_list)])
        max_val = np.max([np.max(self.map1_list), np.max(self.map2_list), np.max(self.map3_list), np.max(self.combined_list)])
        self.map1_list = (self.map1_list - min_val) / (max_val - min_val)
        self.map2_list = (self.map2_list - min_val) / (max_val - min_val)
        self.map3_list = (self.map3_list - min_val) / (max_val - min_val)
        self.combined_list = (self.combined_list - min_val) / (max_val - min_val)

    
    def crop_data(self, crop_size):
        map1_list_cropped = np.zeros((len(self.map1_list), crop_size, crop_size))
        map2_list_cropped = np.zeros((len(self.map2_list), crop_size, crop_size))
        map3_list_cropped = np.zeros((len(self.map3_list), crop_size, crop_size))
        combined_list_cropped = np.zeros((len(self.combined_list), crop_size, crop_size))
        for i in range(len(self.map1_list)):
            center = self.get_center(i)
            box = np.array([center[0] - crop_size / 2, center[1] - crop_size / 2, center[0] + crop_size / 2, center[1] + crop_size / 2])
            box = box.astype(int)
            box = np.clip(box, 0, 512)
            map1_list_cropped[i] = self.crop_map(self.map1_list[i], box, crop_size)
            map2_list_cropped[i] = self.crop_map(self.map2_list[i], box, crop_size)
            map3_list_cropped[i] = self.crop_map(self.map3_list[i], box, crop_size)
            combined_list_cropped[i] = self.crop_map(self.combined_list[i], box, crop_size)
            
        self.map1_list = map1_list_cropped
        self.map2_list = map2_list_cropped
        self.map3_list = map3_list_cropped
        self.combined_list = combined_list_cropped

    def get_center(self, i):
        mask = self.mask3_list[i]
        center = np.array([0, 0])
        center[0] = np.sum(mask, axis=1).argmax()
        center[1] = np.sum(mask, axis=0).argmax()
        return center

    def crop_map(self, image, box, crop_size):
        image = image[box[0]:box[2], box[1]:box[3]]
        pad = np.ones((crop_size, crop_size)) * np.mean(image)
        pad[0:image.shape[0], 0:image.shape[1]] = image
        return pad
        
    def export_dataframe(self):
        data = {'map1': self.map1_list.tolist(), 'map2': self.map2_list.tolist(), 'map3': self.map3_list.tolist(), 'combined': self.combined_list.tolist()}
        df = pd.DataFrame(data)
        #df.to_csv('data.csv', index=False)
        return df
    
    def plot_map(self, data_index, map_index):
        if map_index == 1:
            map = self.map1_list[data_index]
        elif map_index == 2:
            map = self.map2_list[data_index]
        elif map_index == 3:
            map = self.map3_list[data_index]
        else:
            print('Map index out of range')
            return
        plt.imshow(map, cmap='jet')
        plt.show()
    
    def plot_combined(self, data_index):
        plt.imshow(self.combined_list[data_index], cmap='jet')
        plt.show()
    
    def plot_radial_profile(self, data_index, map_indexes):
        maps = []
        if 1 in map_indexes:
            maps.append(self.map1_list[data_index])
        if 2 in map_indexes:
            maps.append(self.map2_list[data_index])
        if 3 in map_indexes:
            maps.append(self.map3_list[data_index])
        else:
            print('No map index selected')
            return
        for map in maps:
            image = map
            azimut = self.azimut_list[data_index]
            center = int(image.shape[0] / 2)
            image = np.rot90(image, k=azimut)[center, :]
            plt.plot(image)
        plt.show()