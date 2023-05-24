from scipy.io import loadmat
import numpy as np
import pandas as pd

class Map:
    def __init__(self, image, coeff, mask):
        self.image = image
        self.coeff = coeff
        self.mask = mask

class Data:
    def __init__(self, map1, map2, map3, combined, azimut):
        self.map1 = map1
        self.map2 = map2
        self.map3 = map3
        self.combined = combined
        self.azimut = azimut

class Dataset:
    def __init__(self, file_path = None):
        self.data_list = []
        self.data_num = 0

        if file_path != None:
            self.add_data(file_path)
    
    def add_data(self, file_path):
        file = loadmat(file_path)
        map_list_1 = file['Map1_outForCeline'][0]
        coeff_list_1 = file['coef1_outForCeline'][0]
        mask_list_1 = file['mask1_outForceline'][0]
        map_list_2 = file['Map2_outForCeline'][0]
        coeff_list_2 = file['coef2_outForCeline'][0]
        mask_list_2 = file['mask2_outForceline'][0]
        map_list_3 = file['Map3_outForCeline'][0]
        coeff_list_3 = file['coef3_outForCeline'][0]
        mask_list_3 = file['mask3_outForceline'][0]
        combined_list = file['Acombine_outForCeline'][0]
        azimut_list = file['OUT_Fa_celine'][0]
        for i in range(len(map_list_1)):
            map1 = Map(map_list_1[i], coeff_list_1[i], mask_list_1[i])
            map2 = Map(map_list_2[i], coeff_list_2[i], mask_list_2[i])
            map3 = Map(map_list_3[i], coeff_list_3[i], mask_list_3[i])
            combined = combined_list[i]
            azimut = azimut_list[i]
            data = Data(map1, map2, map3, combined, azimut)
            # add data to dataframe without using append
            self.data_list.append(data)
        self.data_num += len(map_list_1)
    
    def export_dataframe(self):
        data_list = []
        for data in self.data_list:
            data_list.append([data.map1.image, data.map1.coeff, data.map1.mask, data.map2.image, data.map2.coeff, data.map2.mask, data.map3.image, data.map3.coeff, data.map3.mask, data.combined, data.azimut])
        df = pd.DataFrame(data_list, columns=['Map1', 'Coeff1', 'Mask1', 'Map2', 'Coeff2', 'Mask2', 'Map3', 'Coeff3', 'Mask3', 'Combined', 'Azimut'])
        return df
    
    def plot_map(self, data_index, map_index):
        import matplotlib.pyplot as plt
        data = self.data_list[data_index]
        if map_index == 1:
            map = data.map1
        elif map_index == 2:
            map = data.map2
        elif map_index == 3:
            map = data.map3
        else:
            print('Map index out of range')
            return
        plt.imshow(np.log10(np.abs(map.image)), cmap='jet')
        plt.show()
    
    def plot_combined(self, data_index):
        import matplotlib.pyplot as plt
        data = self.data_list[data_index]
        plt.imshow(np.log10(np.abs(data.combined)))
        plt.show()
    
    def plot_radial_profile(self, data_index, map_indexes):
        import matplotlib.pyplot as plt
        data = self.data_list[data_index]
        maps = []
        if 1 in map_indexes:
            maps.append(data.map1)
        if 2 in map_indexes:
            maps.append(data.map2)
        if 3 in map_indexes:
            maps.append(data.map3)
        else:
            print('No map index selected')
            return
        for map in maps:
            image = np.abs(map.image)
            azimut = data.azimut
            image = np.rot90(image, k=azimut)[256]
            plt.plot(np.log10(np.abs(image)))
        plt.show()