from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


class FilePaths:
    def __init__(self):
        self.root_dir = None
        self.data_dir = None
        if "google.colab" in sys.modules:
            self.root_dir = os.path.join(
                os.getcwd(), "drive", "MyDrive", "Colab Notebooks"
            )
            self.data_dir = os.path.join(self.root_dir, "data")
        else:
            self.root_dir = os.getcwd()
            self.data_dir = os.path.join(self.root_dir, "data")

    def get_map_path(self, channel):
        # channel = (FOVfiting|3quadrants)_(2|3|4|5|6|9)
        channel_type = channel.split("_")[0]
        channel_number = channel.split("_")[1]
        return os.path.join(
            self.data_dir, "mapsCeline_" + channel_type + "_" + channel_number + "_pol-60.mat"
        )

    def get_coeff_path(self, channel):
        # channel = (FOVfiting|3quadrants)_(2|3|4|5|6|9)
        channel_type = channel.split("_")[0]
        channel_number = channel.split("_")[1]
        channel_dir = os.path.join(self.data_dir, "3MI-" + channel_number)
        if channel_type == "FOVfitting":
            return [os.path.join(channel_dir, "Test_coef1_FOVfitting.mat")]
        elif channel_type == "3quadrants":
            return [
                os.path.join(channel_dir, "Test_coef1_3quadrants_1.mat"),
                os.path.join(channel_dir, "Test_coef1_3quadrants_2.mat"),
                os.path.join(channel_dir, "Test_coef1_3quadrants_3.mat"),
            ]
        else:
            raise ValueError("Channel type must be FOVfitting or 3quadrants")


class Dataset:
    def __init__(self, channels):
        self.channels = channels
        self.file_paths = FilePaths()

        self.df = pd.DataFrame(
            {
                "azimut": [],
                "elevation": [],
                "map1": [],
                "coeff1": [],
                "mask1": [],
                "map2": [],
                "coeff2": [],
                "mask2": [],
                "map3": [],
                "coeff3": [],
                "mask3": [],
                "min_val": [],
                "max_val": [],
                "combined": [],
                "saved_map1": [],
                "saved_map2": [],
                "saved_map3": [],
                "saved_combined": [],
            }
        )

        self.crop_size = None

        for channel in channels:
            self.add_data(channel)

    def add_data(self, channel):

        # ==================== COEFFS ====================
        
        path = self.file_paths.get_coeff_path(channel)
        df_coeff = pd.DataFrame(
            {
                "azimut": [],
                "elevation": [],
                "coeff1": [],
            }
        )
        for path in path:
            file_coeff = loadmat(path)

            azimut_list = file_coeff["OUT_Fa_celine_LMB"][0]
            elevation_list = file_coeff["OUT_Fb_celine_LMB"][0]
            coeff1_list = file_coeff["Coef_theo"][0]

            df_coeff_temp = pd.DataFrame(
                {
                    "azimut": azimut_list,
                    "elevation": elevation_list,
                    "coeff1": coeff1_list,
                }
            )

            df_coeff = pd.concat([df_coeff, df_coeff_temp], ignore_index=True)

        # ==================== MAPS ====================

        file_map = loadmat(self.file_paths.get_map_path(channel))

        azimut_list = file_map["OUT_Fa_celine"][0]
        elevation_list = file_map["OUT_Fb_celine"][0]

        map1_list = file_map["Map1_outForCeline"][0]
        #coeff1_list = file_map["coef1_outForCeline"][0]
        mask1_list = file_map["mask1_outForceline"][0]

        map2_list = file_map["Map2_outForCeline"][0]
        coeff2_list = file_map["coef2_outForCeline"][0]
        mask2_list = file_map["mask2_outForceline"][0]

        map3_list = file_map["Map3_outForCeline"][0]
        coeff3_list = file_map["coef3_outForCeline"][0]
        mask3_list = file_map["mask3_outForceline"][0]

        min_val_list = np.zeros(len(map1_list))
        max_val_list = np.zeros(len(map1_list))

        saved_map1_list = map1_list.copy()
        saved_map2_list = map2_list.copy()
        saved_map3_list = map3_list.copy()

        df_map = pd.DataFrame(
            {
                "azimut": azimut_list,
                "elevation": elevation_list,
                "map1": map1_list,
                "mask1": mask1_list,
                "map2": map2_list,
                "coeff2": coeff2_list,
                "mask2": mask2_list,
                "map3": map3_list,
                "coeff3": coeff3_list,
                "mask3": mask3_list,
                "min_val": min_val_list,
                "max_val": max_val_list,
                "saved_map1": saved_map1_list,
                "saved_map2": saved_map2_list,
                "saved_map3": saved_map3_list,
            }
        )
        df_temp = pd.merge(df_map, df_coeff, on=["azimut", "elevation"])

        # drop the 2 mysterious outliers
        channel_type = channel.split("_")[0]
        if channel_type == "3quadrants":
            df_temp = df_temp.drop([97, 408])
            df_temp = df_temp.reset_index(drop=True)
        
        # shearch for outliers
        outliers = []
        for i in range(len(df_temp)):
            if np.max(df_temp["mask1"][i]) == 1.0:
                outliers.append(i)
        df_temp = df_temp.drop(outliers)
        df_temp = df_temp.reset_index(drop=True)

        self.apply_coeff(df_temp)

        combined_list = []
        for i in range(len(df_temp)):
            map3 = df_temp["map3"][i] * (1 - df_temp["mask3"][i])
            map2 = df_temp["map2"][i] * (1 - df_temp["mask2"][i]) * df_temp["mask3"][i]
            map1 = (
                df_temp["map1"][i]
                * (1 - df_temp["mask1"][i])
                * df_temp["mask2"][i]
                * df_temp["mask3"][i]
            )
            combined_list.append(map1 + map2 + map3)
        
        saved_combined_list = combined_list.copy()

        df_temp["combined"] = combined_list
        df_temp["saved_combined"] = saved_combined_list

        # add df_temp to df
        self.df = pd.concat([self.df, df_temp], ignore_index=True)

        print("Added data from " + channel)
        print("New lenght: " + str(len(self.df)))

    def apply_coeff(self, df):
        for i in range(len(df)):
            df["map1"][i] = df["map1"][i] * df["coeff1"][i]
            df["map2"][i] = df["map2"][i] * df["coeff2"][i]
            df["map3"][i] = df["map3"][i] * df["coeff3"][i]

    def preprocess_data(self):
        def pretreatment(x):
            x = np.abs(x)
            x[x == 0] = 1e-15
            x = np.log10(x)
            return x

        self.df["map1"] = self.df["map1"].apply(pretreatment)
        self.df["map2"] = self.df["map2"].apply(pretreatment)
        self.df["map3"] = self.df["map3"].apply(pretreatment)
        self.df["combined"] = self.df["combined"].apply(pretreatment)

        # normalize data between 0 and 1 using min and max values from the dataframe
        for i in range(len(self.df)):
            self.df["min_val"][i] = np.min(
                [
                    np.min(self.df["map1"][i]),
                    np.min(self.df["map2"][i]),
                    np.min(self.df["map3"][i]),
                ]
            )
            self.df["max_val"][i] = np.max(
                [
                    np.max(self.df["map1"][i]),
                    np.max(self.df["map2"][i]),
                    np.max(self.df["map3"][i]),
                ]
            )
            self.df["map1"][i] = (self.df["map1"][i] - self.df["min_val"][i]) / (
                self.df["max_val"][i] - self.df["min_val"][i]
            )
            self.df["map2"][i] = (self.df["map2"][i] - self.df["min_val"][i]) / (
                self.df["max_val"][i] - self.df["min_val"][i]
            )
            self.df["map3"][i] = (self.df["map3"][i] - self.df["min_val"][i]) / (
                self.df["max_val"][i] - self.df["min_val"][i]
            )
            self.df["combined"][i] = (
                self.df["combined"][i] - self.df["min_val"][i]
            ) / (self.df["max_val"][i] - self.df["min_val"][i])

    def rev_preprocess(self, map, i):
        min_val, max_val = self.df["min_val"][i], self.df["max_val"][i]
        map = map * (max_val - min_val) + min_val
        map = np.power(10, map)
        return map

    def embed_map(self, pred, i):
        pred = self.rev_preprocess(pred, i)
        x, y = self.get_center(self.df["saved_map1"][i])
        box = np.array(
            [
                x - self.crop_size / 2,
                y - self.crop_size / 2,
                x + self.crop_size / 2,
                y + self.crop_size / 2,
            ]
        )
        box = box.astype(int)
        box = np.clip(box, 0, 512)
        map3_cropped = self.crop_map(self.df["saved_map3"][i], box)
        mask3_cropped = self.crop_map(self.df["mask3"][i], box)
        pred_fused = pred * mask3_cropped + map3_cropped * (1 - mask3_cropped)
        map = self.df["saved_map3"][i].copy()
        map[box[0] : box[2], box[1] : box[3]] = pred_fused[
            0 : (box[2] - box[0]), 0 : (box[3] - box[1])
        ]
        return map

    def compute_stray_light(self, map):
        i, j = self.get_center(map)
        nominal = np.sum(map[i : i + 2, j : j + 2])
        map = map / nominal
        #map[i : i + 2, j : j + 2] = 0
        # set a 20 radius disc around (i,j) to 0
        for x in range(i - 20, i + 20):
            for y in range(j - 20, j + 20):
                if x < 0 or y < 0 or x >= 512 or y >= 512:
                    continue
                if np.sqrt((x - i) ** 2 + (y - j) ** 2) < 20:
                    map[x, y] = 0
        return i, j, np.sum(map)

    def crop_data(self, crop_size):
        self.crop_size = crop_size
        for i in range(len(self.df)):
            x, y = self.get_center(self.df["map1"][i])
            box = np.array(
                [
                    x - crop_size / 2,
                    y - crop_size / 2,
                    x + crop_size / 2,
                    y + crop_size / 2,
                ]
            )
            box = box.astype(int)
            box = np.clip(box, 0, 512)
            self.df["map1"][i] = self.crop_map(self.df["map1"][i], box)
            self.df["map2"][i] = self.crop_map(self.df["map2"][i], box)
            self.df["map3"][i] = self.crop_map(self.df["map3"][i], box)
            self.df["combined"][i] = self.crop_map(self.df["combined"][i], box)

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
                sum = np.sum(map[i : i + 2, j : j + 2])
                if sum > max_sum:
                    max_sum = sum
                    nom_i = i
                    nom_j = j
        return (nom_i, nom_j)

    def crop_map(self, map, box):
        map = map[box[0] : box[2], box[1] : box[3]]
        pad = np.ones((self.crop_size, self.crop_size)) * np.min(map)
        pad[0 : map.shape[0], 0 : map.shape[1]] = map
        return pad
