from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

import random
from tqdm import tqdm
import math

def get_neighbours(map, x, y):
    neighbours = []
    if x > 0:
        neighbours.append((x - 1, y))
    if x < len(map) - 1:
        neighbours.append((x + 1, y))
    if y > 0:
        neighbours.append((x, y - 1))
    if y < len(map) - 1:
        neighbours.append((x, y + 1))
    neighbours = [n for n in neighbours if not np.isnan(map[n[0]][n[1]])]
    return neighbours

def load_FPST(path, total_tasks):
    coords = loadmat(os.path.join(path, "FPSTdata_coords.mat"))["coords"]

    filenames = os.listdir(path)
    filenames.remove("FPSTdata_coords.mat")
    # sort the filenames
    filenames = sorted(filenames, key=lambda x: int(x.split("_")[1].split(".")[0]))

    print("Overhead done, starting the loop", flush=True)

    fpsts = []

    progress_bar = tqdm(desc='Process', total=total_tasks, unit='task', dynamic_ncols=True)

    loaded = 0
    for filename in filenames:
        if loaded >= total_tasks:
            break
        file = loadmat(os.path.join(path, filename))
        fpsts += file["fpsts"].tolist()
        progress_bar.update(len(file["fpsts"].tolist()))
        loaded += len(file["fpsts"].tolist())
    
    fpsts = np.array(fpsts)

    progress_bar.close()

    return fpsts, coords

def load_data(path, len_map_big):
    filenames = os.listdir(path)
    filenames.remove("GridInfo.mat")
    # sort the filenames
    filenames = sorted(filenames, key=lambda x: int(x.split("_")[1].split(".")[0]))

    file = loadmat(os.path.join(path, "GridInfo.mat"))
    Coord_X = file["Colc"][:, 0]
    Coord_Y = file["Linc"][:, 0]
    coords = [(Coord_X[i], Coord_Y[i]) for i in range(len(Coord_X))]

    maps = np.empty((len(filenames), len_map_big, len_map_big))

    progress_bar = tqdm(desc='Load', total=len(filenames), unit='task', dynamic_ncols=True)

    for i, filename in enumerate(filenames):
        file = loadmat(os.path.join(path, filename))
        maps[i] = file["SPST"]
        progress_bar.update(1)

    progress_bar.close()

    return maps, coords

def save_data(path, fpsts, coords):
    dict = {
        "coords": coords
    }
    savemat(os.path.join(path, "FPSTdata_coords.mat"), dict)

    for i in range(0, len(fpsts), 1000):
        dict = {
            "fpsts": fpsts[i:min(i + 1000, len(fpsts))],
        }
        savemat(os.path.join(path, f"FPSTdata_{i}.mat"), dict)
    
    return

def get_map(i, j, forbidden, len_map, maps, coords):
    values = [maps[l][i][j] for l in range(len(maps)) if l not in forbidden]
                
    map = np.ones((len_map, len_map)) * np.nan

    stride = math.ceil(len(maps[0]) / len_map)
    for i in range(len(values)):
        x = math.floor(coords[i][0] / stride)
        y = math.floor(coords[i][1] / stride)
        map[x][y] = x
    return map

def get_maps(maps, coords, ind_forbiden, len_map, len_map_big):
    total_tasks = len_map_big * len_map_big

    fpsts = np.zeros((total_tasks, len_map, len_map))

    # Initialize the progress bars
    progress_bar = tqdm(desc='Process', total=total_tasks, unit='task', dynamic_ncols=True)

    ind = 0
    for i in range(len_map_big):
        for j in range(len_map_big):
            fpst = get_map(i, j, ind_forbiden, len_map, maps, coords)
            # transpose the maps
            fpst = fpst.T
            #reverse the y axis
            fpst = fpst[::-1]

            fpsts[ind] = fpst
            ind += 1
            progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    return fpsts

def densify_to(map, target_grid, orig_pos):
    len_map = len(map)

    if orig_pos is None:
        orig_pos = np.zeros((len_map, len_map, 2), dtype=int)
        for x in range(len_map):
            for y in range(len_map):
                orig_pos[x][y] = np.array([x, y])

    map2 = np.ones((target_grid, target_grid)) * np.nan
    for x in range(len_map):
        for y in range(len_map):
            x_ind = int(x * target_grid / len_map)
            y_ind = int(y * target_grid / len_map)
            map2[x_ind][y_ind] = map[x][y]
    
    for x in range(len(orig_pos)):
        for y in range(len(orig_pos)):
            x_ind = int(orig_pos[x][y][0] * target_grid / len_map)
            y_ind = int(orig_pos[x][y][1] * target_grid / len_map)
            orig_pos[x][y] = np.array([x_ind, y_ind])
    
    while np.isnan(map2).any():
        map2_copy = map2.copy()
        for x in range(len(map2_copy)):
            for y in range(len(map2_copy)):
                if np.isnan(map2_copy[x][y]):
                    neighbours = get_neighbours(map2_copy, x, y)
                    if len(neighbours) > 0:
                        map2[x][y] = np.mean([map2_copy[n[0]][n[1]] for n in neighbours])
    return map2, orig_pos

def rev_densify_to(map, target_grid):
    len_map = len(map)

    map2 = np.ones((target_grid, target_grid))
    for x in range(target_grid):
        for y in range(target_grid):
            x_ind = int(x * len_map / target_grid)
            y_ind = int(y * len_map / target_grid)
            map2[x][y] = map[x_ind][y_ind]
    return map2

def get_corners(len_map, size):
    # the corners are triangles of size size

    corners = []
    for i in range(size):
        for j in range(size):
            if i + j < size:
                corners.append([i, j])
            else:
                break
    
    # replicate the corners to the other 3 quadrants
    corners = corners + [[len_map - i - 1, j] for i, j in corners]
    corners = corners + [[i, len_map - j - 1] for i, j in corners]
    corners = corners + [[len_map - i - 1, len_map - j - 1] for i, j in corners]
    
    return corners

def nullify_corners(map):
    len_map = map.shape[0]
    corners = []
    if len_map == 26:
        corners = get_corners(len_map, 2)
    elif len_map == 32:
        corners = get_corners(len_map, 2)
    elif len_map == 64:
        corners = get_corners(len_map, 8)
    elif len_map == 128:
        pass

    for i, j in corners:
        map[i, j] = 0

    return map

def smooth_map(map):
    len_map = map.shape[0]
    corners = []
    if len_map == 26:
        corners = get_corners(len_map, 2)
    elif len_map == 32:
        corners = get_corners(len_map, 2)
    elif len_map == 64:
        corners = get_corners(len_map, 8)
    elif len_map == 128:
        pass

    map = np.nan_to_num(map)
    for i in range(len_map):
        for j in range(len_map):
            if map[i, j] == 0:
                if [i, j] not in corners:
                    map[i, j] = np.mean(map[max(0, i - 1):min(len_map, i + 2), max(0, j - 1):min(len_map, j + 2)])

    return map

def process(map):
    map = smooth_map(map)
    map[map == 0] = 10e-17
    map = np.log10(np.abs(map))
    return map

def rev_process(map):
    map = np.power(10, map)
    map[map <= 10e-17] = 0
    return map