# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Calculate Hamming Distance based on XOR logic control in bit level
"""
import numpy as np
import pandas as pd


def transfer_to_32bits(x):
    """
    Returns: Transfer sample vector x into binary system. Output is (32, feature_num) array if transfer into int32 bits
    Variables: x-np.array of each sample
    """
    x_bits = np.zeros([32, len(x)], dtype=np.int32)
    for i in range(len(x)):
        num_bi = bin(x[i])[2:].zfill(32)
        num_bi = np.array([int(num) for num in num_bi])
        x_bits[:, i] = num_bi
    return x_bits


def hamming_distance_32bits_pairwise(x1, x2):
    """
    Returns: Hamming distance based on XOR logic control in 32bit level
             数字越大, 说明两个sample越不相似, return 值大于0 < 32 * feature num / 32 = feature num
             这个bit level的hamming distance描述了在bit的视角上 number of different features
    Variables: x1-np.array recording features value of sample 1
               x2-np.array recording features value of sample 2
    """
    x1_bits = transfer_to_32bits(x1)
    x2_bits = transfer_to_32bits(x2)
    dif_bits = abs(x1_bits - x2_bits)
    return np.sum(dif_bits) / 32


def hamming_distance_32bits(x):
    """
    Returns: pdist based on customized Hamming Distance
    Variables: x-DF or np.array of features of different samples
    """
    if isinstance(x, pd.DataFrame):
        x = np.array(x)

    hamming_distance = list()
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            print(i)
            print(j)
            print("------------------------")
            hamming_distance.append(hamming_distance_32bits_pairwise(x[i, :], x[j, :]))
    return np.array(hamming_distance)
