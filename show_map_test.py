#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

class mapping:

    def __init__(self):
        mapping = []
        f = open('save_mapping.csv', 'r')
        rdr = csv.reader(f)
        for line in rdr:
            mapping.append([round(float(line[0]), 1), round(float(line[1]), 1)])
        self.raw_data = np.array(mapping)

        pose = []
        f = open('save_pos.csv', 'r')
        rdr = csv.reader(f)
        for line in rdr:
            pose.append([float(line[0]), float(line[1])])
        self.raw_pose = np.array(pose)

        self.overlap_data = self.del_overlap()

    def del_overlap(self):

        map = pd.DataFrame(self.raw_data).drop_duplicates().to_numpy()

        return np.array(map)

    def plot_data(self):

        plt.plot(self.overlap_data[:, 0], self.overlap_data[:, 1], ".g", markersize=3)
        plt.plot(self.raw_pose[:, 0], self.raw_pose[:, 1], "or", markersize=3)
        plt.show()

    def test1(self):

        map_filter = []
        print("전체길이 : {}".format(len(self.raw_pose)))
        for i in range(len(self.raw_pose)):
            tmp_data = []
            print(i)
            if self.raw_pose[i][0] == 0:
                pass
            else:
                for k in range(len(self.overlap_data)):

                    if self.overlap_data[k][0] == 0:
                        pass
                    else:
                        dist = np.hypot(self.raw_pose[i][0] - self.overlap_data[k][0],self.raw_pose[i][1] - self.overlap_data[k][1])
                        tmp_data.append([dist,self.overlap_data[k][0],self.overlap_data[k][1]])

                tmp_data.sort(key=lambda x:x[0])
                for i in range(len(tmp_data[0:50])):
                    map_filter.append(tmp_data[i])

        map_filter = np.array(map_filter)

        df_1 = pd.DataFrame(map_filter)

        df_1.to_csv('save_map_filter.csv',mode = 'w', index=False)

        plt.plot(map_filter[:, 1], map_filter[:, 2], ".g", markersize=3)
        plt.plot(self.raw_pose[:, 0], self.raw_pose[:, 1], "or", markersize=3)
        plt.show()


if __name__ == '__main__':
    run = mapping()
    run.plot_data()
    run.test1()