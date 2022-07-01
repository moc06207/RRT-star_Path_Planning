import csv
from scipy import io
from mat4py import loadmat
import pymap3d as pm
import math
from scipy import linalg
import numpy as np
from JU_matrix import Kalman_Filter, m_n_inverse, n_n_inverse,DCM2eul_bn,eul2DCM_bn,product_matrix_dx,product_matrix,skew,plus_matrix,minus_matrix,scalar_matrix,cross_product,n_1_inverse
from matplotlib import pyplot as plt
import time as timee
show_animation = True
import copy
import datetime


def main():

    # vn300 시간 파일 집어넣기
    # vn300에서 나온 시간이 이상할 수 있다(그린위치 기준 기간으로 나올 수 있음)
    # ROS 시간과 차이가 난다면 ROS 시간은 우리나라기준이고, vn300은 그린위치 기준이니까 다시 설정할것

    f = open('vn300_0311_time.csv', 'r')
    rdr = csv.reader(f)
    vn300_0311_time = []
    for line in rdr:
        row_list = []
        for i in range(len(line)):
            num = float(line[i])
            row_list.append(num)
        vn300_0311_time.append(row_list)
    #vn300_0311_time = np.array(vn300_0311_time)


    q = open('vn300_0311_time_2_sec.csv', 'w', newline='')
    wr = csv.writer(q)


    time_list = []
    for i in range(len(vn300_0311_time)):

        year = int(vn300_0311_time[i][0] + 2000)
        month = int(vn300_0311_time[i][1])
        day = int(vn300_0311_time[i][2])
        hour = int(vn300_0311_time[i][3])
        min = int(vn300_0311_time[i][4])
        sec = int(vn300_0311_time[i][5])
        ms = int(vn300_0311_time[i][6])
        ns = int(vn300_0311_time[i][5])

        #t = datetime.datetime(year,month,day,hour,min,sec,ms)
        t = datetime.datetime(year, month, day, hour, min, sec, ms*1000+ns)
        t = (t - datetime.datetime(1970, 1, 1)).total_seconds()
        t = t * 1000000000
        t = int(t)
        time_list.append([t])
    time_list = np.array(time_list)


    for i in range(len(time_list)):
        wr.writerow(time_list[i])





    q.close()





    f.close()

if __name__ == '__main__':
    main()
