#!/usr/bin/env python
# -*- coding:utf-8 -*-	# 한글 주석을 달기 위해 사용한다.

from sensor_msgs.msg import Imu, MagneticField, NavSatFix, PointCloud2
import socket
import time
from multiprocessing import Value, Process, Manager, Queue
import struct
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
import copy
import numpy as np
from scipy import linalg
import pymap3d as pm
from JU_matrix import Kalman_Filter, m_n_inverse, n_n_inverse, DCM2eul_bn, eul2DCM_bn, product_matrix_dx, \
    product_matrix, skew, plus_matrix, minus_matrix, cross_product, n_1_inverse
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2
import pcl
from ICP import *

from pcl_helper import *
import open3d as o3d
import random
import rospy
from std_msgs.msg import String
import open3d as o3d
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension, Float32MultiArray
import csv
import pandas as pd

show_animation = True



def do_voxel_grid_downssampling(pcl_data, leaf_size):
    '''
    Create a VoxelGrid filter object for a input point cloud
    :param pcl_data: point cloud data subscriber
    :param leaf_size: voxel(or leaf) size
    :return: Voxel grid downsampling on point cloud
    :https://github.com/fouliex/RoboticPerception
    '''
    vox = pcl_data.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)  # The bigger the leaf size the less information retained
    return vox.filter()


def JU_ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    # 동적할당을 numpy를 이용해 정적할당으로 바꾸는게 좋을듯함
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    return pcl_data


def LidarCallback(data):

    raw_LidarQ.empty()  # 1. 비우고

    data = JU_ros_to_pcl(data)  # 0.02초 소
    #data = data.to_array()

    # down sampling # 내가 만든것보다 압도적..
    LEAF_SIZE = 0.1
    data = do_voxel_grid_downssampling(data, LEAF_SIZE)

    data = data.to_array()

    data = np.delete(data, [2, 3], axis=1)  # z, intensity 삭제

    raw_LidarQ.put(data)  # 2. 채우고





def Localization(LidarQ, prev_LidarQ,save_mappingQ,Mapping_CNT,Kalman_pos
                 ):
    idx_init = 0  # 초기화를 위해
    cur_pose = []
    map_cnt = Mapping_CNT.value
    save_mapping = []
    save_mapping_tmp = []
    save_pos =[]
    save_pos_tmp = []


    while True:

        Mapping_CNT.value = map_cnt + 1
        if Mapping_CNT.value > 255:
            Mapping_CNT.value = 0

        # position, yaw
        data = np.array(Kalman_pos.get())
        Yaw, kalman_E, kalman_N =  data[0],data[1],data[2]

        data = LidarQ.get() # cur Lidar data

        # rotate cur lidar data to enu
        cur_mapping = []
        for i in range(len(data)):
            X = kalman_E - math.cos(Yaw * math.pi / 180) * data[i][1] + math.sin(
                Yaw * math.pi / 180) * (data[i][0] + 0.7)
            Y = kalman_N + math.sin(Yaw * math.pi / 180) * data[i][1] + math.cos(
                Yaw * math.pi / 180) * (data[i][0] + 0.7)

            cur_mapping.append([X,Y])

        save_mapping = np.array(save_mapping_tmp)

        save_pos_tmp.append([kalman_E,kalman_N])

        save_pos = np.array(save_pos_tmp)
        cur_pose = np.array([kalman_E , kalman_N])
        # print(cur_mapping)
        # print(cur_pose)
        df_1 = pd.DataFrame(cur_mapping)
        df_2 = pd.DataFrame(cur_pose)

        df_1.to_csv('save_mapping.csv',mode = 'a', index=False)
        df_2.to_csv('save_pos.csv',mode = 'a', index=False)

        # if show_animation:
        #     plt.cla()
        #     # for stopping simulation with the esc key.
        #     plt.gcf().canvas.mpl_connect(
        #         'key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])
        #
        #     try:
        #         plt.plot(cur_mapping[:, 0], cur_mapping[:, 1], ".b", markersize=3)
        #         plt.plot(save_mapping[:, 0], save_mapping[:, 1], ".g", markersize=3)
        #         plt.plot(save_pos[:,0], save_pos[:,1], "or", markersize=3)
        #         plt.xlim(kalman_E - 30, kalman_E + 30 )
        #         plt.ylim(kalman_N - 30, kalman_N + 30)
        #
        #
        #     except:
        #         pass
        #
        #     plt.grid(True)
        #     plt.pause(0.00001)





def GNSS_Callback(data):

    kalman_lat.value, kalman_lon.value, kalman_alt.value = data.data[0],data.data[1],data.data[2]
    current_lat.value, current_lon.value, current_alt.value = data.data[3],data.data[4],data.data[5]
    kalman_roll.value, kalman_pitch.value, kalman_yaw.value = data.data[6],data.data[7],data.data[8]
    current_accel_x.value, current_accel_y.value, current_accel_z.value = data.data[9],data.data[10],data.data[11]
    current_vel_x.value, current_vel_y.value, current_vel_z.value = data.data[12],data.data[13],data.data[14]
    current_quat_x.value, current_quat_y.value, current_quat_z.value = data.data[15],data.data[16],data.data[17]
    current_quat_w.value = data.data[18]
    GPS_NED_N.value, GPS_NED_E.value, GPS_NED_D.value = data.data[19],data.data[20],data.data[21]
    GPS_ENU_E.value, GPS_ENU_N.value, GPS_ENU_U.value= data.data[22],data.data[23],data.data[24]


def ENU_Callback(data):

    cnt = ENU_CNT.value
    kalman_ENU_E.value, kalman_ENU_N.value, kalman_ENU_U.value = data.data[0],data.data[1],data.data[2]
    ENU_CNT.value = cnt + 1

    if ENU_CNT.value > 255:
        ENU_CNT.value = 0

def NED_Callback(data):

    kalman_NED_N.value, kalman_NED_E.value, kalman_NED_D.value = data.data[0],data.data[1],data.data[2]


def Subscribe_for_Mapping():

    rospy.init_node('Subscribe_for_Mapping_only_Lidar', anonymous=True)

    rospy.Subscriber("/3_velodyne_points_Clustering", PointCloud2, LidarCallback)

    rospy.Subscriber("/INS", Float32MultiArray, GNSS_Callback)

    rospy.Subscriber("/ENU", Float32MultiArray, ENU_Callback)

    rospy.Subscriber("/NED", Float32MultiArray, NED_Callback)

    rospy.spin()


def Q_save(LidarQ,raw_LidarQ,Mapping_CNT,kalman_ENU_E,kalman_ENU_N,kalman_ENU_U,kalman_yaw,
                 GPS_NED_N, GPS_NED_E, GPS_NED_D,
                 GPS_ENU_E, GPS_ENU_N, GPS_ENU_U,Kalman_pos):

    save_data = []
    cnt = Mapping_CNT.value

    while True:
        data = np.array(raw_LidarQ.get())
        save_data.append(data)
        if Mapping_CNT.value != cnt:
            LidarQ.empty()
            LidarQ.put(save_data[0])
            Kalman_pos.empty()
            Kalman_pos.put([kalman_yaw.value,kalman_ENU_E.value,kalman_ENU_N.value])
            del save_data[0]


if __name__ == '__main__':
    current_lat = Value('d', 0.0)
    current_lon = Value('d', 0.0)
    current_alt = Value('d', 0.0)
    current_accel_x = Value('d', 0.0)
    current_accel_y = Value('d', 0.0)
    current_accel_z = Value('d', 0.0)
    current_vel_x = Value('d', 0.0)
    current_vel_y = Value('d', 0.0)
    current_vel_z = Value('d', 0.0)
    current_quat_x = Value('d', 0.0)
    current_quat_y = Value('d', 0.0)
    current_quat_z = Value('d', 0.0)
    current_quat_w = Value('d', 0.0)
    current_yaw = Value('d', 0.0)

    # obj = [dist,x_cent, y_cent, x_min,x_max, y_min, y_max]
    obj1_dist = Value('d', 0.0)
    obj2_dist = Value('d', 0.0)
    obj3_dist = Value('d', 0.0)
    obj4_dist = Value('d', 0.0)

    obj1_x_cent = Value('d', 0.0)
    obj2_x_cent = Value('d', 0.0)
    obj3_x_cent = Value('d', 0.0)
    obj4_x_cent = Value('d', 0.0)

    obj1_y_cent = Value('d', 0.0)
    obj2_y_cent = Value('d', 0.0)
    obj3_y_cent = Value('d', 0.0)
    obj4_y_cent = Value('d', 0.0)

    obj1_x_min = Value('d', 0.0)
    obj2_x_min = Value('d', 0.0)
    obj3_x_min = Value('d', 0.0)
    obj4_x_min = Value('d', 0.0)

    obj1_x_max = Value('d', 0.0)
    obj2_x_max = Value('d', 0.0)
    obj3_x_max = Value('d', 0.0)
    obj4_x_max = Value('d', 0.0)

    obj1_y_min = Value('d', 0.0)
    obj2_y_min = Value('d', 0.0)
    obj3_y_min = Value('d', 0.0)
    obj4_y_min = Value('d', 0.0)

    obj1_y_max = Value('d', 0.0)
    obj2_y_max = Value('d', 0.0)
    obj3_y_max = Value('d', 0.0)
    obj4_y_max = Value('d', 0.0)

    IMU_CTC = Value('d', 0.0)
    IMU_CNT = Value('d', 0.0)
    GPS_CTC = Value('d', 0.0)
    GPS_CNT = Value('d', 0.0)
    LIDAR_CTC = Value('d', 0.0)
    LIDAR_CNT = Value('d', 0.0)

    LIDAR_obj_1 = Value('d', 0.0)
    LIDAR_obj_2 = Value('d', 0.0)
    LIDAR_obj_3 = Value('d', 0.0)

    LIDAR_obj_4 = Value('d', 0.0)

    current_roll = Value('d', 0.0)
    current_pitch = Value('d', 0.0)

    kalman_yaw = Value('d', 0.0)
    kalman_pitch = Value('d', 0.0)
    kalman_roll = Value('d', 0.0)

    kalman_lat = Value('d', 0.0)
    kalman_lon = Value('d', 0.0)
    kalman_alt = Value('d', 0.0)

    kalman_NED_N = Value('d', 0.0)
    kalman_NED_E = Value('d', 0.0)
    kalman_NED_D = Value('d', 0.0)

    kalman_ENU_E = Value('d', 0.0)
    kalman_ENU_N = Value('d', 0.0)
    kalman_ENU_U = Value('d', 0.0)

    GPS_NED_N = Value('d', 0.0)
    GPS_NED_E = Value('d', 0.0)
    GPS_NED_D = Value('d', 0.0)

    GPS_ENU_E = Value('d', 0.0)
    GPS_ENU_N = Value('d', 0.0)
    GPS_ENU_U = Value('d', 0.0)

    ENU_CNT = Value('d', 0.0)
    Mapping_CNT = Value('d', 0.0)

    raw_LidarQ = Queue()
    LidarQ = Queue()
    prev_LidarQ = Queue()
    save_mappingQ = Queue()
    processedQ = Queue()
    Kalman_pos = Queue()

    th5 = Process(target=Subscribe_for_Mapping, args=())


    th4 = Process(target=Localization, args=(LidarQ, prev_LidarQ,save_mappingQ,Mapping_CNT,Kalman_pos
                                             ))


    th3 = Process(target=Q_save, args=(LidarQ,raw_LidarQ,Mapping_CNT,kalman_ENU_E,kalman_ENU_N,kalman_ENU_U,kalman_yaw,
                 GPS_NED_N, GPS_NED_E, GPS_NED_D,
                 GPS_ENU_E, GPS_ENU_N, GPS_ENU_U,Kalman_pos))

    th3.start()
    th5.start()
    th4.start()
