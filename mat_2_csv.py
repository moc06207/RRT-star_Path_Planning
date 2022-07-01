import csv
from scipy import io
from mat4py import loadmat
import numpy as np

def main():

    data = loadmat('kcity_rtk.mat')

    data = list(data.values())
    f = open('kcity_rtk.csv', 'w', newline='')
    wr = csv.writer(f)

    for i in range(len(data[0])):

        wr.writerow(data[0][i])

    f.close()

    data_1 = loadmat('kcity_span.mat')

    data_1 = list(data_1.values())
    f_1 = open('kcity_span.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data_1[0])):
        wr.writerow(data_1[0][i])

    f_1.close()

    data_1 = loadmat('vid.mat')

    data_1 = list(data_1.values())

    f_1 = open('vid_E_INS_vid.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data_1[0])):

        wr.writerow(data_1[0][i])

    f_1.close()

    f_1 = open('vid_N_INS_vid.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data_1[1])):
        wr.writerow(data_1[1][i])

    f_1.close()

    f_1 = open('vid_save_X_vid.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data_1[2])):
        wr.writerow(data_1[2][i])

    f_1.close()

    # -----------------------------------

    data_msg1 = loadmat('kcity_VID.mat')


    data = data_msg1['msg1']['timeGps']


    f_1 = open('kcity_VID_timeGps.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()f

    #-----------------------------------

    data = data_msg1['msg1']['timeUtc']['year']

    f_1 = open('kcity_VID_timeUtc_year.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['timeUtc']['month']

    f_1 = open('kcity_VID_timeUtc_month.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['timeUtc']['day']

    f_1 = open('kcity_VID_timeUtc_day.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['timeUtc']['hour']

    f_1 = open('kcity_VID_timeUtc_hour.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['timeUtc']['min']

    f_1 = open('kcity_VID_timeUtc_min.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['timeUtc']['sec']

    f_1 = open('kcity_VID_timeUtc_sec.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['timeUtc']['ms']

    f_1 = open('kcity_VID_timeUtc_ms.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['uncompMag']

    f_1 = open('kcity_VID_uncompMag.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['uncompAccel']

    f_1 = open('kcity_VID_uncompAccel.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['uncompGyro']

    f_1 = open('kcity_VID_uncompGyro.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['temp']

    f_1 = open('kcity_VID_temp.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['pres']

    f_1 = open('kcity_VID_pres.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['deltaTheta']

    f_1 = open('kcity_VID_deltaTheta.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['deltaV']

    f_1 = open('kcity_VID_deltaV.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['accel']

    f_1 = open('kcity_VID_accel.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['gyro']

    f_1 = open('kcity_VID_gyro.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['ypr']

    f_1 = open('kcity_VID_ypr.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['accelNed']

    f_1 = open('kcity_VID_accelNed.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['insStatus']

    f_1 = open('kcity_VID_insStatus.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['posLla']

    f_1 = open('kcity_VID_posLla.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['velBody']

    f_1 = open('kcity_VID_velBody.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()

    # -----------------------------------

    data = data_msg1['msg1']['velNed']

    f_1 = open('kcity_VID_velNed.csv', 'w', newline='')
    wr = csv.writer(f_1)

    for i in range(len(data)):
        wr.writerow(data[i])

    f_1.close()










if __name__ == '__main__':
    main()