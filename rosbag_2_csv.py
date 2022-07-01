import csv
from scipy import io
from mat4py import loadmat
import numpy as np

def main():

    f_wr = open('RTK_LLH.csv', 'w', newline='')

    f = open('fix.csv', 'r')

    rdr = csv.reader(f)

    wr = csv.writer(f_wr)

    for line in rdr:
        row_list = []

        if line == 0:
            pass

        for i in range(len(line)):
            num = float(line[i])
            row_list.append(num)

        wr.writerow(row_list)

    f.close()







if __name__ == '__main__':
    main()