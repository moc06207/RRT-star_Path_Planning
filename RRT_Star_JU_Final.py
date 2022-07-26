import random
import math
import pygame
import matplotlib.pyplot as plt
import numpy as np
show_animation = True
import time

# TODO
# 0704 
# step다시 보기
# 뭔가 새로 점 갱신될때 제대로 안되는듯

class RRTGraph():
    def __init__(self, start, goal, Map_dim, Obstacle_list):

        self.Node = np.array([[start[0], start[1]]])

        self.Map_h, self.Map_w = Map_dim

        (x, y) = start
        self.start = start

        self.goal = goal

        self.goal_Flag = False

        self.End_Flag = False

        self.mahs, self.mapw = Map_dim

        self.x = []
        self.y = []

        self.Parent = np.array([0])

        self.parent = []

        # Initialize the tree
        self.y.append(y)

        self.parent.append(0)

        # obstacle
        self.Obstacle_list = Obstacle_list

        self.remeber_Node =  np.array([[start[0], start[1]]])

        # path
        self.goalstate = None

        self.Paht = np.array([])
        self.paht = []

        self.final_node = 0

        self.rand = (0,0)

        self.tmp = [0,0,0,0,0,0]

        self.near = (0,0)

        self.test_rrt_star = (0, 0)
        self.test = (0, 0)

        self.near_node = np.empty((0, 2), float)

        self.Cost = np.array([0])


        self.target_node = (0,0)


    def add_node(self,n,x,y):

        self.Node = np.insert(self.Node, n,  np.array([[x, y]]), axis=0)

        self.remeber_Node = np.insert(self.Node, n,  np.array([[x, y]]), axis=0)

        return True

    def remove_node(self,n):
        self.Node = np.delete(self.Node, n , axis = 0)

    def add_edge(self,parent,child):

        self.Parent = np.insert(self.Parent, child, np.array([parent]), axis=0)


    def remove_edge(self,n):

        self.Parent = np.delete(self.Parent, n , axis = 0)

    def number_of_nodes(self):
        return  len(self.Node)

    def distance(self,n1,n2):

        dist = np.hypot(self.Node[n1][0] - self.Node[n1][1],self.Node[n2][0] - self.Node[n2][1])


        (x1,y1) = (self.Node[n1][0],self.Node[n1][1])
        (x2,y2) = (self.Node[n2][0], self.Node[n2][1])
        px = (float(x1)-float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2

        return (px+py) ** (0.5)

    def sample_envir(self): # 효과적으로 수정 필요
        x = int(random.uniform(self.start[0],self.mapw))
        y = int(random.uniform(self.start[1], self.mapw))

        return x,y

    def nearest(self, n): # search neasrst node to goal

        dmin = self.distance(0,n)

        nnear = 0

        for i in range(0,n):

            if self.distance(i,n) < dmin:
                dmin = self.distance(i,n)
                nnear = i

        return nnear


    def isFree(self):
        n = self.number_of_nodes() - 1

        x2, y2 = self.Node[n][0], self.Node[n][1]

        Obs = self.Obstacle_list.copy()

        for i in range(len(Obs)):
            d = (Obs[i][0] - x2) ** 2 + (Obs[i][1] - y2) ** 2

            if d >= Obs[i][2] ** 2:
                pass
                # return False
            else:
                if d != 0:
                    self.remove_node(n)
                    return False  # collision
        return True



    def cross_obs(self,x1,x2,y1,y2): # check collision

        Obs = self.Obstacle_list.copy()

        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        for i in range(len(Obs)):
            d = (Obs[i][0] - x2) ** 2 + (Obs[i][1] - y2) ** 2

            if d >= (Obs[i][2]+0.5) ** 2:
                pass
                # return False
            else:
                if d != 0:
                    return True  # collision
        return False


    def connect(self,n1,n2):



        (x1,y1) = (self.Node[n1][0], self.Node[n1][1])

        (x2, y2) = (self.Node[n2][0], self.Node[n2][1])

        if self.cross_obs(x1,x2,y1,y2): # check collision

            self.remove_node(n2)

            return False
        else:

            self.add_edge(n1,n2)

            self.Cost = np.append(self.Cost, np.array([((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 ]))

            return True

    def cal_near_node(self,new_idx,dist):

        tmp = np.empty((0, 2), float)

        for i in range(0, new_idx):

            if self.distance(i, new_idx) < dist:
                node_idx = i

                # i = node idx

                tmp = np.append(tmp, np.array([[node_idx, self.distance(i, new_idx)]]), axis=0)

        return tmp


    def check_parent_in_near_node(self,parent_idx):

        for i in range(len(self.near_node)):

            # 부모 노드 존재

            if parent_idx == int(self.near_node[i][0]):

                return True
            else:
                pass
        # 아무것도 없다면
        return False



    def cal_cost(self, new_idx):


        # 처음 탐색한 node의 cost

        min_cost = 99999
        min_idx = 0

        for i in range(len(self.near_node)):
            near_idx = int(self.near_node[i][0])

            Cost = self.Cost[near_idx]

            # 처음 탐색한 node의 parent
            parent_idx = int(self.Parent[near_idx])

            # near node 내부에 parent 있는지 확인
            while True:

               # near node 안에 parent 존재
                if self.check_parent_in_near_node(parent_idx):

                    Cost = Cost + self.Cost[parent_idx]

                    parent_idx = int(self.Parent[parent_idx])

                    if parent_idx == 0:
                        break

                else:
                    break

            if Cost <= min_cost: # 이전 cost 보다 작을경우 갱신
                min_cost = Cost

                # 최소 cost 를 가지는 index
                min_idx = near_idx

            else:
                pass

        self.test_rrt_star = np.array([self.Node[min_idx][0],self.Node[min_idx][1]])
        # COST 추출까지 완료!!!!!

        return min_idx



    def best_parent(self,x,y):

        # new_idx = 세로 생성한 node
        new_idx = self.number_of_nodes() - 1

        self.target_node = np.array([self.Node[new_idx][0],self.Node[new_idx][1]])

        dist = 30

        # 주변 노드 탑색
        self.near_node = self.cal_near_node(new_idx,dist)

        # cost 계산
        # new_idx 에서 near node 중 가장 cost 가 적은 idx 반환
        min_idx = self.cal_cost(new_idx)

        return min_idx


    def step(self,nnear,nrand, dmax = 4):

        d = self.distance(nnear,nrand) # 맨 처음과 맨 마지막 거리?

        # 초기화
        min_idx = nnear


        if d > dmax:

            (xnear, ynear) = (self.Node[nnear][0],self.Node[nnear][1])
            (xrand, yrand) = (self.Node[nrand][0], self.Node[nrand][1])
            self.near = (xnear, ynear)
            self.rand = (xrand, yrand)

            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py,px)

            # 일정 거리에 대해 새로운 좌표 생성
            (x,y) = (int(xnear + dmax * math.cos(theta)), int(ynear + dmax * math.sin(theta)))
            self.test = (x,y)

            self.remove_node(nrand) # 마지막 노드를 지운건가?

            if abs(x - self.goal[0]) < dmax and abs(y - self.goal[1]) < dmax: # find final path
                self.add_node(nrand, self.goal[0], self.goal[1])

                self.goalstate = nrand

                self.goal_Flag = True

            else:
                self.add_node(nrand, x, y)

                min_idx = self.best_parent(x, y)

        if min_idx == None:
            min_idx = nnear

        return min_idx



    def bias(self, ngoal):
        n = self.number_of_nodes()

        self.add_node(n, ngoal[0], ngoal[1])

        nnear = self.nearest(n)

        best_parent_idx = self.step(nnear,n)

        self.connect(best_parent_idx, n)

    def expand(self):
        n = self.number_of_nodes()

        x, y = self.sample_envir()

        self.add_node(n,x,y)

        if self.isFree():

            xnearest = self.nearest(n)

            best_parent_idx = self.step(xnearest,n)

            self.connect(best_parent_idx,n)

    def plt_map(self):

        if show_animation:
            plt.cla()

            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(self.start[0], self.start[1], "xr")
            plt.plot(self.goal[0], self.goal[1], "xr")

            plt.plot(self.Node[:,0], self.Node[:,1], "og")

            for (ox, oy, size) in self.Obstacle_list:
                self.plot_circle(ox, oy, size)

            plt.plot(self.tmp[4], self.tmp[5], "ob", markersize=5)

            plt.plot(self.rand[0] , self.rand[1], "^k")
            plt.plot(self.near[0], self.near[1], "^b")
            plt.plot(self.test[0],self.test[1],"ok", markersize=5)


            # # 탐색 노드중 주변 노드 plot
            # for i in range(len(self.near_node)):
            #     idx = int(self.near_node[i][0])
            #
            #     plt.plot(self.Node[idx][0], self.Node[idx][1], "ob", markersize=5)


            plt.plot(self.target_node[0], self.target_node[1], "ok", markersize=9)
            plt.plot([self.test_rrt_star[0], self.target_node[0]], [self.test_rrt_star[1], self.target_node[1]],
                     "-b")

            if self.End_Flag:


                # 마지막 노드에서 주변 점 중 가장 cost 가 적은node
                # plt.plot(self.test_rrt_star[0], self.test_rrt_star[1], "ob", markersize=9)
                for i in range(len(self.Node)):
                    parent = self.Parent[i]
                    plt.plot([self.Node[i][0], self.Node[parent][0]], [self.Node[i][1], self.Node[parent][1]], "-k")

                plt.plot(self.final_node[:, 0], self.final_node[:, 1], "or", markersize=5)

            plt.grid(True)
            plt.pause(0.0001)


    def plot_circle(self, x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    def calc_dist_to_goal(self, x, y):
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        return math.hypot(dx, dy)

    def check_goal(self):

        if self.goal_Flag:

            cal_final_parent_id = self.goalstate
            self.final_node = []

            while True:
                if cal_final_parent_id == 0:
                    tmp = self.Node[cal_final_parent_id]

                    self.final_node.append(tmp)

                    self.final_node = np.array(self.final_node)

                    break

                tmp = self.Node[cal_final_parent_id]

                self.final_node.append(tmp)
                cal_final_parent_id = self.Parent[cal_final_parent_id]

            self.final_node = np.array(self.final_node)

            self.End_Flag =True


            return True
        else:
            return False

if __name__ == '__main__':

    obstacleList = [[10,31, 4],[20,5, 4],[15,14, 4],
                    [16,24, 4],[27,32, 4],[69,39, 4],[74,35, 4],[63,86, 4]]

    dim = (300, 300)
    start = (1, 1)
    goal = (250, 250)
    iteration = 0

    graph = RRTGraph(start, goal, dim,obstacleList)

    while (iteration < 2000):

        if iteration % 10 == 0:
           # print('.....bias....')
           graph.bias(goal)

        else:
            # print('.....expand....')
            graph.expand()

        if graph.check_goal():
            print(iteration)
            graph.plt_map()
            plt.pause(100)
            # print("!Find Path!")
            break
        else:

            pass

        iteration += 1

        # graph.plt_map()
        # plt.pause(0.0001)

    graph.plt_map()
    plt.pause(100)

