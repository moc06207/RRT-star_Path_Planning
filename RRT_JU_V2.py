import random
import math
import pygame
import matplotlib.pyplot as plt
import numpy as np
show_animation = True
import time

class RRTMap():
    def __init__(self,start,goal,Map_dim,obs_dim,obs_num):
        self.start = start
        self.goal = goal
        self.Maph, self.Mapw = Map_dim

        #window
        self.MapWindowName = "RRT"
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.Mapw,self.Maph))
        self.map.fill((255,255,255))

        self.nodeRad = 2
        self.nodeThickness = 0
        self.edgeThickness = 1

        self.obstacles = []
        self.obs_dim = obs_dim
        self.obs_Number = obs_num


        # color
        self.grey = (70,70,70)
        self.Blue  = (0,0,255)

    def draw_map(self,obstacles):
        pygame.draw.circle(self.map,self.Blue,self.start,self.nodeRad+5, 0)
        pygame.draw.circle(self.map, self.Blue, self.goal, self.nodeRad + 20, 1)
        self.draw_obs(obstacles)


    def draw_path(self):
        pass

    def draw_obs(self,obstacles):
        obstacles_list = obstacles.copy()
        while (len(obstacles_list)>0):
            obstacle = obstacles_list.pop(0)
            pygame.draw.rect(self.map,self.grey,obstacle)


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
        self.Obstacles = np.array([[5,5,5]])
        self.Obstacle_list = Obstacle_list
        self.obstacles = []

        # path
        self.goalstate = None

        self.Paht = np.array([])
        self.paht = []

        self.final_node = 0


    def add_node(self,n,x,y):

        self.Node = np.append(self.Node, np.array([[x, y]]), axis=0)



    def remove_node(self,n):
        self.Node = np.delete(self.Node, n , axis = 0)



    def add_edge(self,parent,child):

        self.Parent = np.append(self.Parent, np.array([parent]), axis=0)

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

    def sample_envir(self):
        x = int(random.uniform(0,self.mapw))
        y = int(random.uniform(0, self.mapw))

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

        X,Y = self.Node[n][0],self.Node[n][1]

        obs = self.obstacles.copy()

        Obs = self.Obstacles.copy()

        while len(obs) > 0:
            for i in range(len(Obs)):
                if (Obs[i][0] - X) ** 2 + (Obs[i][1] - Y) ** 2 <= Obs[i][2] ** 2:

                    self.remove_node(n)
                    return False # collistion

        return True

    def cross_obs(self,x1,x2,y1,y2):

        Obs = self.Obstacles.copy()

        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        for i in range(len(Obs)):
            try:

                d = abs(a * Obs[i][0] + b * Obs[i][1] + c) / ((a ** 2 + b ** 2) ** 0.5)

                if d >= Obs[i][2]:

                    return False
                else:
                    if d != 0:
                        return True #  collision
            except:
                return False
        return False


    def connect(self,n1,n2):

        (x1,y1) = (self.Node[n1][0], self.Node[n1][1])

        (x2, y2) = (self.Node[n2][0], self.Node[n2][1])

        if self.cross_obs(x1,x2,y1,y2):

            self.remove_node(n2)

            return False
        else:
            self.add_edge(n1,n2)

            return True

    def step(self,nnear,nrand, dmax = 2):

        d = self.distance(nnear,nrand) # ??? ????????? ??? ????????? ???????

        if d > dmax:

            u = dmax / d
            (xnear, ynear) = (self.Node[nnear][0],self.Node[nnear][1])
            (xrand, yrand) = (self.Node[nrand][0], self.Node[nrand][1])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py,px)

            # ?????? ????????? ?????? ????????? ?????? ??????
            (x,y) = (int(xnear + dmax * math.cos(theta)), int(ynear + dmax * math.sin(theta)))

            self.remove_node(nrand) # ????????? ????????? ?????????????

            if abs(x - self.goal[0]) < dmax and abs(y - self.goal[1]) < dmax:
                self.add_node(nrand, self.goal[0], self.goal[1])
                self.goalstate = nrand
                self.goal_Flag = True
            else:
                self.add_node(nrand, x, y)

    def path_to_goal(self):
        pass

    def get_path_coords(self):
        pass

    def bias(self, ngoal):
        n = self.number_of_nodes()

        self.add_node(n, ngoal[0], ngoal[1])

        nnear = self.nearest(n)

        self.step(nnear,n)

        self.connect(nnear, n)





    def expand(self):
        n = self.number_of_nodes()
        x, y = self.sample_envir()
        self.add_node(n,x,y)
        if self.isFree():
            xnearest = self.nearest(n)
            self.step(xnearest,n)
            self.connect(xnearest,n)



    def cost(self):
        pass

    def plt_map(self):

        if show_animation:
            plt.cla()

            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(self.start[0], self.start[1], "xr")
            plt.plot(self.goal[0], self.goal[1], "xr")


            try:
                plt.plot(self.Node[:,0], self.Node[:,1], "-g")
            except:
                pass

            for (ox, oy, size) in self.Obstacle_list:
                self.plot_circle(ox, oy, size)

            if self.End_Flag:
                plt.plot(self.final_node[:, 0], self.final_node[:, 1], "-r")

            plt.grid(True)
            plt.pause(0.00001)

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
        if self.calc_dist_to_goal(self.Node[-1][0], self.Node[-1][1]) <= 0.5:

            final_parent = np.unique(self.Parent.copy())
            self.final_node = []
            for i in range(len(final_parent)):
                tmp = self.Node[final_parent[i]].copy()
                self.final_node.append(tmp)

            self.final_node = np.array(self.final_node)

            print(self.final_node)
            self.End_Flag =True

            return True
        else:
            return False

if __name__ == '__main__':

    obstacleList = [(16,24, 2),(27,32, 2),(69,39, 2),(74,35, 2),(63,86, 2)]

    dim = (100, 100)
    start = (1, 1)
    goal = (30, 30)
    iteration = 0

    graph = RRTGraph(start, goal, dim,obstacleList)

    while (iteration < 500):

        if iteration % 10 == 0:
           graph.bias(goal)

        else:
            graph.expand()

        if graph.check_goal():
            print("!Find Path!")
            break
        else:
            pass

        iteration += 1

        graph.plt_map()
        plt.pause(0.01)

    graph.plt_map()
    plt.pause(100)

