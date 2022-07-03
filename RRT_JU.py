import random
import math
import pygame
import matplotlib.pyplot as plt

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
    def __init__(self, start, goal, Map_dim, obs_dim, obs_num):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goal_Flag = False
        self.mahs, self.mapw = Map_dim
        self.x = []
        self.y = []
        self.parent = []

        # Initialize the tree
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)

        # obstacle
        self.obstacles = []
        self.obs_Dim = obs_dim
        self.obs_Num = obs_num

        # path
        self.goalstate = None
        self.paht = []

    def make_rand_rect(self): # 임의의 장애물 가장자리 생성
        upper_corner_x = int(random.uniform(0,self.mapw-self.obs_Dim))
        upper_corner_y = int(random.uniform(0, self.mapw - self.obs_Dim))

        return (upper_corner_x,upper_corner_y)

    def make_obs(self):
        obs = []

        for i in range(0,self.obs_Num):
            rectang = None
            startgoalcol = True
            while startgoalcol:
                upper = self.make_rand_rect()

                rectang = pygame.Rect(upper,(self.obs_Dim,self.obs_Num))
                if rectang.collidepoint(self.start) or rectang.collidepoint(self.goal):
                    startgoalcol = True
                else:
                    startgoalcol = False

            obs.append(rectang)
        self.obstacles = obs.copy()

        return obs

    def add_node(self,n,x,y):
        self.x.insert(n,x)
        self.y.append(y)


    def remove_node(self,n):
        self.x.pop(n)
        self.y.pop(n)


    def add_edge(self,parent,child):
        self.parent.insert(child,parent)

    def remove_edge(self,n):
        self.parent.pop(n)

    def number_of_nodes(self):
        return len(self.x)

    def distance(self,n1,n2):
        (x1,y1) = (self.x[n1],self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        px = (float(x1)-float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2
        return (px+py) ** (0.5)

    def sample_envir(self):
        x = int(random.uniform(0,self.mapw))
        y = int(random.uniform(0, self.mapw))

        return x,y

    def nearest(self, n):
        dmin = self.distance(0,n)
        nnear = 0

        for i in range(0,n):
            if self.distance(i,n) < dmin:
                dmin = self.distance(i,n)
                nnear = i
        return nnear

    def isFree(self):
        n = self.number_of_nodes() - 1
        (x,y) = (self.x[n], self.y[n])
        obs = self.obstacles.copy()
        while len(obs) > 0:
            rectang = obs.pop(0)
            if rectang.collidepoint(x,y):
                self.remove_node(n)
                return False
        return True

    def cross_obs(self,x1,x2,y1,y2):
        obs = self.obstacles.copy()
        while(len(obs)>0):
            rectang = obs.pop(0)

            for i in range(0,101):
                u = i / 100
                x = x1 * u + x2 * (1-u)
                y = y1 * u + y2 * (1 - u)
                if rectang.collidepoint(x,y):
                    return True
        return False


    def connect(self,n1,n2):

        (x1,y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])

        if self.cross_obs(x1,x2,y1,y2):
            self.remove_node(n2)
            return False
        else:
            self.add_edge(n1,n2)
            return True

    def step(self,nnear,nrand, dmax = 35):
        d = self.distance(nnear,nrand)
        if d > dmax:
            u = dmax / d
            (xnear, ynear) = (self.x[nnear],self.y[nnear])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py,px)
            (x,y) = (int(xnear + dmax * math.cos(theta)), int(ynear + dmax * math.sin(theta)))
            self.remove_node(nrand)
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
        return self.x, self.y, self.parent


    def expand(self):
        n = self.number_of_nodes()
        x, y = self.sample_envir()
        self.add_node(n,x,y)
        if self.isFree():
            xnearest = self.nearest(n)
            self.step(xnearest,n)
            self.connect(xnearest,n)
        return self.x, self.y, self.parent


        pass

    def cost(self):
        pass


if __name__ == '__main__':
    dim = (600,1000)
    start = (50,50)
    goal = (510,510)
    obsdim = 30
    obsnum = 50
    iteration = 0

    pygame.init()
    map = RRTMap(start,goal,dim,obsdim,obsnum)
    graph = RRTGraph(start, goal, dim, obsdim, obsnum)

    obstacles = graph.make_obs()
    map.draw_map(obstacles)

    while (iteration < 500):
        if iteration % 10 == 0:
            X, Y, Parent  = graph.bias(goal)
            pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad + 2 , 0) # -1 은 last node in tree
            pygame.draw.line(map.map, map.Blue,(X[-1], Y[-1]),(X[Parent[-1]], Y[Parent[-1]]),
                             map.edgeThickness)
        else:
            X, Y, Parent = graph.expand()
            pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad + 2, 0)
            pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]),
                             map.edgeThickness)
        if iteration  % 5 == 0 :
            pygame.display.update()
        iteration += 1
    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)





    # while True:
    #     x,y = graph.sample_envir()
    #     n = graph.number_of_nodes()
    #
    #     graph.add_node(n,x,y)
    #     graph.add_edge(n-1,n)
    #
    #     x1,y1 = graph.x[n],graph.y[n]
    #     x2, y2 = graph.x[n-1], graph.y[n-1]
    #
    #     if graph.isFree() :
    #         pygame.draw.circle(map.map,map.Blue, (graph.x[n],graph.y[n]),map.nodeRad,map.nodeThickness)
    #         if not graph.cross_obs(x1,x2,y1,y2):
    #             pygame.draw.line(map.map, map.grey, (x1,y1), (x2,y2), map.edgeThickness)
    #     pygame.display.update()
    # pygame.event.clear()
    # pygame.event.wait(0)