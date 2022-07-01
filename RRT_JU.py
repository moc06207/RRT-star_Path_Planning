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

        self.nodeRad = 0
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

    def add_node(self):
        pass

    def remove_node(self):
        pass

    def add_edge(self):
        pass
    def remove_edge(self):
        pass

    def number_of_nodes(self):
        pass
    def distance(self):
        pass

    def nearest(self):
        pass

    def isFree(self):
        pass

    def cross_obs(self):
        pass

    def connect(self):
        pass

    def step(self):
        pass

    def path_to_goal(self):
        pass

    def get_path_coords(self):
        pass

    def bias(self):
        pass

    def expand(self):
        pass

    def cost(self):
        pass


if __name__ == '__main__':
    dim = (600,1000)
    start = (50,50)
    goal = (510,510)
    obsdim = 30
    obsnum = 50

    pygame.init()
    map = RRTMap(start,goal,dim,obsdim,obsnum)
    graph = RRTGraph(start, goal, dim, obsdim, obsnum)

    obstacles = graph.make_obs()
    map.draw_map(obstacles)
    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)