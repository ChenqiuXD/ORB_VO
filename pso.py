# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt
v_theta = 0.5*np.pi
range_x=[-1,1]
v_x = 1
range_y=[-1,1]
v_y = 1


class PSO(object):
    def __init__(self, population_size, max_steps, pA, pB):
        # firtst is theta, second is x, the third is y
        self.amount = len(pA)
        self.pA = pA
        self.pB = pB
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.population_size = population_size  # 粒子群数量
        self.dim = 3  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [[0, 2*np.pi],range_x,range_y]  # 解空间范围
        self.x = np.array([[np.random.uniform(self.x_bound[i][0], self.x_bound[i][1]) for i in
                             range(3)] for _ in range(self.population_size)])  # 初始化粒子群位置
        self.v = np.array([[np.random.uniform(self.x_bound[i][0], self.x_bound[i][1]) for i in
                            range(3)] for _ in range(self.population_size)])  # 初始化粒子群速度
        fitness = self.calculate_fitness()
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度

    # def loss_function(self, matrix):
    #     self.calculate_matrix()

    def calculate_matrix(self, x):
        """
        Rotation matrix.
        :param x: [d_theta, d_x, dy].
        :return: np.array([rotation matrix])
        """
        return np.array([[np.cos(x[0]), -np.sin(x[0]), x[1]], [np.sin(x[0]), np.cos(x[0]), x[2]],
                         [0, 0, 1]])

    def calculate_fitness(self):
        result = []
        for k in range(self.population_size):
            tem = 0
            matrix = self.calculate_matrix(self.x[k])  # 根据粒子的三个属性值计算矩阵
            for i in range(self.amount):
                error = self.pB[i] - matrix.dot(self.pA[i])  # 计算残差
                tem += np.sum(np.square(error))
            result.append(tem)
        return np.array(result)

    def evolve(self):
        # fig = plt.figure()
        for step in range(self.max_steps):

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg
                                                                                          - self.x)
            self.x = self.v + self.x
            for i in range(self.x.shape[0]):
                if self.x[i][0]<=0:
                    self.x[i][0]=0
                elif self.x[i][0]>2*np.pi:
                    self.x[i][0] = 2*np.pi
                if self.x[i][1]<=range_x[0]:
                    self.x[i][0]=range_x[0]
                elif self.x[i][1]>range_x[1]:
                    self.x[i][0] = range_x[1]
                if self.x[i][2]<=range_y[0]:
                    self.x[i][2]=range_y[0]
                elif self.x[i][2]>range_y[1]:
                    self.x[i][2] = range_y[1]



            # plt.clf()
            # plt.scatter(self.x[:, 0], self.x[:, 1], s=30, color='k')
            # plt.xlim(self.x_bound[0], self.x_bound[1])
            # plt.ylim(self.x_bound[0], self.x_bound[1])
            # plt.pause(0.01)
            fitness = self.calculate_fitness()

            # 需要更新的个体
            update_bool = np.greater(self.individual_best_fitness, fitness)
            update_id = np.argwhere(update_bool==True)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)


            # print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness,
            # np.mean(fitness)))
            if self.global_best_fitness < 0.0001:
                break
        return self.pg


# pso = PSO(10, 100, np.array([[1, 8, 1],[8, 4, 1],[6, 7, 1],[5, 2, 1]]), np.array([[3, 2, 1],
# [1, 6, 1], [10, 4, 1], [4,3,1]]))
# pso.evolve()
# plt.show()
