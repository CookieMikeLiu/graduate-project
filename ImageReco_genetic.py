import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
import math
import random
import re

# test: 600 * 400 template: 330 * 100
img1 = cv2.imread('./data/test.jpg')
img2 = cv2.imread('./data/template.jpg')

def individual_initialize(n):
    '''
    :param n: the number of the individual we need to initialize
    :return generation: the whole generation after initialzing
    '''
    generation = []
    for i in range(n):
        x = np.random.randint(0,270)
        y = np.random.randint(0,300)
        # seta = round(np.random.uniform(0,2*math.pi),2)
        M = round(np.random.uniform(0.3,2),2)
        seta = 0
        # M = 1
        a = [x,y,seta,M]
        generation.append(a)
    return generation

def individual_encode(generation):
    """
    :param generation: Matrix of every individual by ten decimal, 4 * n
    :return: Matrix of every individual after encode by two decimal, 4 * n
    """
    coded_generation = []
    m = 0
    for i in generation:
        n = 0
        x = []
        for j in i:
            n = n + 1
            if n <= 2:
                a = '{:04b}'.format(math.floor(j / 100))
                b = '{:04b}'.format(math.floor((j % 100) / 10))
                c = '{:04b}'.format(math.floor((j % 100) % 10))
                j = a + b + c
                x.append(j)
            else:
                j = j * 100
                a = '{:04b}'.format(math.floor(j / 100))
                b = '{:04b}'.format(math.floor((j % 100) / 10))
                c = '{:04b}'.format(math.floor((j % 100) % 10))
                j = a + b + c
                x.append(j)
        coded_generation.append(x)
        m = m + 1
    return np.array(coded_generation)

def individual_decode(generation):
    """
    :param generation: encoded generation
    :return: generation after decode
    """
    decoded_generation = []
    m = 0
    for i in generation:
        n = 0
        x = []
        for j in i:
            n = n + 1
            if n <= 2:
                a = int(j[0:4],2)
                b = int(j[4:8],2)
                c = int(j[8:12],2)
                x.append(int(a*100 + b*10 + c))
            else:
                a = int(j[0:4], 2)
                b = int(j[4:8], 2)
                c = int(j[8:12], 2)
                x.append(round(a + b/10 + c/100,2))
        decoded_generation.append(x)
        m = m + 1
    return np.array(decoded_generation)

def fitness_function(generation,img,template,template_pixel):
    """
    :param template: template image
    :param img: test image
    :param generation: the generation after decode
    :return: fitness
    """
    fitness = []
    for a in generation:
        x = int(a[0])
        y = int(a[1])
        seta = a[2]
        M = a[3]

        w0 = template.shape[1]
        h0 = template.shape[0]

        w = img.shape[1]
        h = img.shape[0]

        sum_pixel = 0

        for i in range(x,x+w0):
            for j in range(y,y+h0):
                # x_new = round(M * (j * math.cos(seta) - i * math.sin(seta) + f1))
                # y_new = round(M * (j * math.sin(seta) + i * math.cos(seta) + f2))
                x_new = round((i - x) * math.cos(seta) - (j - y) * math.sin(seta) + x)
                y_new = round((i - x) * math.sin(seta) + (j - y) * math.cos(seta) + y)
                x_new = int(round(x_new + M * (i - x_new)))
                y_new = int(round(y_new + M * (j - y_new)))
                if x_new >= w or y_new >= h:
                    sum_pixel = sum_pixel + 0
                else:
                    sum_pixel = sum_pixel + img[y_new][x_new]
        f = sum_pixel/template_pixel
        # f = abs(1-f)
        fitness.append(f)
    return fitness


def selection(population,fitness):
    #轮盘赌选择

    fitness_sum=[]
    for i in range(len(fitness)):
        if i ==0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i-1]+fitness[i])

    for i in range(len(fitness_sum)):
        fitness_sum[i]/=sum(fitness)
    #select new population
    population_new=[]
    for i in range(len(fitness)):
        rand=np.random.uniform(0,1)
        for j in range(len(fitness)):
            if j==0:
                if 0<rand and rand<=fitness_sum[j]:
                    population_new.append(population[j])
            else:
                if fitness_sum[j-1]<rand and rand<=fitness_sum[j]:
                    population_new.append(population[j])
    return np.array(population_new)



def crossing(g,pc = 0.6):
    """
    :param generation: old generation
    :param pc:crossing probability
    :return: new generation
    """
    m, n = g.shape
    updatageneration = np.zeros((m,n),dtype=int)
    numbers = int(m * pc)
    if numbers % 2 != 0:
        numbers += 1

    index = random.sample(range(m),numbers)
    for j in range(m):
        if not index.__contains__(j):
            updatageneration[j,:] = g[j,:]

    while len(index) > 0:
        a = index.pop()
        b = index.pop()
        crossoverPoint1 = random.sample(range(1,n-1),1)[0]

        updatageneration[a,0:crossoverPoint1] = g[a,0:crossoverPoint1]
        updatageneration[a,crossoverPoint1:] = g[b,crossoverPoint1:]
        updatageneration[b,0:crossoverPoint1] = g[b,0:crossoverPoint1]
        updatageneration[b,crossoverPoint1:] = g[a,crossoverPoint1:]

    return updatageneration

def mutation(g,pm = 0.01):
    """

    :param generation:the generation after crossing
    :param pm:the probability of mutation
    :return:the new generation after mutation
    """
    m,n = g.shape
    updatageneration = np.copy(g)
    gen_num = int(m * n * pm)
    mutation_index = random.sample(range(0, m*n),gen_num)
    for i in mutation_index:
        individual_index = i // n
        gene_index = i % n
        if updatageneration[individual_index,gene_index] == 0 :
            updatageneration[individual_index,gene_index] == 1
        else:
            updatageneration[individual_index,gene_index] == 0
    return updatageneration

def seperate(generation):
    """
    :param generation: generation in string matrix form
    :return: generation in binary matrix form
    """
    g = []
    for i in range(generation.shape[0]):
        g.append(list(map(int, generation[i][0] + generation[i][1] + generation[i][2] + generation[i][3])))
    g = np.array(g)
    return g

def merge(generation):
    """
    :param generation: generation in binary array form
    :return: generation in string  array form
    """
    m,n = generation.shape
    new_g = []
    new_generation = []
    for x in range(m):
        new_generation.append(list(map(str,generation[x])))
        a = ''
        b = ''
        c = ''
        d = ''
        count = 0
        for y in range(n):
            if count < 12:
                a = a + new_generation[x][y]
            if 11 < count < 24:
                b = b + new_generation[x][y]
            if 23< count < 36:
                c = c + new_generation[x][y]
            if 35 < count <48:
                d = d + new_generation[x][y]
            count = count +1
        new_g.append([a,b,c,d])
    return np.array(new_g)

def list2dict(f,generation):
    new_dict = {}
    a = 0
    for i in f:
        new_dict[i] = generation[a]
        a = a+1
    return new_dict


if __name__ == '__main__':

    template = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    h,w = template.shape
    # initialize individual
    population = np.array(individual_initialize(40))
    t = []
    max_iter = 10

    template_pixel = 0
    for i in range(template.shape[1]):
        for j in range(template.shape[0]):
            template_pixel = template_pixel + template[j][i]

    for i in range(max_iter):
        img = test.copy()
        f = fitness_function(population,test,template,template_pixel)
        # print(f)
        dict = list2dict(f, population)
        optimal_value = sorted(dict.keys())[-1]
        optimal_individua = dict[optimal_value]
        x = int(optimal_individua[0])
        y = int(optimal_individua[1])
        print(optimal_individua)
        print(x+w,y+h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0 , 255), 2)
        cv2.imshow('MyWindow', img)
        cv2.waitKey(0)

        t.append(optimal_value)

        print(t)

        #selection
        population_new = selection(population,f)
        population_new = individual_encode(population_new)
        population_new = seperate(population_new)
        #crossing
        offspring = crossing(population_new,0.7)
        #mutation
        population = mutation(offspring,0.02)
        population = merge(population)
        population = individual_decode(population)



    # for i in range(max_iter):
    #     # select optimal individual
    #     f = fitness_function(g, test, template)
    #     dict = list2dict(f, g)
    #     optimal_value = sorted(dict.keys())[-1]
    #     optimal_individual = dict[optimal_value]
    #     print(optimal_value,i)
    #     if optimal_value > 0.5:
    #         break
    #     else:
    #         # select optimal individual
    #         s = selection(g,f)
    #         generation = individual_encode(s)
    #         # produce new offspring
    #         new_g = crossing(generation)
    #         m = mutation(new_g)
    #         # generate new generation
    #         g = np.vstack((generation,m))
    #         g = individual_decode(g)
    # print(optimal_value,optimal_individual)
