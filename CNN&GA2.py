import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
import math
import random
import re
import tensorflow as tf



# 载入卷积神经网络的权值参数
W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, "save/parameters.ckpt")
    W1 = W1.eval()
    W2 = W2.eval()
    parameters = {"W1": W1,
                  "W2": W2}

#  实现卷积神经网络前向运算
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, 2, activation_fn=None)

    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    return cost


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
        coded_generation.append(list(map(int,x[0] + x[1] + x[2] + x[3])))
        m = m + 1
    return np.array(coded_generation)

def individual_decode(generation):
    """
    :param generation: encoded generation
    :return: generation after decode
    """
    decoded_generation = []
    for i in generation:
        j = "".join(map(lambda x:str(x),i))
        n = 1
        x = []

        a = int(j[0:4],2)
        b = int(j[4:8],2)
        c = int(j[8:12],2)
        x.append(int(a*100 + b*10 + c))

        a = int(j[12:16], 2)
        b = int(j[16:20], 2)
        c = int(j[20:24], 2)
        x.append(int(a * 100 + b * 10 + c))

        a = int(j[24:28], 2)
        b = int(j[28:32], 2)
        c = int(j[32:36], 2)
        x.append(round(a + b / 10 + c / 100, 2))

        a = int(j[36:40], 2)
        b = int(j[40:44], 2)
        c = int(j[44:48], 2)
        x.append(round(a + b/10 + c/100,2))

        decoded_generation.append(x)
    return np.array(decoded_generation)

def fitness_function(generation,img,template):
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
        target_img = img[y:y + int(M*h0),x:x + int(M*w0)]
        X = []
        X.append(target_img)
        X = np.array(X)
        X = X / 255.
        Z3 = forward_propagation(X, parameters)
        prob = tf.nn.softmax(Z3)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            prob = prob.eval()
        fitness.append(prob[0][0])
    return fitness


def selection(population,fitness):
    #轮盘赌选择
    fitness_sum=[]
    for i in range(len(fitness)):
        if i == 0:
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

# def seperate(generation):
#     """
#     :param generation: generation in string matrix form
#     :return: generation in binary matrix form
#     """
#     g = []
#     for i in range(generation.shape[0]):
#         g.append(list(map(int, generation[i][0] + generation[i][1] + generation[i][2] + generation[i][3])))
#     g = np.array(g)
#     return g

# def merge(generation):
#     """
#     :param generation: generation in binary array form
#     :return: generation in string  array form
#     """
#     m,n = generation.shape
#     new_g = []
#     new_generation = []
#     for x in range(m):
#         new_generation.append(list(map(str,generation[x])))
#         a = ''
#         b = ''
#         c = ''
#         d = ''
#         count = 0
#         for y in range(n):
#             if count < 12:
#                 a = a + new_generation[x][y]
#             if 11 < count < 24:
#                 b = b + new_generation[x][y]
#             if 23< count < 36:
#                 c = c + new_generation[x][y]
#             if 35 < count <48:
#                 d = d + new_generation[x][y]
#             count = count +1
#         new_g.append([a,b,c,d])
#     return np.array(new_g)

def list2dict(f,generation):
    new_dict = {}
    a = 0
    for i in f:
        new_dict[i] = generation[a]
        a = a+1
    return new_dict


if __name__ == '__main__':

    # test: 600 * 400 template: 330 * 100
    test = cv2.imread('./data/test.jpg')
    template = cv2.imread('./data/template.jpg')

    h,w,c = template.shape
    # initialize individual
    population = np.array(individual_initialize(20))
    t = []
    max_iter = 10

    # template_pixel = 0
    # for i in range(template.shape[1]):
    #     for j in range(template.shape[0]):
    #         template_pixel = template_pixel + template[j][i]

    for i in range(max_iter):
        img = test.copy()
        f = fitness_function(population,test,template)
        dict = list2dict(f, population)
        optimal_value = sorted(dict.keys())[-1]
        print(optimal_value)

        optimal_individua = dict[optimal_value]

        x = int(optimal_individua[0])
        y = int(optimal_individua[1])

        if optimal_value > 0.72:
            break

        t.append(optimal_value)

        #selection
        population_new = selection(population,f)
        population_new = individual_encode(population_new)


        #crossing
        offspring = crossing(population_new,0.8)
        #mutation
        population = mutation(offspring,0.02)
        population = individual_decode(population)
        # print(population)

    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0 , 255), 2)
    cv2.imshow('MyWindow', img)
    cv2.waitKey(0)


