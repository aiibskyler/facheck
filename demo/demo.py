import numpy as np


def sigmoid(x):
    # sigmoid激活函数: f(x) = 1 / (1 + e^(-x))，将无限制的输入转换为可预测形式的输出，输出介于0和1。
    # 即把 (−∞,+∞) 范围内的数压缩到 (0, 1)以内。正值越大输出越接近1，负向数值越大输出越接近0。
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # 求激活函数的偏导数: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_real, y_expect):
    # 计算损失函数，其实就是所有数据方差的平均值(均方误差),预测结果越好，损失就越低，训练神经网络就是将损失最小化
    # y_real 和 y_expect 是同样长度的数组.
    return ((y_real - y_expect) ** 2).mean()


class NeuralNetworkDemo:
    '''
    一个神经网络包括:
      - 2个输入
      - 1个有2 个神经元 (h1, h2)的隐藏层
      - 一个神经元的输出层 (o1)
   '''

    def __init__(self):
        # 随机初始化权重值
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # 随机初始化偏置值
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x包含x[0]、x[1],计算前馈：神经元的输入向前传递获得输出的过程
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    # 训练过程：
    # 1、从数据集中选择一个样本；
    # 2、计算损失函数对所有权重和偏置的偏导数；
    # 3、使用更新公式更新每个权重和偏置；
    # 4、回到第1步。
    def train(self, data, all_y_reals):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_reals is a numpy array with n elements.
          Elements in all_y_reals correspond to those in data.
        '''

    learn_rate = 0.1

    # 循环遍历整个数据集的次数
    iterations = 1000

    for iteration in range(iterations):
        for x, y_real in zip(data, all_y_reals):
            # 计算h1的前馈
            sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
            h1 = sigmoid(sum_h1)

            # 计算h2的前馈
            sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
            h2 = sigmoid(sum_h2)

            # 计算o1的前馈
            sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
            o1 = sigmoid(sum_o1)
            y_expect = o1

            # --- 计算部分偏导数
            # --- d_L_d_w1 表示 "偏导数 L / 偏导数 w1"
            d_L_d_ypred = -2 * (y_real - y_expect)

            # 神经元 o1
            d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
            d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
            d_ypred_d_b3 = deriv_sigmoid(sum_o1)

            d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
            d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

            # 神经元 h1
            d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
            d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
            d_h1_d_b1 = deriv_sigmoid(sum_h1)

            # 神经元 h2
            d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
            d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
            d_h2_d_b2 = deriv_sigmoid(sum_h2)

            # --- 更新权重值和偏置值
            # 神经元 h1
            self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
            self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
            self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

            # 神经元 h2
            self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
            self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
            self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

            # 神经元 o1
            self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
            self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
            self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

        # 每一次循环计算总的损失 --- Calculate total loss at the end of each iteration
        if iteration % 10 == 0:
            y_expects = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_reals, y_expects)
            print("Iteration %d Mes loss: %.3f" % (iteration, loss))


# 定义数据集
data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
])
all_y_reals = np.array([
    1,
    0,
    0,
    1,
])

# 训练神经网络
network = NeuralNetworkDemo()
network.train(d)
