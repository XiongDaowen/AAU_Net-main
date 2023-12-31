from numpy import random
import numpy as np
import math

def feature_normalize(train_data, test_data):
    data  = np.concatenate((train_data, test_data))
    mu = np.mean(data)
    std = np.std(data)
    return (train_data - mu)/std, (test_data - mu)/std

def reshape_vector(signal, outputsize):
    return signal.reshape((outputsize[0],outputsize[1]))

def h(t):
    eL = 0.02
    eR = 0.005
    f1 = 2000
    if t < 0:
        ht=0
    else:
        ht = math.exp((-eR/math.sqrt(1-eR**2))*(2*math.pi*f1*t)**2)*math.cos(2*math.pi*f1*t)
    return ht

def generate_impulse(signal_length, rotate_fre, Fs, phase=0):
    impulse = np.zeros(2*signal_length)
    T = 1/rotate_fre
    deltat=1/Fs
    t0 = 0
    deltaTk = 0
    for i in range(2*signal_length):
        for k in range(50):
            impulse[i] = impulse[i] + h((i+1)*deltat-k*T-deltaTk-t0)
    return impulse[phase:phase+signal_length]
            
def generate_signal(signal_length, rotate_fre, Fs, sigma,return_all=False, gen_normal=True, coeff=1):
    """
    这段代码是一个信号生成函数，用于生成具有特定特征和噪声的信号。以下是对代码的解读：

    参数：

    signal_length: 信号的长度。
    rotate_fre: 信号的旋转频率。注意旋转频率乘以2pi就是角频率
    Fs: 采样率（采样频率）。
    sigma: 噪声的标准差。
    return_all: 是否返回所有生成的信号，默认为False。
    gen_normal: 是否生成正常信号，默认为True。
    coeff: 脉冲信号的系数
    """
    phase = np.random.randint(0,signal_length)
    t = np.arange(0,2*signal_length/Fs,1/Fs)
    
    mesh_signal = np.sin(20*2*np.pi*rotate_fre*t)[phase:phase+signal_length] # 时间轴设置为了信号预定长度的两倍
    rotate_signal = np.sin(2*np.pi*rotate_fre*t)[phase:phase+signal_length]
    
    feature  = 1*mesh_signal+0.5*rotate_signal
    noise = np.random.normal(loc=0, scale=sigma, size=signal_length)
    """
    loc=0：指定正态分布的均值（平均值），这里设定为0。
    scale=sigma：指定正态分布的标准差，也就是噪声的尺度参数。
    size=signal_length：指定生成随机数的数量，即噪声信号的长度。
    """
    normal = feature+noise

    Afeature = 0.5*mesh_signal +  1*rotate_signal + noise

    if return_all:
        anomaly = 0.1*feature + noise + coeff*generate_impulse(signal_length, rotate_fre, Fs, phase)
        return feature, normal, anomaly
    else:
        if gen_normal:
            return normal
        else:
            anomaly = 0.1*feature + noise + coeff*generate_impulse(signal_length, rotate_fre, Fs, phase)
            return anomaly


def data_prepare(sample_number, anomaly_ratio, test_size, signal_length, rotate_fre, Fs, sigma, coeff_type=[1,1]):

    anomaly_number = int(sample_number*anomaly_ratio)
    test_number = int((sample_number-anomaly_number)*test_size)
    train_number = sample_number-anomaly_number-test_number
    # train data
    train_1d, train_2d = [], []
    print('prepare train data...')
    for i in range(train_number):
        signal = generate_signal(signal_length, rotate_fre, Fs, sigma)
        train_1d.append(signal)
        train_2d.append(reshape_vector(signal, [32,32]))
    train_1d, train_2d = np.array(train_1d)[:,np.newaxis], np.array(train_2d)[:,np.newaxis] # 在第二个维度上增维
    train_label = np.array([0 for i in range(train_number)]) # 正常样本的标签是0
    # test data
    test_1d, test_2d, test_gth_1d, test_gth_2d = [], [], [], []
    print('prepare test data...')
    for i in range(test_number):
        feture, normal, anomaly = generate_signal(signal_length, rotate_fre, Fs, sigma, return_all=True)
        test_1d.append(normal)
        test_2d.append(reshape_vector(normal, [32,32]))
        test_gth_1d.append(feture)
        test_gth_2d.append(reshape_vector(feture, [32,32]))
    test_label = [0 for i in range(test_number)]
    for i in range(anomaly_number):
        coeff = np.linspace(coeff_type[0], coeff_type[1], anomaly_number)[i] # np.linspace()中的三个参数分别表示起始值，终止值，以及在这个区间的等间隔生成的值的数量
        feture, normal, anomaly = generate_signal(signal_length, rotate_fre, Fs, sigma, return_all=True, coeff=coeff)
        test_1d.append(anomaly)
        test_2d.append(reshape_vector(anomaly, [32,32]))
        test_gth_1d.append(feture)
        test_gth_2d.append(reshape_vector(feture, [32,32]))
        test_label.append(1)

    test_1d, test_2d, test_gth_1d, test_gth_2d = np.array(test_1d)[:,np.newaxis], np.array(test_2d)[:,np.newaxis], np.array(test_gth_1d)[:,np.newaxis], np.array(test_gth_2d)[:,np.newaxis]
    test_label = np.array(test_label)
    dataset = {'train_1d':train_1d, 'train_2d':train_2d, 'train_label':train_label, 'test_1d':test_1d, 'test_2d':test_2d,'test_gth_1d':test_gth_1d, 'test_gth_2d':test_gth_2d, 'test_label':test_label}
    return dataset