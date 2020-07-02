import numpy as np 


# 使用derefa量化 将浮点数量化表示
def uniform_quantize(input, bit=2):
    n = float(2 ** bit - 1)
    out = np.round(input * n) / n
    
    return out

# 量化w
# 量化到（-1， 1）之间的特定数值（-1， 0， 1）
def weight_quantize_float(input, bit=2):
    weight = np.tanh(input)
    weight = weight / np.max(np.abs(weight)) 
    # 这里是因为量化的数值限制在（-1，1）
    weight_q = uniform_quantize(weight, bit=bit-1)

    return weight_q

# 量化w
# 量化到 （-2**（bit-1）, + 2**(bit-1))
# 当bit=2时表示的是三元量化，即只能取到值-1， 0， 1
def weight_quantize_int(input, bit=2):
    weight = np.tanh(input)
    weight = weight / np.max(np.abs(weight))
    weight_q = weight * (2**(bit-1) - 1)
    weight_q = np.round(weight_q)
    # print(weight_q)
    weight_q = weight_q.astype(np.int32)
    return weight_q

# 这里计算出bn层等价的w和b
def bn_act_w_bias_float(gamma, beta, mean, var, eps):
    w = gamma / (np.sqrt(var) + eps)
    b = beta - (mean / (np.sqrt(var) + eps) * gamma)
    return w, b

# 将bn层与act层放在一起计算得到一个等差数列
# 等差数列的下标表示输入数据激活量化后的输出值
# 例如等差数为[3, 7, 11, 15, 19, 23, 27]
# 输入17应该返回4， 输入3返回0
# 注意特征数据是无符号的，权值参数是有符号的
# w 应该不为0
# l_shift是出于精度考虑将结果乘以一定的倍数保存
# 弃用
# def bn_act_quantize_int(gamma, beta, mean, var, eps, w_bit=2, in_bit=4, out_bit=4, l_shift=4):
#     # 先计算出等价的w和b
#     w, b = bn_act_w_bias_float(gamma, beta, mean, var, eps)
#     inc_f = 1 / w
#     bias_f = b / w
#     inc = inc_f * (2 ** (w_bit - 1) - 1) * (2 ** in_bit - 1) * (2 ** l_shift) / (2 ** out_bit - 1)
#     bias = bias_f * (2 ** (w_bit - 1) - 1) * (2 ** in_bit - 1) * (2 ** l_shift) + inc / 2
#     inc_q = (inc + 0.5).astype(np.int32)
#     bias_q = (bias + 0.5).astype(np.int32)
#     print(inc_q)
#     return inc_q, bias_q

def bn_act_quantize_int(gamma, beta, mean, var, eps, w_bit=2, in_bit=4, out_bit=4, l_shift=4):
    # 先计算出等价的w和b
    w, b = bn_act_w_bias_float(gamma, beta, mean, var, eps)

    n = 2**(w_bit - 1 + in_bit + l_shift) / ((2 ** (w_bit-1) - 1) * (2 ** in_bit - 1))
    inc_q = (2 ** out_bit - 1) * n * w 
    bias_q = (2 ** (w_bit-1) - 1) * (2 ** in_bit - 1) * (2 ** out_bit - 1) * n * b
    inc_q = np.round(inc_q).astype(np.int32)
    bias_q = np.round(bias_q).astype(np.int32)
    print('inc_q: ', inc_q)
    print('bias_q: ', bias_q)
    return inc_q, bias_q




if __name__ == "__main__":
    a = np.array([-0.6, 0.1, -0.2, 0.5, 0.3, 0.8, -3.9])
    # print(a.astype(np.int32))


    # print(weight_quantize_float(a, bit=3))
    print(weight_quantize_int(a, bit=4))
