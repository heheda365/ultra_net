from qnn_param_reader import QNNParamReader
import numpy as np
import os
import sys

# 将数组中元素拼接组合
# 例如 输入 [1, 1, 1] elem_bit = 1, 返回 111
# 返回值是 int 类型 其程度可能超过 64位
def array_to_string(array, elem_bit):
        val = 0	
        #for i in range(len(array)-1, -1, -1):
        for i in range(len(array)):
            tmp = array[i]
            tmp2 = tmp

            if tmp < 0:
                tmp2 = 2**(elem_bit) + tmp 

            tmp2 = int(tmp2)
            tmp3 = tmp2 * 2**(elem_bit*i)
            val = val + tmp3
        return val

# 处理一层的参数
#  处理得到 转化后的w矩阵和 bn的inc，bias
# w矩阵为二维矩阵，row为输出通道数
# 相对于原始w张量，其内存排列顺序转换为 out_ch, row, col, in_ch
# class ParamProcess:
#     def __init__(self, w, inc, bias, w_bit, in_bit, out_bit, l_shift):
#         # self.qnn_read = QNNParamReader(file_name)
#         self.w = w
#         self.inc = inc
#         self.bias = bias
#         self.w_bit = w_bit
#         self.in_bit = in_bit
#         self.out_bit = out_bit
#         self.l_shift = l_shift

#     def conv_process(self):
#         # con_w 是一个4维张量
#         # 将输入通道维度放到最后
#         con_w = self.w
#         con_w.transpose(0, 2, 3, 1)
#         # 处理为二维矩阵
#         con_w = con_w.reshape(con_w.shape[0], -1)

#         # qinc, qbias 当前不需要处理
#         return con_w, self.inc, self.bias
    
#     def linear_process(self, w_bit, in_bit, out_bit, l_shift):
#         # linear_w0 是一个二维矩阵不需要处理
#         linear_w0 = self.qnn_read.read_qlinear_weight(w_bit)
#         linear_bn0_inc, linear_bn0_bias = self.qnn_read.read_qbarch_norm_act_param(w_bit, in_bit, out_bit, l_shift)

#         return linear_w0, linear_bn0_inc, linear_bn0_bias

#     def last_linear_process(self, w_bit):
#         # 全连接层
#         linear_w0 = self.qnn_read.read_qlinear_weight(w_bit=2)
#         return linear_w0

# 处理一层参数
# 这里的一层指的是 将conv 与 bn act 合并为一层
# 将参数整理成满足 硬件设计需求的形式
class QNNLayerMemProcess:
    # 处理 中间层用
    def __init__(self, name, reader, config, w_bit, in_bit, out_bit, l_shift, pe, simd, conv_linear=False):
        
        self.name = name
        self.reader = reader
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.out_bit = out_bit
        self.l_shift = l_shift
        self.pe = pe
        self.simd = simd
        self.config = config[name] 
        self.conv_linear = conv_linear
    
    # 将矩阵整理成所需要的储存样式
    # 转化位 pe * tiles 矩阵
    def w_to_hls_array(self, w):
        # print("w shape :", w.shape)
        assert w.shape[0] % self.pe == 0, 'out_ch mod pe must 0'
        # w 矩阵的宽 其值 为 k * k * in_ch
        h = w.shape[1]
        # res0 size = out_ch, k * k * in_ch // simd + (0 or 1)
        res0 = [[0 for i in range(h // self.simd)] for j in range(w.shape[0])]
        for out_ch in range(w.shape[0]):
            for i in range(h // self.simd):
                arr = w[out_ch][i*self.simd:(i+1)*self.simd]
                res0[out_ch][i] = array_to_string(arr, self.w_bit)
            
        # 处理不够整除的部分
        if h % self.simd != 0:
            print('h mod simd != 0')
            for out_ch in range(w.shape[0]):
                arr = w[out_ch][h // self.simd * self.simd:]
                res0[out_ch].append(array_to_string(arr, self.w_bit))

        # print('res0 = ', len(res0), len(res0[0]))
        # print(np.array(res0))
        
        tiles = len(res0[0]) * (len(res0) // self.pe) 
        self.w_tiles = tiles
        # print('tiles', tiles)
        res = [[0 for i in range(tiles)] for i in range(self.pe)]

        tiles_cnt = 0
        for i in range(len(res0) // self.pe):
            for j in range(len(res0[0])):

                for pe_cnt in range(self.pe):
                    res[pe_cnt][tiles_cnt] = res0[i * self.pe + pe_cnt][j]
                tiles_cnt += 1  
        return res

    # 处理 inc 和 bias
    def inc_bias_to_hls_array(self, inc, bias):
        inc = inc.reshape(-1, self.pe)
        inc = inc.T
        bias = bias.reshape(-1, self.pe)
        bias = bias.T
        self.a_tiles = inc.shape[1]
        
        return inc, bias
    
    # 卷积参数整理
    # 返回的w因为元素可能大于64位 所以用list储存
    # inc, bias 是numpy.array类型
    def conv(self):
        w = self.reader.read_qconv_weight(self.w_bit)
        inc, bias = self.reader.read_qbarch_norm_act_param(w_bit=self.w_bit, in_bit=self.in_bit, out_bit=self.out_bit, l_shift=self.l_shift)
        # w 是二维矩阵形式
        con_w = w.transpose(0, 2, 3, 1)
        # 处理为二维矩阵
        con_w = con_w.reshape(con_w.shape[0], -1)
        # print(w)
        # 先把 w 处理为每个元素位宽都是 simd * w_bit 形式
        con_w = self.w_to_hls_array(con_w)

        inc, bias = self.inc_bias_to_hls_array(inc, bias)

        self.hls_w = con_w
        self.hls_inc = inc
        self.hls_bias = bias

        self.inc_bit_width = self.get_inc_bit_width(inc)
        self.bias_bit_width = self.get_bias_bit_width(bias)
        return con_w, inc, bias
    
    def last_conv(self):
        w = self.reader.read_qconv_weight(self.w_bit)
        # inc, bias = self.reader.read_qbarch_norm_act_param(w_bit=self.w_bit, in_bit=self.in_bit, out_bit=self.out_bit, l_shift=self.l_shift)
        # w 是二维矩阵形式
        con_w = w.transpose(0, 2, 3, 1)
        # 处理为二维矩阵
        con_w = con_w.reshape(con_w.shape[0], -1)
        # print(w)
        # 先把 w 处理为每个元素位宽都是 simd * w_bit 形式
        con_w = self.w_to_hls_array(con_w)

        # inc, bias = self.inc_bias_to_hls_array(inc, bias)

        self.hls_w = con_w
        # self.hls_inc = inc
        # self.hls_bias = bias

        # self.inc_bit_width = self.get_inc_bit_width(inc)
        # self.bias_bit_width = self.get_bias_bit_width(bias)
        return con_w #, inc, bias


    def linear(self):
        w = self.reader.read_qlinear_weight(self.w_bit)
        inc, bias = self.reader.read_qbarch_norm_act_param(w_bit=self.w_bit, in_bit=self.in_bit, out_bit=self.out_bit, l_shift=self.l_shift)

        # w = self.
        # m * n
        # 如果上一层是卷积层 需要调整参数位置
        if(self.conv_linear == True):
            last_conv_shape = self.config["last_layer_shape"]
            w = w.reshape(w.shape[0], last_conv_shape[0], last_conv_shape[1], last_conv_shape[2])
            w = w.transpose(0, 2, 3, 1)
            w = w.reshape(w.shape[0], -1)
        w = self.w_to_hls_array(w)
        inc, bias = self.inc_bias_to_hls_array(inc, bias)

        self.hls_w = w
        self.hls_inc = inc
        self.hls_bias = bias

        self.inc_bit_width = self.get_inc_bit_width(inc)
        self.bias_bit_width = self.get_bias_bit_width(bias)
        return w, inc, bias
    

    # 最后一个全连接层
    def last_linear(self):
        w = self.reader.read_qlinear_weight(self.w_bit)
        w = self.w_to_hls_array(w)
        self.hls_w = w
        return w
    
    def w_to_hls_init_str(self, w) -> str:
        w_mem_type = "const ap_uint<"+str(self.w_bit * self.simd)+">"

        res = '// ' + self.name + '_w\n'
        res += "//PEs = %d, SIMD = %d\n" % (self.pe, self.simd)
        res += '//bit = %d\n' % self.w_bit
        res += w_mem_type
        res += (' ' + self.name + '_w') 
        res += '[%d][%d] = {\n' % (len(w), len(w[0]))

        res += ",\n".join(map(lambda pe:"{\""+("\", \"".join(map(hex, pe)))+"\"}", w))
        res += '};\n'

        return res
    

    # 确定 inc 位宽 
    def get_inc_bit_width(self, inc):
        abs_max = np.abs(inc).max()
        bit_width = len(str(bin(abs_max))) - 2
        return bit_width + 1
    
    # 确定bias的位宽
    # bias 有整数和负数
    # 当前算法得出的还不是最优
    def get_bias_bit_width(self, bias):
        abs_max = np.abs(bias).max()
        bit_width = len(str(bin(abs_max))) - 2
        return bit_width + 1
    
    def inc_to_hls_init_str(self, inc) -> str:
        inc_bit_width = self.inc_bit_width

        w_mem_type = "const ap_int<"+str(inc_bit_width)+">"

        res = '// inc\n'
        res += '// ' + self.name + '_inc\n'
        res += '// w_bit = %d\n' % inc_bit_width
        res += w_mem_type
        res += (' ' + self.name + '_inc') 
        res += '[%d][%d] = {\n' % (len(inc), len(inc[0]))

        res += ",\n".join(map(lambda pe:"{\""+("\", \"".join(map(hex, pe)))+"\"}", inc))
        res += '};\n'

        return res  
    
    def bias_to_hls_init_str(self, bias) -> str:
        bias_bit_width = self.bias_bit_width

        w_mem_type = "const ap_int<"+str(bias_bit_width)+">"
        res = '// bias\n'
        res += '// ' + self.name + '_bias\n'
        res += '// w_bit = %d\n' % bias_bit_width
        res += w_mem_type
        res += (' ' + self.name + '_bias') 
        res += '[%d][%d] = {\n' % (len(bias), len(bias[0]))

        res += ",\n".join(map(lambda pe:"{\""+("\", \"".join(map(hex, pe)))+"\"}", bias))
        res += '};\n'

        return res

    def layer_param_to_init_str(self, w, inc, bias) -> str:
        res = self.w_to_hls_init_str(w)
        res += self.inc_to_hls_init_str(inc)
        res += self.bias_to_hls_init_str(bias)

        return res
    
    def last_layer_param_to_init_str(self, w) -> str:
        res = self.w_to_hls_init_str(w)
       
        return res

    def add_a_config_str(self, config_name, value) -> str:
        res = '#define %s_%s %d \n' % (self.name.upper(), config_name.upper(), value)
        return res

    def conv_config_str(self) -> str:
        res = '// ' + self.name + '\n'
        res += self.add_a_config_str('K', self.config['k'])
        res += self.add_a_config_str('S', self.config['s'])
        res += self.add_a_config_str('P', self.config['p'])
        res += self.add_a_config_str('IFM_CH', self.config['in_shape'][0])
        res += self.add_a_config_str('IFM_ROW', self.config['in_shape'][1])
        res += self.add_a_config_str('IFM_COL', self.config['in_shape'][2])

        res += self.add_a_config_str('OFM_CH', self.config['out_shape'][0])
        res += self.add_a_config_str('OFM_ROW', self.config['out_shape'][1])
        res += self.add_a_config_str('OFM_COL', self.config['out_shape'][2])

        res += self.add_a_config_str('SIMD', self.simd)
        res += self.add_a_config_str('PE', self.pe)

        res += self.add_a_config_str('IN_BIT', self.in_bit)
        res += self.add_a_config_str('OUT_BIT', self.out_bit)
        res += self.add_a_config_str('W_BIT', self.w_bit)
        res += self.add_a_config_str('INC_BIT', self.inc_bit_width)
        res += self.add_a_config_str('BIAS_BIT', self.bias_bit_width)

        res += self.add_a_config_str('W_TILES', self.w_tiles)
        res += self.add_a_config_str('A_TILES', self.a_tiles)
        res += self.add_a_config_str('L_SHIFT', self.l_shift)

        res += '\n'

        return res

    def last_conv_config_str(self) -> str:
        res = '// ' + self.name + '\n'
        res += self.add_a_config_str('K', self.config['k'])
        res += self.add_a_config_str('S', self.config['s'])
        res += self.add_a_config_str('P', self.config['p'])
        res += self.add_a_config_str('IFM_CH', self.config['in_shape'][0])
        res += self.add_a_config_str('IFM_ROW', self.config['in_shape'][1])
        res += self.add_a_config_str('IFM_COL', self.config['in_shape'][2])

        res += self.add_a_config_str('OFM_CH', self.config['out_shape'][0])
        res += self.add_a_config_str('OFM_ROW', self.config['out_shape'][1])
        res += self.add_a_config_str('OFM_COL', self.config['out_shape'][2])

        res += self.add_a_config_str('SIMD', self.simd)
        res += self.add_a_config_str('PE', self.pe)

        res += self.add_a_config_str('IN_BIT', self.in_bit)
        # res += self.add_a_config_str('OUT_BIT', self.out_bit)
        res += self.add_a_config_str('W_BIT', self.w_bit)
        # res += self.add_a_config_str('INC_BIT', self.inc_bit_width)
        # res += self.add_a_config_str('BIAS_BIT', self.bias_bit_width)

        res += self.add_a_config_str('W_TILES', self.w_tiles)
        res += self.add_a_config_str('L_SHIFT', self.l_shift)
        # res += self.add_a_config_str('A_TILES', self.a_tiles)

        res += '\n'

        return res

    def linear_config_str(self) -> str:
        res = '// ' + self.name + '\n'

        res += self.add_a_config_str('IN_LEN', self.config['in_len'])
        res += self.add_a_config_str('OUT_LEN', self.config['out_len'])

        res += self.add_a_config_str('SIMD', self.simd)
        res += self.add_a_config_str('PE', self.pe)

        res += self.add_a_config_str('IN_BIT', self.in_bit)
        res += self.add_a_config_str('OUT_BIT', self.out_bit)
        res += self.add_a_config_str('W_BIT', self.w_bit)
        res += self.add_a_config_str('INC_BIT', self.inc_bit_width)
        res += self.add_a_config_str('BIAS_BIT', self.bias_bit_width)

        res += self.add_a_config_str('W_TILES', self.w_tiles)
        res += self.add_a_config_str('A_TILES', self.a_tiles)
        res += self.add_a_config_str('L_SHIFT', self.l_shift)

        res += '\n'
        return res

    def last_linear_config_str(self) -> str:
        res = '// ' + self.name + '\n'

        res += self.add_a_config_str('IN_LEN', self.config['in_len'])
        res += self.add_a_config_str('OUT_LEN', self.config['out_len'])

        res += self.add_a_config_str('SIMD', self.simd)
        res += self.add_a_config_str('PE', self.pe)

        res += self.add_a_config_str('IN_BIT', self.in_bit)
        # res += self.add_a_config_str('OUT_BIT', self.out_bit)
        res += self.add_a_config_str('W_BIT', self.w_bit)

        res += self.add_a_config_str('L_SHIFT', self.l_shift)

        return res

        

if __name__ == "__main__":
    import json
    config_file = open('config.json', 'r', encoding='utf-8')
    config = json.load(config_file)
    # qnn_men_pro = QNNMemProcess('miniConvNet.npz')
    reader = QNNParamReader('miniConvNet.npz')
    processer = QNNLayerMemProcess('conv_0', reader, config, w_bit=2, in_bit=8, out_bit=4, l_shift=0, pe=4, simd=9)


    w, inc, bias = processer.conv()
    con_str = processer.conv_config_str()
    print(con_str)
    # # w_str = processer.w_to_hls_init_str(w)
    # conv_str = processer.layer_param_to_init_str(w, inc, bias)
    # print(conv_str)

    # processer1 = QNNLayerMemProcess('conv_1', reader, w_bit=2, in_bit=4, out_bit=4, l_shift=0, pe=16, simd=32)
    # w, inc, bias = processer1.conv()
    # # w_str = processer.w_to_hls_init_str(w)
    # conv_str = processer1.layer_param_to_init_str(w, inc, bias)
    # print(conv_str)

    # print(inc)

    # w, inc, bias = qnn_men_pro.conv(2, 4, 4, l_shift=0, pe=4, simd=32)
    # print(inc)
    # w, inc, bias = qnn_men_pro.conv(2, 4, 4, l_shift=0, pe=4, simd=32)
    # print(inc)

    # w, inc, bias = qnn_men_pro.conv(2, 4, 4, l_shift=0, pe=4, simd=32)
    # print(inc)

    # w, inc, bias = qnn_men_pro.linear(2, 4, 4, l_shift=0, pe=4, simd=32)
    # print(inc)
    # w_str = qnn_men_pro.w_to_hls_init_str('conv_0_w', w, 2, 4, 9)
    # print(np.array(w))
    # print(inc)
    # print(bias)
    # print(w_str)

    # a = -2
    # print(bin(a))
    # print(len(str(bin(a))))

    

    # # 维度顺序变换
    # a = [[[1, 2, 3], 
    #     [4, 5, 6],
    #     [7, 8, 9]],

    #     [[1, 2, 3], 
    #     [4, 5, 6],
    #     [7, 8, 9]],

    #     [[1, 2, 3], 
    #     [4, 5, 6],
    #     [7, 8, 9]],

    #     [[1, 2, 3], 
    #     [4, 5, 6],
    #     [7, 8, 9]]]
    # a = np.array(a)
    # print(a.shape[1])
    # b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # c = b[1:]
    # print(c)
    # print(a)
    # print(a.transpose(1, 2, 0))

    # print(a.reshape(4, -1))

    # def ArrayToString(array, precision, precFract=0, debug=False):
    #     val = 0	
    #     #for i in range(len(array)-1, -1, -1):
    #     for i in range(len(array)):
    #         tmp = array[i]
    #         tmp2 = tmp

    #         if tmp < 0:
    #             tmp2 = 2**(precision) + tmp 

    #         tmp2 = int(tmp2)
    #         tmp3 = tmp2 * 2**(precision*i)
    #         val = val + tmp3

    #     return val
    
    # a = ArrayToString([5, 6, 7, 8, 9, 1, 2, 2, 1, 2, 3, 4, 4], 40)
    # print(hex(a))

    # a = [1, 2]
    # b = a[0]   # 1
    # # b = 6
    # a[0] = 10
    # print(a[0], b)

    # a = [[1, 2], [3, 4]]
    # b = a[0]
    # a[0][0] = 10

    # a = 1
    # print(id(a))
    # a = a + 1
    # print(id(a))
    # print(b)

    # a = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # print(len(str(a)))

    # a = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]
    # b = ["%s " % str(x) for x in a]
    # c = [y for y in a]
    # print(b)
    # print(c)

    # def square(x):
    #     return x + 1
    # d = map(lambda pe: map(square, pe), a)  
    # # print(list())

    # e = '\n{'.join(map(lambda pe: ','.join(map(hex, pe)), a))
    # print(e)

    # f = ",\n".join(map(lambda pe:"{"+(", ".join(map(hex, pe)))+"}", a))
    # print(f)

    # print('a'.join())

    # a = []
    # a[0] = 1