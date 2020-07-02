import numpy as np
import quantization


# 读出参数
# 然后根据量化规则将其处理量化为指定位宽
# 使用int32类型储存
class QNNParamReader:
    def __init__(self, paramFile):
        self.param_dict = np.load(paramFile)
        self.current_param_cnt = 0
    
    def get_last(self):
        ret = self.param_dict["arr_" + str(self.current_param_cnt)]
        # self.current_param_cnt += 1
        return ret

    def __get_current(self):
        ret = self.param_dict["arr_" + str(self.current_param_cnt)]
        self.current_param_cnt += 1
        return ret

    def read_conv_raw(self):
        w = self.__get_current()
        return w

    def read_linear_raw(self):
        w = self.__get_current()
        return w

    def read_batch_norm_raw(self):
        gamma = self.__get_current()
        beta = self.__get_current()
        mean = self.__get_current()
        var = self.__get_current()
        eps = self.__get_current()
        return (gamma, beta, mean, var, eps)

    # 读量化后卷积层参数
    # 量化后用 int32 表示每个数据
    # 默认将卷积层位宽参数量化到2个bit
    # 符号位占用一个bit
    def read_qconv_weight(self, w_bit=2):
        w = self.read_conv_raw()
        # 执行 w 量化
        qw = quantization.weight_quantize_int(w, w_bit)
        return qw

    # 读量化后的全连接层的参数
    # 量化后用 int32 表示每个数据 实际有效的只有 w_bit
    # 符号位占用一个 bit
    def read_qlinear_weight(self, w_bit=2):
        w = self.read_linear_raw()
        qw = quantization.weight_quantize_int(w, w_bit)
        return qw

    # 读取量化后的 bn 层参数
    # 将bn层和act层放在一起处理，量化后其可以用一个等差数列表示
    # 其中inc表示公差， bias表示初始值
    def read_qbarch_norm_act_param(self, w_bit=2, in_bit=4, out_bit=4, l_shift=4):
        gamma, beta, mean, var, eps = self.read_batch_norm_raw()
        qinc, qbias = quantization.bn_act_quantize_int(gamma, beta, mean, var, eps, w_bit=w_bit, in_bit=in_bit, out_bit=out_bit, l_shift=l_shift)
        return qinc, qbias
    

if __name__ == "__main__":
    import os
    import sys
    target_dir_int_param = 'param/int32/'
    if not os.path.exists(target_dir_int_param):
        os.makedirs(target_dir_int_param)

    # 测试读出完整的参数，量化成int32后保存
    qnn_read = QNNParamReader('miniConvNet.npz')
    
    # 网络有4个卷积层，两个全连接层
    # 卷积层和中间的 bn层
    for i in range(4):
        con_w = qnn_read.read_qconv_weight(w_bit=2)
        if i == 0:
            in_bit = 8
            print(con_w)
        else:
            in_bit = 4
        qinc, qbias = qnn_read.read_qbarch_norm_act_param(w_bit=2, in_bit=in_bit, out_bit=4, l_shift=0)


        con_w.tofile(target_dir_int_param + 'conv_' + str(i) + '_w.bin')
        qinc.tofile(target_dir_int_param + 'conv_' + str(i) + '_bn_inc.bin')
        qbias.tofile(target_dir_int_param + 'conv_' + str(i) + '_bn_bias.bin')
    
    # 全连接层
    linear_w0 = qnn_read.read_qlinear_weight(w_bit=2)
    linear_bn0_inc, linear_bn0_bias = qnn_read.read_qbarch_norm_act_param(w_bit=2, in_bit=4, out_bit=4, l_shift=0)

    linear_w0.tofile(target_dir_int_param + 'linear_0_w' + '.bin')
    linear_bn0_inc.tofile(target_dir_int_param + 'linear_0_bn_inc' + '.bin')
    linear_bn0_bias.tofile(target_dir_int_param + 'linear_0_bn_bias' + '.bin')

    linear_w1 = qnn_read.read_qlinear_weight(w_bit=2)
    linear_w1.tofile(target_dir_int_param + 'linear_1_w' + '.bin')
    print('generate parameter succeed')

    


