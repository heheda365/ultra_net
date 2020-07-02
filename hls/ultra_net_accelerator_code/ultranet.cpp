// #define DEBUG

#ifdef DEBUG
#include <iostream>
#include <fstream>
using namespace std;

#endif


#include <stdint.h>
#include <ap_int.h>
#include <hls_video.h>
#include "stream_tools.h"
#include "function.h"
#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"
#include "config.h"
#include "param.h"
#include "conv2d.h"
#include "pool2d.h"
#include "bn_qrelu2d.h"

#define IN_IMAGE_WIDTH  640
#define IN_IMAGE_HEIGHT 360

#define RESIZE_IMAGE_WIDTH 320
#define RESIZE_IMAGE_HEIGHT 160

void stream_to_mat (hls::stream<ap_uint<24>>&in, 
		 hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> & raw_img) {
    
	for (int i=0; i<IN_IMAGE_HEIGHT; i++) {
		for (int j=0; j<IN_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            hls::Scalar<3, ap_uint<8>> pix;
            ap_uint<24> in_data = in.read();
            for (unsigned int p=0; p < 3; p ++) {
                
                pix.val[p] = in_data(8*p+7, 8*p);
            }
			raw_img << pix;
		}	
	}

}

void mat_to_stream (hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> & resize_img,
                    hls::stream<ap_uint<24>> & out ) {
    
	for (int i=0; i<RESIZE_IMAGE_HEIGHT; i++) {
		for (int j=0; j<RESIZE_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            hls::Scalar<3, ap_uint<8>> pix;
            resize_img >> pix;
            ap_uint<24> out_data;
            for (unsigned int p=0; p < 3; p ++) {
                out_data(8*p+7, 8*p) = pix.val[p];
            }
            out.write(out_data);
		}	
	}

}

void resize(hls::stream<ap_uint<24>> &in, hls::stream<ap_uint<24>> & out) {
#pragma HLS dataflow
    hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> raw_img;
#pragma HLS STREAM variable=raw_img depth=128 dim=1
    hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> resize_img;
#pragma HLS STREAM variable=resize_img depth=128 dim=1
    stream_to_mat(in, raw_img);
    // hls::Resize(raw_img, resize_img, HLS_INTER_LINEAR);
    hls::Resize_opr_linear(raw_img, resize_img);
    mat_to_stream(resize_img, out);
}

void resize_batch(hls::stream<ap_uint<24>> &in, hls::stream<ap_uint<24>> & out, unsigned int reps) {
    for (unsigned int rep=0; rep < reps; rep ++) {
        resize(in, out);
    }
}

void do_compute(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps) {
#pragma HLS DATAFLOW

    const unsigned int num_per_rep = 360 * 640 * 3 * 8 / 64;

    hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable=in_stream_extract depth=16 dim=1
	ExtractPixels<64, num_per_rep> (in, in_stream_extract, reps);

    hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
#pragma HLS STREAM variable=in_stream0 depth=16 dim=1
    StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract, in_stream0, reps);

	hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> > in_stream1("in_stream1");
#pragma HLS STREAM variable=in_stream1 depth=16 dim=1

	StreamingDataWidthConverter_Batch<64 * 3, CONV_0_IN_BIT * CONV_0_IFM_CH, num_per_rep / 3> (in_stream0, in_stream1, reps);
#ifdef DEBUG
    cout << "in_stream1 size " << in_stream1.size() << endl;

#endif

    hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH> > in_stream2("in_stream2");
#pragma HLS STREAM variable=in_stream2 depth=16 dim=1
    resize_batch(in_stream1, in_stream2, reps);
#ifdef DEBUG
    cout << "in_stream2 size " << in_stream2.size() << endl;
    // hls::stream<ap_uint<8>> res("res");
    // StreamingDataWidthConverter_Batch<CONV_0_IN_BIT * CONV_0_IFM_CH, 8, 320*3>(in_stream, res, 1);
    // int data[3][320][3];
    // for (int n=0; n < 3; n ++)
    //     for (int i=0; i < 320; i ++) {
    //         for (int j=0; j < 3; j ++)
    //             data[n][i][j] = res.read();
    //     }

    // for (int n=0; n < 3; n ++)
    //     for (int i=0; i < 3; i ++) {
    //         for (int j=0; j < 3; j ++)
    //             cout << data[n][i][j] << " ";
    //     }
    // return;
    // 输入数据没问�??
#endif

    hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_OFM_CH>>  conv_0_out("conv_0_out");
#pragma HLS STREAM variable=conv_0_out depth=128 dim=1
    conv3x3_bn_act< 
                    CONV_0_IFM_ROW,
                    CONV_0_IFM_COL,
                    CONV_0_IFM_CH,
                    CONV_0_IN_BIT,

                    CONV_0_OFM_CH,
                    CONV_0_OUT_BIT,

                    CONV_0_W_BIT,
                    32,                     
                    CONV_0_INC_BIT,
                    CONV_0_BIAS_BIT,

                    CONV_0_SIMD,
                    CONV_0_PE,
                    CONV_0_L_SHIFT>(
                in_stream2,
                conv_0_w,
                conv_0_inc,
                conv_0_bias,
                conv_0_out,
                reps );
#ifdef DEBUG
    cout << "conv_0_out size " << conv_0_out.size() << endl;
    // hls::stream<ap_uint<4>> res("res");
    // StreamingDataWidthConverter_Batch<CONV_0_OUT_BIT * CONV_0_OFM_CH, 4, 1>(conv_0_out, res, 1);
    // for (int i=0; i < 16; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
    hls::stream<ap_uint<CONV_0_OUT_BIT*CONV_0_OFM_CH>> pool_0_out("pool_0_out");
#pragma HLS STREAM variable=pool_0_out depth=128 dim=1
    max_pool2d< 2,
                CONV_0_OFM_ROW,
                CONV_0_OFM_COL,
                CONV_0_OFM_CH,
                CONV_0_OUT_BIT>(
                    conv_0_out,
                    pool_0_out,
                    reps);
#ifdef DEBUG
    cout << "pool_0_out size " << pool_0_out.size() << endl;
#endif

    hls::stream<ap_uint<CONV_1_OUT_BIT * CONV_1_OFM_CH>>  conv_1_out("conv_1_out");
#pragma HLS STREAM variable=conv_1_out depth=128 dim=1
    conv3x3_bn_act< 
                    CONV_1_IFM_ROW,
                    CONV_1_IFM_COL,
                    CONV_1_IFM_CH,
                    CONV_1_IN_BIT,

                    CONV_1_OFM_CH,
                    CONV_1_OUT_BIT,

                    CONV_1_W_BIT,
                    32,                     
                    CONV_1_INC_BIT,
                    CONV_1_BIAS_BIT,

                    CONV_1_SIMD,
                    CONV_1_PE,
                    CONV_1_L_SHIFT>(
                pool_0_out,
                conv_1_w,
                conv_1_inc,
                conv_1_bias,
                conv_1_out,
                reps );
#ifdef DEBUG
    cout << "conv_1_out size " << conv_1_out.size() << endl;
#endif
    hls::stream<ap_uint<CONV_1_OUT_BIT*CONV_1_OFM_CH>> pool_1_out("pool_out");
#pragma HLS STREAM variable=pool_1_out depth=128 dim=1
    max_pool2d< 2,
                CONV_1_OFM_ROW,
                CONV_1_OFM_COL,
                CONV_1_OFM_CH,
                CONV_1_OUT_BIT>(
                    conv_1_out,
                    pool_1_out,
                    reps);
#ifdef DEBUG
    cout << "pool_1_out size " << pool_1_out.size() << endl;
#endif

    hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OFM_CH>>  conv_2_out("conv_2_out");
#pragma HLS STREAM variable=conv_2_out depth=128 dim=1
    conv3x3_bn_act< 
                    CONV_2_IFM_ROW,
                    CONV_2_IFM_COL,
                    CONV_2_IFM_CH,
                    CONV_2_IN_BIT,

                    CONV_2_OFM_CH,
                    CONV_2_OUT_BIT,

                    CONV_2_W_BIT,
                    32,                     
                    CONV_2_INC_BIT,
                    CONV_2_BIAS_BIT,

                    CONV_2_SIMD,
                    CONV_2_PE,
                    CONV_2_L_SHIFT>(
                pool_1_out,
                conv_2_w,
                conv_2_inc,
                conv_2_bias,
                conv_2_out,
                reps );
#ifdef DEBUG
    cout << "conv_2_out size " << conv_2_out.size() << endl;
#endif
    hls::stream<ap_uint<CONV_2_OUT_BIT*CONV_2_OFM_CH>> pool_2_out("pool_out");
#pragma HLS STREAM variable=pool_2_out depth=128 dim=1
    max_pool2d< 2,
                CONV_2_OFM_ROW,
                CONV_2_OFM_COL,
                CONV_2_OFM_CH,
                CONV_2_OUT_BIT>(
                    conv_2_out,
                    pool_2_out,
                    reps);
#ifdef DEBUG
    cout << "pool_2_out size " << pool_2_out.size() << endl;
#endif

    hls::stream<ap_uint<CONV_3_OUT_BIT * CONV_3_OFM_CH>>  conv_3_out("conv_3_out");
#pragma HLS STREAM variable=conv_3_out depth=128 dim=1
    conv3x3_bn_act< 
                    CONV_3_IFM_ROW,
                    CONV_3_IFM_COL,
                    CONV_3_IFM_CH,
                    CONV_3_IN_BIT,

                    CONV_3_OFM_CH,
                    CONV_3_OUT_BIT,

                    CONV_3_W_BIT,
                    32,                     
                    CONV_3_INC_BIT,
                    CONV_3_BIAS_BIT,

                    CONV_3_SIMD,
                    CONV_3_PE,
                    CONV_3_L_SHIFT>(
                pool_2_out,
                conv_3_w,
                conv_3_inc,
                conv_3_bias,
                conv_3_out,
                reps );
#ifdef DEBUG
    cout << "conv_3_out size " << conv_3_out.size() << endl;
#endif
    hls::stream<ap_uint<CONV_3_OUT_BIT*CONV_3_OFM_CH>> pool_3_out("pool_3_out");
#pragma HLS STREAM variable=pool_3_out depth=128 dim=1
    max_pool2d< 2,
                CONV_3_OFM_ROW,
                CONV_3_OFM_COL,
                CONV_3_OFM_CH,
                CONV_3_OUT_BIT>(
                    conv_3_out,
                    pool_3_out,
                    reps);
#ifdef DEBUG
    cout << "pool_3_out size " << pool_3_out.size() << endl;
#endif

    hls::stream<ap_uint<CONV_4_OUT_BIT * CONV_4_OFM_CH>>  conv_4_out("conv_4_out");
#pragma HLS STREAM variable=conv_4_out depth=128 dim=1
    conv3x3_bn_act< 
                    CONV_4_IFM_ROW,
                    CONV_4_IFM_COL,
                    CONV_4_IFM_CH,
                    CONV_4_IN_BIT,

                    CONV_4_OFM_CH,
                    CONV_4_OUT_BIT,

                    CONV_4_W_BIT,
                    32,                     
                    CONV_4_INC_BIT,
                    CONV_4_BIAS_BIT,

                    CONV_4_SIMD,
                    CONV_4_PE,
                    CONV_4_L_SHIFT>(
                pool_3_out,
                conv_4_w,
                conv_4_inc,
                conv_4_bias,
                conv_4_out,
                reps );
#ifdef DEBUG
    cout << "conv_4_out size " << conv_4_out.size() << endl;
#endif

    hls::stream<ap_uint<CONV_5_OUT_BIT * CONV_5_OFM_CH>>  conv_5_out("conv_5_out");
#pragma HLS STREAM variable=conv_5_out depth=128 dim=1
    conv3x3_bn_act< 
                    CONV_5_IFM_ROW,
                    CONV_5_IFM_COL,
                    CONV_5_IFM_CH,
                    CONV_5_IN_BIT,

                    CONV_5_OFM_CH,
                    CONV_5_OUT_BIT,

                    CONV_5_W_BIT,
                    32,                     
                    CONV_5_INC_BIT,
                    CONV_5_BIAS_BIT,

                    CONV_5_SIMD,
                    CONV_5_PE,
                    CONV_5_L_SHIFT>(
                conv_4_out,
                conv_5_w,
                conv_5_inc,
                conv_5_bias,
                conv_5_out,
                reps );
#ifdef DEBUG
    cout << "conv_5_out size " << conv_5_out.size() << endl;
#endif

    hls::stream<ap_uint<CONV_6_OUT_BIT * CONV_6_OFM_CH>>  conv_6_out("conv_6_out");
#pragma HLS STREAM variable=conv_6_out depth=128 dim=1
    conv3x3_bn_act< 
                    CONV_6_IFM_ROW,
                    CONV_6_IFM_COL,
                    CONV_6_IFM_CH,
                    CONV_6_IN_BIT,

                    CONV_6_OFM_CH,
                    CONV_6_OUT_BIT,

                    CONV_6_W_BIT,
                    32,                     
                    CONV_6_INC_BIT,
                    CONV_6_BIAS_BIT,

                    CONV_6_SIMD,
                    CONV_6_PE,
                    CONV_6_L_SHIFT>(
                conv_5_out,
                conv_6_w,
                conv_6_inc,
                conv_6_bias,
                conv_6_out,
                reps );
#ifdef DEBUG
    cout << "conv_6_out size " << conv_6_out.size() << endl;
#endif

    hls::stream<ap_uint<CONV_7_OUT_BIT * CONV_7_OFM_CH>>  conv_7_out("conv_7_out");
#pragma HLS STREAM variable=conv_7_out depth=128 dim=1
    conv3x3_bn_act< 
                    CONV_7_IFM_ROW,
                    CONV_7_IFM_COL,
                    CONV_7_IFM_CH,
                    CONV_7_IN_BIT,

                    CONV_7_OFM_CH,
                    CONV_7_OUT_BIT,

                    CONV_7_W_BIT,
                    32,                     
                    CONV_7_INC_BIT,
                    CONV_7_BIAS_BIT,

                    CONV_7_SIMD,
                    CONV_7_PE,
                    CONV_7_L_SHIFT>(
                conv_6_out,
                conv_7_w,
                conv_7_inc,
                conv_7_bias,
                conv_7_out,
                reps );
#ifdef DEBUG
    cout << "conv_7_out size " << conv_7_out.size() << endl;
    // hls::stream<ap_uint<4>> res("res");
    // StreamingDataWidthConverter_Batch<CONV_7_OUT_BIT * CONV_7_OFM_CH, 4, 1>(conv_7_out, res, 1);
    // for (int i=0; i < 64; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif
    hls::stream<ap_uint<CONV_8_IN_BIT * CONV_8_SIMD>>  conv_8_in("conv_8_in");
#pragma HLS STREAM variable=conv_8_in depth=64 dim=1
    StreamingDataWidthConverter_Batch<CONV_7_OFM_CH*CONV_7_OUT_BIT, 
            CONV_8_IN_BIT * CONV_8_SIMD, 
            CONV_7_OFM_ROW*CONV_7_OFM_COL>(conv_7_out, conv_8_in, reps);
    hls::stream<ap_uint<32 * CONV_8_PE>>  conv_8_out("conv_8_out");
#pragma HLS STREAM variable=conv_8_out depth=64 dim=1
    conv1x1 < 
                    CONV_8_IFM_ROW,
                    CONV_8_IFM_COL,
                    CONV_8_IFM_CH,
                    CONV_8_IN_BIT,
                    CONV_8_OFM_CH,

                    CONV_8_W_BIT,
                    32,                     

                    CONV_8_SIMD,
                    CONV_8_PE>(
                conv_8_in,
                conv_8_w,
                conv_8_out,
                reps );
#ifdef DEBUG
    cout << "conv_8_out size " << conv_8_out.size() << endl;
    // hls::stream<ap_uint<32>> res("res");
    // StreamingDataWidthConverter_Batch<32 * CONV_8_PE, 32, 18>(conv_8_out, res, 1);
    // for (int i=0; i < 36; i ++) {
    //     ap_int<32> a =  res.read();
    //     cout << a << " ";
    // }
    // cout << endl;
    // return;
#endif
    AddLast<CONV_8_OFM_ROW*CONV_8_OFM_COL*CONV_8_OFM_CH/2>(conv_8_out, out, reps);

}
void ultra_net(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps) {

#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=reps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable = conv_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_1_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_2_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_3_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_4_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_5_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_5_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_6_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_6_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_7_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_7_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_8_w complete dim = 1

    do_compute(in, out, reps);

}

#ifdef DEBUG

void load_data(const char *path, char *ptr, unsigned int size)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f)
    {
        std::cout << "no such file,please check the file name!/n";
        exit(0);
    }
    f.read(ptr, size);
    f.close();
}

void write_data(const char *path, char *ptr, unsigned int size)
{
    std::ofstream f(path, std::ios::out | std::ios::binary);
    if (!f)
    {
        std::cout << "write no such file,please check the file name!/n";
        exit(0);
    }
    f.write(ptr, size);
    f.close();
}

int main(int argc, char const *argv[])
{
    uint8_t img[360][640][3];
    load_data("data/boat6_0.bin", (char *) img, sizeof(img));

    uint8_t * data = (uint8_t *) img;
    const int data_points_per_line = 8;        // ch * 10
    const int nums_line_pre_img = 360 * 640 * 3 / 8;

    hls::stream<my_ap_axis> input_stream("input stream");
	for (unsigned int i = 0; i < nums_line_pre_img; i++) {
		my_ap_axis temp;
		for (unsigned int j = 0; j < data_points_per_line; j++) {
			temp.data( 8*(j+1)-1, 8*j ) = data[i * data_points_per_line + j];
		}
		input_stream.write(temp);
	}
    cout << "start ..... " << endl;
    hls::stream<my_ap_axis> output_stream("output stream");
    ultra_net(input_stream, output_stream, 1);

    cout << "output size :" << output_stream.size() << endl;
    

    // write_data("data/boat6_0_res.bin", (char *) res, sizeof(res));
    
    // int res1[10][20][5] = {0};
    // for (int i=0; i < 10; i ++) {
    //     for (int j=0; j < 20; j ++) {
    //         for (int k=0; k < 5; k ++) {
                
    //             int temp = 0;
    //             for (int w=0; w < 6; w ++) {
    //                 temp += res[i][j][w * 6 + k];
    //             }
    //             res1[i][j][k] = temp;
    //         }
    //     }
    // }
    
    return 0;
}

#endif 
