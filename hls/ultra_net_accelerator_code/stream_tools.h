#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
using namespace std;

// axi data
struct my_ap_axis {
    ap_uint<64> data;
    ap_uint<1> last;
    ap_uint<8> keep;
};

template <unsigned NumLines>
void AddLast(stream<ap_uint<64>> &in, stream<my_ap_axis> &out,
             const unsigned reps = 1) {
    my_ap_axis temp;
    temp.keep = 0xff;

    for (unsigned i = 0; i < reps * NumLines - 1; i++) {
        temp.data = in.read();
        temp.last = 0;
        out.write(temp);
    }

    temp.data = in.read();
    temp.last = 1;
    out.write(temp);
}

template <unsigned LineWidth, unsigned NumLines>
void Mem2Stream(ap_uint<LineWidth> *in, stream<ap_uint<LineWidth>> &out,
                const unsigned reps = 1) {
    for (unsigned i = 0; i < reps * NumLines; i++) {
        out.write(in[i]);
    }
}

template <unsigned LineWidth, unsigned NumLines>
void Stream2Mem(stream<ap_uint<LineWidth>> &in, ap_uint<LineWidth> *out,
                const unsigned reps = 1) {
    for (unsigned i = 0; i < reps * NumLines; i++) {
        out[i] = in.read();
    }
}

template <unsigned StreamW, unsigned NumLines>
void StreamCopy(stream<ap_uint<StreamW>> &in, stream<ap_uint<StreamW>> &out,
                const unsigned reps = 1) {
    ap_uint<StreamW> temp;

    for (unsigned rep = 0; rep < reps * NumLines; rep++) {
        temp = in.read();
        out.write(temp);
    }
}

template <unsigned OutStreamW, unsigned NumLines>
void ExtractPixels(stream<my_ap_axis> &in, stream<ap_uint<OutStreamW>> &out,
                   const unsigned reps = 1) {
    my_ap_axis temp;

    for (unsigned rep = 0; rep < reps * NumLines; rep++) {
#pragma HLS PIPELINE II = 1
        temp = in.read();
        out.write(temp.data(OutStreamW - 1, 0));
    }
}
template <unsigned InStreamW, unsigned OutStreamW, unsigned NumLines>
void AppendZeros(stream<ap_uint<InStreamW>> &in,
                 stream<ap_uint<OutStreamW>> &out, const unsigned reps = 1) {
    static_assert(InStreamW < OutStreamW,
                  "For AppendZeros in stream is wider than out stream.");

    ap_uint<OutStreamW> buffer;

    for (unsigned rep = 0; rep < reps * NumLines; rep++) {
        buffer(OutStreamW - 1, InStreamW) = 0;
        buffer(InStreamW - 1, 0) = in.read();
        out.write(buffer);
    }
}

template <unsigned InStreamW, unsigned OutStreamW, unsigned NumLines>
void ReduceWidth(stream<ap_uint<InStreamW>> &in,
                 stream<ap_uint<OutStreamW>> &out, const unsigned reps = 1) {
    static_assert(InStreamW % OutStreamW == 0,
                  "For ReduceWidth, InStreamW mod OutStreamW is not 0");

    const unsigned parts = InStreamW / OutStreamW;

    for (unsigned rep = 0; rep < reps * NumLines; rep++) {

        ap_uint<InStreamW> temp_in = in.read();
        for (unsigned p = 0; p < parts; p++) {
#pragma HLS PIPELINE II = 1

            ap_uint<OutStreamW> temp_out = temp_in(OutStreamW - 1, 0);
            out.write(temp_out);
            temp_in = temp_in >> OutStreamW;
        }
    }
}

template <unsigned InStreamW, unsigned OutStreamW, unsigned NumLines>
void ExpandWidth(stream<ap_uint<InStreamW>> &in,
                 stream<ap_uint<OutStreamW>> &out, const unsigned reps = 1) {
    static_assert(OutStreamW % InStreamW == 0,
                  "For ExpandWidth, OutStreamW mod InStreamW is not 0");

    const unsigned parts = OutStreamW / InStreamW;
    ap_uint<OutStreamW> buffer;

    for (unsigned rep = 0; rep < reps * NumLines; rep++) {

        for (unsigned p = 0; p < parts; p++) {
#pragma HLS PIPELINE II = 1
            ap_uint<InStreamW> temp = in.read();
            buffer((p + 1) * InStreamW - 1, p * InStreamW) = temp;
        }
        out.write(buffer);
    }
}

/**
 *  data width adjust
 *
 */
template <unsigned IN_BIT, unsigned OUT_BIT, unsigned IN_NUMS>
void adjust_width(stream<ap_uint<IN_BIT>> &in, stream<ap_uint<OUT_BIT>> &out,
                  const unsigned reps = 1) {
    static_assert(!(IN_BIT > OUT_BIT && IN_BIT % OUT_BIT != 0),
                  "For ReduceWidth, InStreamW mod OutStreamW is not 0");
    static_assert(!(IN_BIT < OUT_BIT && OUT_BIT % IN_BIT != 0),
                  "For ExpandWidth, OutStreamW mod InStreamW is not 0");

    if (IN_BIT > OUT_BIT) {
        // 减小位宽
        const unsigned PARTS = IN_BIT / OUT_BIT;

        for (unsigned rep = 0; rep < reps * IN_NUMS; rep++) {

            ap_uint<IN_BIT> temp_in = in.read();
            for (unsigned p = 0; p < PARTS; p++) {
#pragma HLS PIPELINE II = 1

                ap_uint<OUT_BIT> temp_out = temp_in(OUT_BIT - 1, 0);
                out.write(temp_out);
                temp_in = temp_in >> OUT_BIT;
            }
        }

    } else if (IN_BIT == OUT_BIT) {
        // 位宽不变
        // straight-through copy
        for (unsigned int i = 0; i < IN_NUMS * reps; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<IN_BIT> e = in.read();
            out.write(e);
        }
    } else {
        // 增大位宽
        const unsigned PARTS = OUT_BIT / IN_BIT;
        const unsigned OUT_NUMS = IN_NUMS / PARTS;
        ap_uint<OUT_BIT> buffer;

        for (unsigned rep = 0; rep < reps * OUT_NUMS; rep++) {

            for (unsigned p = 0; p < PARTS; p++) {
#pragma HLS PIPELINE II = 1
                ap_uint<IN_BIT> temp = in.read();
                buffer((p + 1) * IN_BIT - 1, p * IN_BIT) = temp;
            }
            out.write(buffer);
        }
    }
}
template <unsigned int InWidth,   // width of input stream
          unsigned int OutWidth,  // width of output stream
          unsigned int NumInWords // number of input words to process
          >
void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth>> &in,
                                       hls::stream<ap_uint<OutWidth>> &out,
                                       const unsigned int numReps) {
    if (InWidth > OutWidth) {
        // emit multiple output words per input word read
        // CASSERT_DATAFLOW(InWidth % OutWidth == 0);
        const unsigned int outPerIn = InWidth / OutWidth;
        const unsigned int totalIters = NumInWords * outPerIn * numReps;
        unsigned int o = 0;
        ap_uint<InWidth> ei = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II = 1
            // read new input word if current out count is zero
            if (o == 0) {
                ei = in.read();
            }
            // pick output word from the rightmost position
            ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
            out.write(eo);
            // shift input to get new output word for next iteration
            ei = ei >> OutWidth;
            // increment written output count
            o++;
            // wraparound indices to recreate the nested loop structure
            if (o == outPerIn) {
                o = 0;
            }
        }
    } else if (InWidth == OutWidth) {
        // straight-through copy
        for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<InWidth> e = in.read();
            out.write(e);
        }
    } else { // InWidth < OutWidth
        // read multiple input words per output word emitted
        // CASSERT_DATAFLOW(OutWidth % InWidth == 0);
        const unsigned int inPerOut = OutWidth / InWidth;
        const unsigned int totalIters = NumInWords * numReps;
        unsigned int i = 0;
        ap_uint<OutWidth> eo = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II = 1
            // read input and shift into output buffer
            ap_uint<InWidth> ei = in.read();
            eo = eo >> InWidth;
            eo(OutWidth - 1, OutWidth - InWidth) = ei;
            // increment read input count
            i++;
            // wraparound logic to recreate nested loop functionality
            if (i == inPerOut) {
                i = 0;
                out.write(eo);
            }
        }
    }
}

/**
 * 向流中写入指定数量的0
 */
template <unsigned IN_BIT>
void append_zero(stream<ap_uint<IN_BIT>> &in, const unsigned n) {
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
        in.write(0);
    }
}
/**
 * 从一个流中一定数量的数据move到另一个流
 */
template <unsigned IN_BIT>
void stream_move(stream<ap_uint<IN_BIT>> &in, stream<ap_uint<IN_BIT>> &out,
                 const unsigned n) {
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<IN_BIT> temp = in.read();
        out.write(temp);
    }
}

/**
 * 多路选择器
 * 将一路输入输出到三路中的一路
 */
template <unsigned BIT, unsigned NumLines>
void demux_stream3(stream<ap_uint<BIT>> &in, stream<ap_uint<BIT>> &out1,
                   stream<ap_uint<BIT>> &out2, stream<ap_uint<BIT>> &out3,
                   const unsigned short which, const unsigned reps = 1) {
    for (unsigned i = 0; i < NumLines * reps; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<BIT> temp = in.read();
        if (which == 0)
            out1.write(temp);
        else if (which == 1)
            out2.write(temp);
        else
            out3.write(temp);
    }
}

/**
 *  data width adjust
 *
 *
 */
template <unsigned IN_BIT, unsigned OUT_BIT>
void adjust_width_var(stream<ap_uint<IN_BIT>> &in,
                      stream<ap_uint<OUT_BIT>> &out, const unsigned in_nums,
                      const unsigned reps = 1) {
    static_assert(!(IN_BIT > OUT_BIT && IN_BIT % OUT_BIT != 0),
                  "For ReduceWidth, InStreamW mod OutStreamW is not 0");
    static_assert(!(IN_BIT < OUT_BIT && OUT_BIT % IN_BIT != 0),
                  "For ExpandWidth, OutStreamW mod InStreamW is not 0");

    if (IN_BIT > OUT_BIT) {
        // 减小位宽
        const unsigned PARTS = IN_BIT / OUT_BIT;

        for (unsigned rep = 0; rep < reps * in_nums; rep++) {

            ap_uint<IN_BIT> temp_in = in.read();
            for (unsigned p = 0; p < PARTS; p++) {
#pragma HLS PIPELINE II = 1

                ap_uint<OUT_BIT> temp_out = temp_in(OUT_BIT - 1, 0);
                out.write(temp_out);
                temp_in = temp_in >> OUT_BIT;
            }
        }

    } else if (IN_BIT == OUT_BIT) {
        // 位宽不变
        // straight-through copy
        for (unsigned int i = 0; i < in_nums * reps; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<IN_BIT> e = in.read();
            out.write(e);
        }
    } else {
        // 增大位宽
        const unsigned PARTS = OUT_BIT / IN_BIT;
        const unsigned out_nums = in_nums / PARTS;
        ap_uint<OUT_BIT> buffer;

        for (unsigned rep = 0; rep < reps * out_nums; rep++) {

            for (unsigned p = 0; p < PARTS; p++) {
#pragma HLS PIPELINE II = 1
                ap_uint<IN_BIT> temp = in.read();
                buffer((p + 1) * IN_BIT - 1, p * IN_BIT) = temp;
            }
            out.write(buffer);
        }
    }
}

/**
 * 2路选择器 并且调整数据位宽
 * 将一路输入 输出到两路中的一路
 *
 */
template <unsigned IN_BIT, unsigned OUT0_BIT, unsigned OUT1_BIT>
void demux_stream1to2_adj(stream<ap_uint<IN_BIT>> &in,
                          stream<ap_uint<OUT0_BIT>> &out0,
                          stream<ap_uint<OUT1_BIT>> &out1,
                          const unsigned short which, const unsigned in_nums,
                          const unsigned reps = 1) {
    if (which == 0) {
        adjust_width_var<IN_BIT, OUT0_BIT>(in, out0, in_nums, reps);
    } else {
        adjust_width_var<IN_BIT, OUT1_BIT>(in, out1, in_nums, reps);
    }
}

/**
 * 2 选 1
 *
 */
template <unsigned IN0_BIT, unsigned IN1_BIT, unsigned OUT_BIT>
void demux_stream2to1_adj(stream<ap_uint<IN0_BIT>> &in0,
                          stream<ap_uint<IN1_BIT>> &in1,
                          stream<ap_uint<OUT_BIT>> &out,
                          const unsigned short which, const unsigned in0_nums,
                          const unsigned in1_nums, const unsigned reps = 1) {
    if (which == 0) {
        adjust_width_var<IN0_BIT, OUT_BIT>(in0, out, in0_nums, reps);
    } else {
        adjust_width_var<IN1_BIT, OUT_BIT>(in1, out, in1_nums, reps);
    }
}

/**
 * 从内存中读取 nums个数到 流中
 */
template <unsigned int BIT>
void mem_to_stream(ap_uint<BIT> *in, stream<ap_uint<BIT>> &out,
                   const unsigned nums_per_rep, const unsigned reps) {
    for (unsigned int rep = 0; rep < reps; rep++) {
        for (unsigned int i = 0; i < nums_per_rep; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<BIT> e = in[i];
            out.write(e);
        }
    }
}

/**
 * 写数据到内存中
 */
template <unsigned int BIT>
void stream_to_mem(stream<ap_uint<BIT>> &in, ap_uint<BIT> *out,
                   const unsigned nums_per_rep, const unsigned reps) {
    for (unsigned int rep = 0; rep < reps; rep++) {
        for (unsigned int i = 0; i < nums_per_rep; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<BIT> e = in.read();
            out[i] = e;
        }
    }
}

template <unsigned BIT>
void in_to_stream(stream<my_ap_axis> &in, stream<ap_uint<BIT>> &out,
                  const unsigned nums = 1) {
    my_ap_axis temp;
    for (unsigned num = 0; num < nums; num++) {
        temp = in.read();
        out.write(temp.data(BIT - 1, 0));
    }
}

template <unsigned BIT>
void stream_to_out(stream<ap_uint<BIT>> &in, stream<my_ap_axis> &out,
                   const unsigned nums = 1) {
    my_ap_axis temp;
    temp.keep = "0xffffffffffffffff";

    for (unsigned i = 0; i < nums - 1; i++) {
#pragma HLS PIPELINE II = 1
        temp.data = in.read();
        temp.last = 0;
        out.write(temp);
    }

    temp.data = in.read();
    temp.last = 1;
    out.write(temp);
}