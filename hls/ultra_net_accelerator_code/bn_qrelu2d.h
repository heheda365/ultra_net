#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include "function.h"
#include "stream_tools.h"

/**
 * 输出 元素宽度只有 OUT_BIT
 */
template <	unsigned IN_ROW,
            unsigned IN_COL,
            unsigned IN_CH,

            unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
            unsigned PE>
void bn_qrelu2d(    
    stream<ap_uint<IN_CH*IN_BIT> >& in, 
    const ap_uint<INC_BIT> inc[PE][IN_CH/PE],
	const ap_int<BIAS_BIT> bias[PE][IN_CH/PE],
	stream<ap_uint<OUT_BIT> >& out, 
	const unsigned reps = 1) {

    hls::stream<ap_uint<IN_BIT>> adj_out("adj_out");
    adjust_width<IN_CH*IN_BIT, IN_BIT, IN_ROW*IN_COL>(in, adj_out, reps);

    unsigned pe = 0;
    unsigned f = 0;
    for(unsigned rep=0; rep < IN_ROW*IN_COL*IN_CH*reps; rep ++) {
        ap_int<IN_BIT> in_elem = adj_out.read();
        ap_uint<OUT_BIT> out_elem = bn_qurelu<IN_BIT, OUT_BIT, INC_BIT, BIAS_BIT>(in_elem, inc[pe][f], bias[pe][f]);
        out.write(out_elem);

        if(++ pe == PE) {
            pe = 0;
            if(++ f == IN_CH/PE) {
                f = 0;
            }
        }
    }
}