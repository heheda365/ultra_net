#include <stdint.h>
#include <ap_int.h>
#include "stream_tools.h"
#include "conv2d.h"

#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"
#include "function.h"
#include "stream_tools.h"

#define IN_ROW      2
#define IN_COL      4
#define IN_CH       3
#define BIT         8


int main(int argc, char const *argv[])
{
    ap_uint<IN_CH*BIT> data[IN_ROW][IN_COL];
    hls::stream<ap_uint<IN_CH*BIT>> in("in");
    for (int i=0; i < IN_ROW; i ++) {
        for (int j=0; j < IN_COL; j ++) {
            // data[i][j] = i * 400 + j + 1;
            in.write(i * IN_COL + j + 1);
        }
    }
    hls::stream<ap_uint<IN_CH*BIT>> padding_out("padding_out");
    padding<IN_ROW, IN_COL, IN_CH, BIT, 1>(in, padding_out, 1);

    // for (int i=0; i < IN_ROW + 2; i ++) {
    //     for (int j=0; j < IN_COL + 2; j ++) {
    //         ap_uint<IN_CH*BIT> a = padding_out.read();
    //         if (i < 3 && j < 3) {

    //             cout << " " << a;
    //         }
    //     }
    // }

    hls::stream<ap_uint<IN_CH*BIT>> swu_out("swu_out");
    SWU<3, 1, IN_COL + 2, IN_COL + 2, IN_CH, BIT>(padding_out, swu_out, 1);

    for(int i=0; i < IN_ROW*IN_COL; i ++) {
        for (int j=0; j < 9; j ++) {
            cout << " " << swu_out.read();
        }
        cout << endl;
    }
    
    return 0;
}
