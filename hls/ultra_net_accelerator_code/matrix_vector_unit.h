#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include "function.h"


/**
 *  simd 乘
 * 使用 逻辑查找表
 */
template <	unsigned W_BIT,
			unsigned IN_BIT,
			unsigned M_BIT,
			unsigned SIMD>
ap_int<M_BIT> simd_mul_lut(
	ap_uint<SIMD*W_BIT> weights, 
	ap_uint<SIMD*IN_BIT> in) 
{	
	ap_int<M_BIT> accumulation = 0;

	for (unsigned p = 0; p < SIMD; p++) {
#pragma HLS UNROLL
		ap_int<W_BIT> temp_w = weights( (p+1)*W_BIT-1, p*W_BIT );
		ap_uint<IN_BIT> temp_in = in( (p+1)*IN_BIT-1, p*IN_BIT );
		ap_int<W_BIT + IN_BIT> result = temp_w * temp_in;
#pragma HLS RESOURCE variable=result core=Mul_LUT
		accumulation += result;
	}
	return accumulation;
}

/**
 *  simd 乘
 *  由 编译器自动选择使用 dsp 或者 lut
 */
template <	unsigned W_BIT,
			unsigned IN_BIT,
			unsigned M_BIT,
			unsigned SIMD>
ap_int<M_BIT> simd_mul(
	ap_uint<SIMD*W_BIT> weights, 
	ap_uint<SIMD*IN_BIT> in) 
{	
	ap_int<M_BIT> accumulation = 0;

	for (unsigned p = 0; p < SIMD; p++) {
#pragma HLS UNROLL
		ap_int<W_BIT> temp_w = weights( (p+1)*W_BIT-1, p*W_BIT );
		ap_uint<IN_BIT> temp_in = in( (p+1)*IN_BIT-1, p*IN_BIT );
		ap_int<W_BIT + IN_BIT> result = temp_w * temp_in;
// #pragma HLS RESOURCE variable=result core=Mul_LUT
		accumulation += result;
	}
	return accumulation;
}

/**
 * 矩阵向量计算单元
 * 
 */
template <	unsigned MAT_ROW,		// 展开后的k × k × in_ch
			unsigned MAT_COL,		// 展开后的out_ch
			unsigned IN_BIT,
			unsigned W_BIT,
			unsigned M_BIT,			// 乘累加后的计算结果的值
			unsigned SIMD,
			unsigned PE,
			unsigned VECT_NUMS>
void matrix_vector_unit(
	stream<ap_uint<SIMD*IN_BIT> >& vec, 
	const ap_uint<SIMD*W_BIT> weights[PE][(MAT_ROW/SIMD)*(MAT_COL/PE)], 
	stream<ap_uint<PE*M_BIT> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MAT_ROW%SIMD == 0, "MAT_ROW mod SIMD is not 0" );
	static_assert( MAT_COL%PE == 0, "MAT_COL mod PE is not 0");

	const unsigned INPUT_FOLD = MAT_ROW/SIMD;
	const unsigned OUTPUT_FOLD = MAT_COL/PE;

	const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	// const unsigned total_reps = 18;
	// 需要保存一行数据
	ap_uint<SIMD*IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

	// 用来保存累加结果
// 	ap_uint<M_BIT> result_vec[PE];
// #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
	unsigned in_fold_cnt = 0;			// 输入折叠计数
	unsigned out_fold_cnt = 0;			// 输出折叠计数
	unsigned tile = 0;

	// 一次 读入的数据 需要保存 in_ch * k * k长度的数据
	ap_uint<SIMD*IN_BIT> temp_vec;
	// 累加结果 这里需要初始化为0 

	// TODO
	ap_int<M_BIT> acc[PE];

	// cout << "acc init value \n";
	// for(unsigned i=0; i < PE; i ++) {
	// 	cout << acc[i] << "  ";
	// }
	// static ap_uint<M_BIT> acc1[PE] = {0};

	// cout << "acc1 init value \n";
	// for(unsigned i=0; i < PE; i ++) {
	// 	cout << acc1[i] << "  ";
	// }

	// total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1
		
		// 这里是在第一次输出之前 就度完了数据，之后一直用
		// 在输出折叠第一次计算时读
		if (out_fold_cnt == 0) {
			temp_vec = vec.read();
			row_store[in_fold_cnt] = temp_vec;
		}
		else {
			temp_vec = row_store[in_fold_cnt];
		}

		// index = wVec*OutputFold+wMat;

		// 初始化累加结果
		if(in_fold_cnt == 0) {
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				acc[p] = 0;
			}
		}

		// 主要计算单元 这里用UNROLL展开 期望用单周期实现计算
		// PE 并行计算
		for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
			// 读 W 子块
			ap_uint<SIMD*W_BIT> temp_mat = weights[p][tile];
			// SIMD 并行
			acc[p] += simd_mul<W_BIT, IN_BIT, M_BIT, SIMD>( temp_mat, temp_vec );
		}

		// 计数逻辑 和输出处理
		tile ++;
		if(++ in_fold_cnt == INPUT_FOLD) {
			in_fold_cnt = 0;
			ap_uint<PE*M_BIT> out_buf;
			// PE 列计算完成 可以输出
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				out_buf((p+1)*M_BIT-1, p*M_BIT) = acc[p];
				// acc[p] = 0;
			}
			out.write(out_buf);
			// 完整的一次矩阵向量计算
			if(++ out_fold_cnt == OUTPUT_FOLD) {
				out_fold_cnt = 0;
				tile = 0;
			}

		}
	}  // end for

}

/**
 * 矩阵向量计算单元
 * 同时进行量化激活处理
 */
template <	unsigned MAT_ROW,		// 展开后的k × k × in_ch
			unsigned MAT_COL,		// 展开后的out_ch

			unsigned IN_BIT,
			unsigned OUT_BIT,		// 

			unsigned W_BIT,
			unsigned M_BIT,			// 乘累加后的计算结果的值

			unsigned INC_BIT,		// 激活等差数列 的步长
			unsigned BIAS_BIT,		// 

			unsigned SIMD,
			unsigned PE,
			unsigned L_SHIFT,
			unsigned VECT_NUMS>
void matrix_vector_act_unit(
	stream<ap_uint<SIMD*IN_BIT> >& vec, 
	const ap_uint<SIMD*W_BIT> weights[PE][(MAT_ROW/SIMD)*(MAT_COL/PE)], 
	const ap_int<INC_BIT> inc[PE][MAT_COL/PE],
	const ap_int<BIAS_BIT> bias[PE][MAT_COL/PE],
	stream<ap_uint<PE*OUT_BIT> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MAT_ROW%SIMD == 0, "MAT_ROW mod SIMD is not 0" );
	static_assert( MAT_COL%PE == 0, "MAT_COL mod PE is not 0");

	const unsigned INPUT_FOLD = MAT_ROW/SIMD;
	const unsigned OUTPUT_FOLD = MAT_COL/PE;

	const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	

	// 需要保存一行数据
	ap_uint<SIMD*IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

	// 用来保存累加结果
	// ap_uint<M_BIT> result_vec[PE];
// #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
	unsigned in_fold_cnt = 0;			// 输入折叠计数
	unsigned out_fold_cnt = 0;			// 输出折叠计数
	unsigned tile = 0;

	// 一次 读入的数据 需要保存 in_ch * k * k长度的数据
	ap_uint<SIMD*IN_BIT> temp_vec;
	// 累加结果 这里需要初始化为0 
	ap_int<M_BIT> acc[PE];

	// total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1
		
		// 这里是在第一次输出之前 就度完了数据，之后一直用
		// 在输出折叠第一次计算时读
		if (out_fold_cnt == 0) {
			temp_vec = vec.read();
			row_store[in_fold_cnt] = temp_vec;
		}
		else {
			temp_vec = row_store[in_fold_cnt];
		}

		// index = wVec*OutputFold+wMat;

		// 初始化累加结果
		if(in_fold_cnt == 0) {
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				acc[p] = 0;
			}
		}

		// 主要计算单元 这里用UNROLL展开 期望用单周期实现计算
		// PE 并行计算
		for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
			// 读 W 子块
			ap_uint<SIMD*W_BIT> temp_mat = weights[p][tile];
			// SIMD 并行
			acc[p] += simd_mul<W_BIT, IN_BIT, M_BIT, SIMD>( temp_mat, temp_vec );
			// if (p == 0)
			// 	cout << temp_vec(7, 0) << " " <<  temp_vec(15, 8) << " " << temp_vec(23, 16) << endl;
		}

		// 计数逻辑 和输出处理
		tile ++;
		if(++ in_fold_cnt == INPUT_FOLD) {
			in_fold_cnt = 0;
			ap_uint<PE*M_BIT> out_buf;
			// PE 列计算完成 可以输出
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) = bn_qurelu<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT, L_SHIFT>(acc[p], inc[p][out_fold_cnt], bias[p][out_fold_cnt]);
				// cout << acc[p] << " " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) << " " << inc[p][out_fold_cnt] << " " << bias[p][out_fold_cnt] << "     ";
				// acc[p] = 0;
			}
			out.write(out_buf);
			// 完整的一次矩阵向量计算
			if(++ out_fold_cnt == OUTPUT_FOLD) {
				out_fold_cnt = 0;
				tile = 0;
			}

		}
	}  // end for

}

/**
 * 矩阵向量计算单元
 * 使用 lut 计算
 * 
 */
template <	unsigned MAT_ROW,		// 展开后的k × k × in_ch
			unsigned MAT_COL,		// 展开后的out_ch
			unsigned IN_BIT,
			unsigned W_BIT,
			unsigned M_BIT,			// 乘累加后的计算结果的值
			unsigned SIMD,
			unsigned PE,
			unsigned VECT_NUMS>
void matrix_vector_unit_lut(
	stream<ap_uint<SIMD*IN_BIT> >& vec, 
	const ap_uint<SIMD*W_BIT> weights[PE][(MAT_ROW/SIMD)*(MAT_COL/PE)], 
	stream<ap_uint<PE*M_BIT> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MAT_ROW%SIMD == 0, "MAT_ROW mod SIMD is not 0" );
	static_assert( MAT_COL%PE == 0, "MAT_COL mod PE is not 0");

	const unsigned INPUT_FOLD = MAT_ROW/SIMD;
	const unsigned OUTPUT_FOLD = MAT_COL/PE;

	const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	// const unsigned total_reps = 18;
	// 需要保存一行数据
	ap_uint<SIMD*IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

	// 用来保存累加结果
// 	ap_uint<M_BIT> result_vec[PE];
// #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
	unsigned in_fold_cnt = 0;			// 输入折叠计数
	unsigned out_fold_cnt = 0;			// 输出折叠计数
	unsigned tile = 0;

	// 一次 读入的数据 需要保存 in_ch * k * k长度的数据
	ap_uint<SIMD*IN_BIT> temp_vec;
	// 累加结果 这里需要初始化为0 

	// TODO
	ap_int<M_BIT> acc[PE];

	// cout << "acc init value \n";
	// for(unsigned i=0; i < PE; i ++) {
	// 	cout << acc[i] << "  ";
	// }
	// static ap_uint<M_BIT> acc1[PE] = {0};

	// cout << "acc1 init value \n";
	// for(unsigned i=0; i < PE; i ++) {
	// 	cout << acc1[i] << "  ";
	// }

	// total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1
		
		// 这里是在第一次输出之前 就度完了数据，之后一直用
		// 在输出折叠第一次计算时读
		if (out_fold_cnt == 0) {
			temp_vec = vec.read();
			row_store[in_fold_cnt] = temp_vec;
		}
		else {
			temp_vec = row_store[in_fold_cnt];
		}

		// index = wVec*OutputFold+wMat;

		// 初始化累加结果
		if(in_fold_cnt == 0) {
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				acc[p] = 0;
			}
		}

		// 主要计算单元 这里用UNROLL展开 期望用单周期实现计算
		// PE 并行计算
		for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
			// 读 W 子块
			ap_uint<SIMD*W_BIT> temp_mat = weights[p][tile];
			// SIMD 并行
			acc[p] += simd_mul_lut<W_BIT, IN_BIT, M_BIT, SIMD>( temp_mat, temp_vec );
		}

		// 计数逻辑 和输出处理
		tile ++;
		if(++ in_fold_cnt == INPUT_FOLD) {
			in_fold_cnt = 0;
			ap_uint<PE*M_BIT> out_buf;
			// PE 列计算完成 可以输出
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				out_buf((p+1)*M_BIT-1, p*M_BIT) = acc[p];
				// acc[p] = 0;
			}
			out.write(out_buf);
			// 完整的一次矩阵向量计算
			if(++ out_fold_cnt == OUTPUT_FOLD) {
				out_fold_cnt = 0;
				tile = 0;
			}

		}
	}  // end for

}

/**
 * 矩阵向量计算单元
 * 同时进行量化激活处理
 * 使用 lut 计算
 */
template <	unsigned MAT_ROW,		// 展开后的k × k × in_ch
			unsigned MAT_COL,		// 展开后的out_ch

			unsigned IN_BIT,
			unsigned OUT_BIT,		// 

			unsigned W_BIT,
			unsigned M_BIT,			// 乘累加后的计算结果的值

			unsigned INC_BIT,		// 激活等差数列 的步长
			unsigned BIAS_BIT,		// 

			unsigned SIMD,
			unsigned PE,
			unsigned L_SHIFT,
			unsigned VECT_NUMS>
void matrix_vector_act_unit_lut(
	stream<ap_uint<SIMD*IN_BIT> >& vec, 
	const ap_uint<SIMD*W_BIT> weights[PE][(MAT_ROW/SIMD)*(MAT_COL/PE)], 
	const ap_uint<INC_BIT> inc[PE][MAT_COL/PE],
	const ap_int<BIAS_BIT> bias[PE][MAT_COL/PE],
	stream<ap_uint<PE*OUT_BIT> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MAT_ROW%SIMD == 0, "MAT_ROW mod SIMD is not 0" );
	static_assert( MAT_COL%PE == 0, "MAT_COL mod PE is not 0");

	const unsigned INPUT_FOLD = MAT_ROW/SIMD;
	const unsigned OUTPUT_FOLD = MAT_COL/PE;

	const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	

	// 需要保存一行数据
	ap_uint<SIMD*IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

	// 用来保存累加结果
	// ap_uint<M_BIT> result_vec[PE];
// #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
	unsigned in_fold_cnt = 0;			// 输入折叠计数
	unsigned out_fold_cnt = 0;			// 输出折叠计数
	unsigned tile = 0;

	// 一次 读入的数据 需要保存 in_ch * k * k长度的数据
	ap_uint<SIMD*IN_BIT> temp_vec;
	// 累加结果 这里需要初始化为0 
	ap_int<M_BIT> acc[PE];

	// total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1
		
		// 这里是在第一次输出之前 就度完了数据，之后一直用
		// 在输出折叠第一次计算时读
		if (out_fold_cnt == 0) {
			temp_vec = vec.read();
			row_store[in_fold_cnt] = temp_vec;
		}
		else {
			temp_vec = row_store[in_fold_cnt];
		}

		// index = wVec*OutputFold+wMat;

		// 初始化累加结果
		if(in_fold_cnt == 0) {
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				acc[p] = 0;
			}
		}

		// 主要计算单元 这里用UNROLL展开 期望用单周期实现计算
		// PE 并行计算
		for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
			// 读 W 子块
			ap_uint<SIMD*W_BIT> temp_mat = weights[p][tile];
			// SIMD 并行
			acc[p] += simd_mul_lut<W_BIT, IN_BIT, M_BIT, SIMD>( temp_mat, temp_vec );
		}

		// 计数逻辑 和输出处理
		tile ++;
		if(++ in_fold_cnt == INPUT_FOLD) {
			in_fold_cnt = 0;
			ap_uint<PE*M_BIT> out_buf;
			// PE 列计算完成 可以输出
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) = bn_qurelu<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT, L_SHIFT>(acc[p], inc[p][out_fold_cnt], bias[p][out_fold_cnt]);
				// cout << acc[p] << " " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) << " " << inc[p][out_fold_cnt] << " " << bias[p][out_fold_cnt] << "     ";
				// acc[p] = 0;
			}
			out.write(out_buf);
			// 完整的一次矩阵向量计算
			if(++ out_fold_cnt == OUTPUT_FOLD) {
				out_fold_cnt = 0;
				tile = 0;
			}

		}
	}  // end for

}

