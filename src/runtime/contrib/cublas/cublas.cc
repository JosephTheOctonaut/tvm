/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file Use external cblas library call.
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include "../../3rdparty/compiler-rt/builtin_fp16.h"
#include "../cblas/gemm_common.h"
#include "cublas_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;
#define kE4M3Float_code DataType::TypeCode::kE4M3Float
#define kE5M2Float_code DataType::TypeCode::kE5M2Float
  
inline cublasOperation_t CUBLASBooleanToTranspose(bool item) {
  return item ? CUBLAS_OP_T : CUBLAS_OP_N;
}

inline void CUBLASTryEnableTensorCore(cublasHandle_t hdl) {
  // TensorCores are only supported in cublas 9.0 or higher
  int version;
  CHECK_CUBLAS_ERROR(cublasGetVersion(hdl, &version));
  if (version >= 9000) CHECK_CUBLAS_ERROR(cublasSetMathMode(hdl, CUBLAS_DEFAULT_MATH));
}

struct CublasHgemmOp {
  typedef half TDatatype;
  cublasHandle_t handle;
  explicit CublasHgemmOp(cublasHandle_t hdl) : handle(hdl) {}

  void operator()(bool ta, bool tb, int M, int N, int K, half alpha, half* A, int lda, half* B,
                  int ldb, half beta, half* C, int ldc) {
    CHECK_CUBLAS_ERROR(cublasHgemm(handle, CUBLASBooleanToTranspose(ta),
                                   CUBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                   &beta, C, ldc));
  }
};

struct CublasSgemmOp {
  typedef float TDatatype;
  cublasHandle_t handle;
  explicit CublasSgemmOp(cublasHandle_t hdl) : handle(hdl) {}

  void operator()(bool ta, bool tb, int M, int N, int K, float alpha, float* A, int lda, float* B,
                  int ldb, float beta, float* C, int ldc) {
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLASBooleanToTranspose(ta),
                                   CUBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                   &beta, C, ldc));
  }
};

struct CublasDgemmOp {
  typedef double TDatatype;
  cublasHandle_t handle;
  explicit CublasDgemmOp(cublasHandle_t hdl) : handle(hdl) {}
  void operator()(bool ta, bool tb, int M, int N, int K, double alpha, double* A, int lda,
                  double* B, int ldb, double beta, double* C, int ldc) {
    CHECK_CUBLAS_ERROR(cublasDgemm(handle, CUBLASBooleanToTranspose(ta),
                                   CUBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                   &beta, C, ldc));
  }
};

struct CublasHgemmBatchOp {
  typedef half TDatatype;
  cublasHandle_t handle;
  explicit CublasHgemmBatchOp(cublasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, half alpha, half* A,
                  int a_stride, int lda, half* B, int b_stride, int ldb, half beta, half* C,
                  int c_stride, int ldc) {
    CHECK_CUBLAS_ERROR(cublasHgemmStridedBatched(
        handle, CUBLASBooleanToTranspose(ta), CUBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda,
        a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

struct CublasSgemmBatchOp {
  typedef float TDatatype;
  cublasHandle_t handle;
  explicit CublasSgemmBatchOp(cublasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, float alpha, float* A,
                  int a_stride, int lda, float* B, int b_stride, int ldb, float beta, float* C,
                  int c_stride, int ldc) {
    CHECK_CUBLAS_ERROR(cublasSgemmStridedBatched(
        handle, CUBLASBooleanToTranspose(ta), CUBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda,
        a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

struct CublasDgemmBatchOp {
  typedef double TDatatype;
  cublasHandle_t handle;
  explicit CublasDgemmBatchOp(cublasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, double alpha, double* A,
                  int a_stride, int lda, double* B, int b_stride, int ldb, double beta, double* C,
                  int c_stride, int ldc) {
    CHECK_CUBLAS_ERROR(cublasDgemmStridedBatched(
        handle, CUBLASBooleanToTranspose(ta), CUBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda,
        a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

// Check cublas supported mix-precision computation type and return computeType
bool CheckMixPrecisionType(DLDataType in_dtype, DLDataType out_dtype, bool int_support = true) {
  if (int_support && TypeMatch(out_dtype, kDLInt, 32)) {
    return TypeMatch(in_dtype, kDLInt, 8);
  } else if (TypeMatch(out_dtype, kDLFloat, 32)) {
    return TypeMatch(in_dtype, kDLInt, 8) || TypeMatch(in_dtype, kDLFloat, 16);
  } else {
    return false;
  }
}

// Check for supported cublasLt fp8 combinations
bool CheckFP8FormatTypes(DLDataType A_dtype, DLDataType B_dtype, DLDataType D_dtype) {
  // Ignoring BF16 and F32 for simplicity, but those are also supported in some combinations
  if (TypeMatch(A_dtype, kE4M3Float_code, 8)) {
    if (TypeMatch(B_dtype, kE4M3Float_code, 8)) {
      // If both operands are e4m3, output must be e4m3 or fp16
      return TypeMatch(D_dtype, kE4M3Float_code, 8) || TypeMatch(D_dtype, kDLFloat, 16);
    }
    else if (TypeMatch(B_dtype, kE5M2Float_code, 8)) {
      // If A is e4m3 and B is e5m2, output can be e4m3, e5m2, or fp16
      return TypeMatch(D_dtype, kE4M3Float_code, 8) || TypeMatch(D_dtype, kE5M2Float_code, 8) || TypeMatch(D_dtype, kDLFloat, 16);
    }
    else return false;
  }
  else if (TypeMatch(A_dtype, kE5M2Float_code, 8)) {
    if (TypeMatch(B_dtype, kE4M3Float_code, 8)) {
      // if A is e5m2, B must be e4m3, and output can be e4m3, e5m2, or fp16
      return TypeMatch(D_dtype, kE4M3Float_code, 8) || TypeMatch(D_dtype, kE5M2Float_code, 8) || TypeMatch(D_dtype, kDLFloat, 16);
    }
    else return false;
  }
  else return false;
}

int roundoff(int v, int d) { return (v + d - 1) / d * d; }

bool CheckIsFP8(DLDataType t) {
  // Check if the dtype is either nvidia fp8 type
  return TypeMatch(t, kE4M3Float_code, 8) || TypeMatch(t, kE5M2Float_code, 8);
}    
  

#if CUDART_VERSION >= 10010

void CallCublasLt(cublasLtHandle_t hdl, cudaStream_t stream,
                  cublasLtMatmulPreference_t matmul_pref_desc, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  void* workspace_ptr, size_t workspace_size, cublasLtEpilogue_t epilogue) {
  ICHECK( TypeEqual(A->dtype, B->dtype)
	  || (CheckIsFP8(A->dtype) && CheckIsFP8(B->dtype)) );  // fp8 allows mixed format (eg e4m3 x e5m2)
  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  auto compute_type = CUBLAS_COMPUTE_32F;
  auto scale_type = CUDA_R_32F;
  cudaDataType_t ab_type = CUDA_R_32F;
  cudaDataType_t c_type = CUDA_R_32F;
  float one_fp32 = 1.0;
  float zero_fp32 = 0.0;
  auto one_fp16 = __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(1.0);
  auto zero_fp16 = __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(0.0);
  int32_t one_i32 = 1;
  int32_t zero_i32 = 0;
  void* alpha = &one_fp32;
  void* beta = &zero_fp32;

  if (TypeMatch(A->dtype, kDLFloat, 16)) {
    ab_type = CUDA_R_16F;
  } else if (TypeMatch(A->dtype, kDLInt, 8)) {
    ab_type = CUDA_R_8I;
  }

  if (TypeMatch(C->dtype, kDLFloat, 16)) {
    c_type = CUDA_R_16F;
    // compute_type = CUBLAS_COMPUTE_16F;
    // scale_type = CUDA_R_16F;
    // alpha = &one_fp16;
    // beta = &zero_fp16;
  } else if (TypeMatch(C->dtype, kDLInt, 32)) {
    c_type = CUDA_R_32I;
    compute_type = CUBLAS_COMPUTE_32I;
    scale_type = CUDA_R_32I;
    alpha = &one_i32;
    beta = &zero_i32;
  }

  cublasLtMatmulDesc_t op_desc;
  cublasOperation_t op_transa = CUBLASBooleanToTranspose(transa);
  cublasOperation_t op_transb = CUBLASBooleanToTranspose(transb);

  CHECK_CUBLAS_ERROR(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &op_transb, sizeof(op_transb)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &op_transa, sizeof(op_transa)));

  if (bias != nullptr) {
    CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias->data, sizeof(float*)));
  }

  if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
    CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                      &epilogue, sizeof(epilogue)));
  }

  int batch_offset_A = A->ndim - 2;
  int batch_offset_B = B->ndim - 2;

  int M = ColumnCount(B, transb, batch_offset_B);
  int N = RowCount(A, transa, batch_offset_A);
  int K = ColumnCount(A, transa, batch_offset_A);
  bool use_batched_gemm = A->ndim > 2 || B->ndim > 2;

  // If A is batched but B is not, flatten all non-reduction axes of A to use the regular GEMM.
  // This trick is only applicable if batch axes and the other spatial axis (M or N) are
  // adjacent in both the input and the output matrix. In particular, if A is of shape (M, K)
  // and B matrix is of shape (Batch, N, K) with transb = true, the output shape
  // is (Batch, M, N). Since the Batch and the N axes are not adjacent in the output, we cannot
  // use the regular GEMM if only B is batched.
  if (A->ndim > 2 && B->ndim == 2 && transa == false) {
    N = 1;
    for (int i = 0; i < A->ndim - 1; ++i) {
      N *= A->shape[i];
    }
    use_batched_gemm = false;
  }

  int lda = transb ? K : M;
  int ldb = transa ? N : K;
  int ldc = M;

  cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
  CHECK_CUBLAS_ERROR(
      cublasLtMatrixLayoutCreate(&A_desc, ab_type, !transb ? M : K, !transb ? K : M, lda));
  CHECK_CUBLAS_ERROR(
      cublasLtMatrixLayoutCreate(&B_desc, ab_type, !transa ? K : N, !transa ? N : K, ldb));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&C_desc, c_type, M, N, ldc));

  if (use_batched_gemm) {
    auto get_batch_count = [](int64_t* shape, int batch_offset) {
      int64_t count = 1;
      for (int i = 0; i < batch_offset; ++i) {
        count *= shape[i];
      }
      return count;
    };
    auto set_batch = [](cublasLtMatrixLayout_t mat_desc, int batch_count, int64_t batch_stride) {
      CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
          mat_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
      CHECK_CUBLAS_ERROR(
          cublasLtMatrixLayoutSetAttribute(mat_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                           &batch_stride, sizeof(batch_stride)));
    };

    int batch_count_A = get_batch_count(A->shape, batch_offset_A);
    int batch_count_B = get_batch_count(B->shape, batch_offset_B);
    int batch_count_C = get_batch_count(C->shape, C->ndim - 2);
    int64_t batch_stride_A = M * K;
    int64_t batch_stride_B = K * N;
    int64_t batch_stride_C = M * N;

    // cuBLASLt does not seem to support batched GEMM with one of matrices having
    // one batch (with batch_stride 0).
    ICHECK_EQ(batch_count_A, batch_count_B);

    set_batch(A_desc, batch_count_A, batch_stride_A);
    set_batch(B_desc, batch_count_B, batch_stride_B);
    set_batch(C_desc, batch_count_C, batch_stride_C);
  }

  auto A_data = static_cast<char*>(A->data) + A->byte_offset;
  auto B_data = static_cast<char*>(B->data) + B->byte_offset;
  auto C_data = static_cast<char*>(C->data) + C->byte_offset;

  cublasLtMatmulPreferenceSetAttribute(matmul_pref_desc, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspace_size, sizeof(size_t));

  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  int returned_result = 0;
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(hdl, op_desc, A_desc, B_desc, C_desc, C_desc,
                                                    matmul_pref_desc, 1, &heuristic_result,
                                                    &returned_result));
  if (returned_result == 0) {
    CHECK_CUBLAS_ERROR(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  CHECK_CUBLAS_ERROR(cublasLtMatmul(hdl, op_desc, alpha, B_data, A_desc, A_data, B_desc, beta,
                                    C_data, C_desc, C_data, C_desc, &heuristic_result.algo,
                                    workspace_ptr, workspace_size, stream));

  cublasLtMatmulDescDestroy(op_desc);
  cublasLtMatrixLayoutDestroy(A_desc);
  cublasLtMatrixLayoutDestroy(B_desc);
  cublasLtMatrixLayoutDestroy(C_desc);
}

inline void CallLtFP8Gemm(TVMArgs args, TVMRetValue* ret, cublasLtHandle_t hdl, cudaStream_t stream) {
  // This function uses the cublsLt notion wherever possible. A and B are operands. C is bias. D is output.
  
  // TODO: make sure TVM alignment is at least 16 bytes (tensor pointers must be 16B aligned)
  // TODO: currently defaulting to column order layout; double check that's right
  
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* D = args[2];
  bool transa = args[3];
  bool transb = args[4];
  DLTensor* A_scale = args[5];
  DLTensor* B_scale = args[6];
  DLTensor* D_scale = args[7];
  bool fast_accum = args[8];

  ICHECK( !(IsInPlaceTransposed(A) || IsInPlaceTransposed(B)) ) << "in place transpose currently unsupported for fp8";
  ICHECK(transa && !transb) << "FP8 cublas calls require A to be tranposed and B not transposed ('TN' format)";
  ICHECK_EQ(A->ndim, 2) << "FP8 batched matmul cublas calls not curently implemented";
  ICHECK_EQ(B->ndim, 2) << "FP8 batched matmul cublas calls not curently implemented";
  ICHECK_EQ(D->ndim, 2) << "FP8 batched matmul cublas calls not curently implemented";
  ICHECK_EQ(ElementStride(A), 1);
  ICHECK_EQ(ElementStride(B), 1);
  ICHECK_EQ(ElementStride(D), 1);
  CheckFP8FormatTypes(A->dtype, B->dtype, D->dtype);
  
  auto compute_type = CUBLAS_COMPUTE_32F; // FP8 calls must use compute type CUBLAS_COMPUTE_32F
  auto scale_type = CUDA_R_32F;  // FP8 calls must use scale type CUDA_R_32F
  cudaDataType_t a_type = TypeMatch(A->dtype, kE4M3Float_code, 8) ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
  cudaDataType_t b_type = TypeMatch(B->dtype, kE4M3Float_code, 8) ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
  float alpha_val = 1.0;
  float beta_val = 0.0;
  void* alpha = &alpha_val;
  void* beta = &beta_val;

  cudaDataType_t d_type = CUDA_R_16F;
  if (TypeMatch(D->dtype, kE4M3Float_code, 8)) {
    d_type = CUDA_R_8F_E4M3;
  } else if (TypeMatch(D->dtype, kE5M2Float_code, 8)) {
    d_type = CUDA_R_8F_E5M2;
  } else ICHECK(TypeMatch(D->dtype, kDLFloat, 16));
    
  int M = ColumnCount(B, transb);
  int N = RowCount(A, transa);
  int K = ColumnCount(A, transa);
  int lda = transb ? K : M;
  int ldb = transa ? N : K;
  int ldd = M;

  
  
  auto A_data = static_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = static_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto D_data = static_cast<void*>(static_cast<char*>(D->data) + D->byte_offset);
  auto A_scale_data = static_cast<void*>(static_cast<char*>(A_scale->data) + A_scale->byte_offset);
  auto B_scale_data = static_cast<void*>(static_cast<char*>(B_scale->data) + B_scale->byte_offset);
  auto D_scale_data = static_cast<void*>(static_cast<char*>(D_scale->data) + D_scale->byte_offset);
  
  cublasLtMatrixLayout_t A_desc, B_desc, D_desc;
  CHECK_CUBLAS_ERROR(
      cublasLtMatrixLayoutCreate(&A_desc, a_type, !transb ? M : K, !transb ? K : M, lda));
  CHECK_CUBLAS_ERROR(
      cublasLtMatrixLayoutCreate(&B_desc, b_type, !transa ? K : N, !transa ? N : K, ldb));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&D_desc, d_type, M, N, ldd));

  cublasLtMatmulDesc_t op_desc;
  cublasOperation_t op_transa = CUBLASBooleanToTranspose(transa);
  cublasOperation_t op_transb = CUBLASBooleanToTranspose(transb);

  CHECK_CUBLAS_ERROR(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &op_transb, sizeof(op_transb)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &op_transa, sizeof(op_transa)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
						    &A_scale_data, sizeof(float*)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
						    &B_scale_data, sizeof(float*)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
						    &D_scale_data, sizeof(float*)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
						    &fast_accum, sizeof(fast_accum)));
						    


  // TODO: other calls have different order, doing " alpha, B_data, A_Desc, A_data, B_desc "
  // TODO: passing in null for C and C_desc. Since beta=0, could use D for in-place update with no bias (other calls do this)
  CHECK_CUBLAS_ERROR(cublasLtMatmul(hdl, op_desc, alpha, A_data, A_desc, B_data, B_desc, beta, nullptr, nullptr, D_data, D_desc,
				    nullptr, nullptr, 0, stream));
  
  cublasLtMatmulDescDestroy(op_desc);
  cublasLtMatrixLayoutDestroy(A_desc);
  cublasLtMatrixLayoutDestroy(B_desc);
  cublasLtMatrixLayoutDestroy(D_desc);
}
  
inline void CallLtIgemm(TVMArgs args, TVMRetValue* ret, cublasLtHandle_t hdl, cudaStream_t stream) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
   bool transa = args[3];
  bool transb = args[4];
  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;
  int M = ColumnCount(B, transb);
  int N = RowCount(A, transa);
  int K = ColumnCount(A, transa);
  int N_out = ColumnCount(C, false);
  int m = M;
  int n = m;
  int k = m;
  int lda = M * K / (roundoff(K, 32) / 32);
  int ldb = K * N / (roundoff(K, 32) / 32);
  int ldc = M * N_out / (roundoff(N_out, 32) / 32);
  ICHECK_EQ(A->ndim, 2);
  ICHECK_EQ(B->ndim, 2);
  ICHECK_EQ(C->ndim, 2);

  ICHECK_EQ(ElementStride(A), 1);
  ICHECK_EQ(ElementStride(B), 1);
  ICHECK_EQ(ElementStride(C), 1);

  ICHECK(TypeEqual(A->dtype, B->dtype));
  ICHECK(TypeMatch(A->dtype, kDLInt, 8));
  ICHECK(TypeMatch(C->dtype, kDLInt, 32));

  ICHECK(CheckMixPrecisionType(A->dtype, C->dtype)) << "Unsupported data type";
  int32_t alpha = args.size() > 5 ? args[5] : 1;
  int32_t beta = args.size() > 6 ? args[6] : 0;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  auto A_data = reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<void*>(static_cast<char*>(C->data) + C->byte_offset);

  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasLtMatmulDesc_t operationDesc = nullptr;
#if CUDART_VERSION >= 11000
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
#else
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescCreate(&operationDesc, CUDA_R_32I));
#endif
  cublasOperation_t opTransA = CUBLASBooleanToTranspose(transa);
  cublasOperation_t opTransB = CUBLASBooleanToTranspose(transb);
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &opTransA, sizeof(opTransA)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &opTransB, sizeof(opTransB)));
  // Create descriptors for the original matrices
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, opTransA == CUBLAS_OP_N ? m : k,
                                                opTransA == CUBLAS_OP_N ? k : m, lda));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, opTransB == CUBLAS_OP_N ? k : n,
                                                opTransB == CUBLAS_OP_N ? n : k, ldb));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));

  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &order_COL32, sizeof(order_COL32)));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &order_COL32, sizeof(order_COL32)));

  CHECK_CUBLAS_ERROR(cublasLtMatmul(hdl, operationDesc, &alpha, B_data, Adesc, A_data, Bdesc, &beta,
                                    C_data, Cdesc, C_data, Cdesc, nullptr, nullptr, 0, stream));
}



#endif

inline void CallGemmEx(TVMArgs args, TVMRetValue* ret, cublasHandle_t hdl) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  ICHECK_EQ(A->ndim, 2);
  ICHECK_EQ(B->ndim, 2);
  ICHECK_EQ(C->ndim, 2);

  ICHECK_EQ(ElementStride(A), 1);
  ICHECK_EQ(ElementStride(B), 1);
  ICHECK_EQ(ElementStride(C), 1);

  ICHECK(TypeEqual(A->dtype, B->dtype));

  // C can never be transposed.
  ICHECK(!IsInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  ICHECK(CheckMixPrecisionType(A->dtype, C->dtype)) << "Unsupported data type";
  ICHECK(!TypeMatch(A->dtype, kDLInt, 8) || ColumnStride(A) % 4 == 0)
      << "leading dimension must divide 4 for int8 gemm";
  ICHECK(!TypeMatch(B->dtype, kDLInt, 8) || ColumnStride(B) % 4 == 0)
      << "leading dimension must divide 4 for int8 gemm";
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;

  cudaDataType_t cuda_in_type = GetCudaDataType(A->dtype);
  cudaDataType_t cuda_out_type = GetCudaDataType(C->dtype);
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  void *alpha_ptr = nullptr, *beta_ptr = nullptr;
  auto alpha_int = static_cast<int32_t>(alpha);
  auto beta_int = static_cast<int32_t>(beta);
  auto alpha_float = static_cast<float>(alpha);
  auto beta_float = static_cast<float>(beta);
  if (C->dtype.code == kDLInt) {
    alpha_ptr = &alpha_int;
    beta_ptr = &beta_int;
  } else if (C->dtype.code == kDLFloat) {
    alpha_ptr = &alpha_float;
    beta_ptr = &beta_float;
  }

  auto A_data = reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<void*>(static_cast<char*>(C->data) + C->byte_offset);

  CHECK_CUBLAS_ERROR(
      cublasGemmEx(hdl, CUBLASBooleanToTranspose(transb), CUBLASBooleanToTranspose(transa),
                   ColumnCount(B, transb), RowCount(A, transa), ColumnCount(A, transa), alpha_ptr,
                   B_data, cuda_in_type, ColumnStride(B), A_data, cuda_in_type, ColumnStride(A),
                   beta_ptr, C_data, cuda_out_type, ColumnStride(C), cuda_out_type, algo));
}

inline void CallBatchGemmEx(TVMArgs args, TVMRetValue* ret, cublasHandle_t hdl) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  ICHECK_EQ(A->ndim, 3);
  ICHECK_EQ(B->ndim, 3);
  ICHECK_EQ(C->ndim, 3);

  int batch_size = BatchCount3D(C);
  ICHECK_EQ(ElementStride3D(A), 1);
  ICHECK_EQ(ElementStride3D(B), 1);
  ICHECK_EQ(ElementStride3D(C), 1);

  ICHECK(TypeEqual(A->dtype, B->dtype));

  // C can never be transposed.
  ICHECK(!IsInPlaceTransposed3D(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed3D(A) ? !transa : transa;
  transb = IsInPlaceTransposed3D(B) ? !transb : transb;

  ICHECK(CheckMixPrecisionType(A->dtype, C->dtype, true)) << "Unsupported data type";
  ICHECK(!TypeMatch(A->dtype, kDLInt, 8) || ColumnStride3D(A) % 4 == 0)
      << "leading dimension must divide 4 for int8 gemm";
  ICHECK(!TypeMatch(B->dtype, kDLInt, 8) || ColumnStride3D(B) % 4 == 0)
      << "leading dimension must divide 4 for int8 gemm";
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;

  int A_stride = A->shape[1] * A->shape[2];
  int B_stride = B->shape[1] * B->shape[2];
  int C_stride = C->shape[1] * C->shape[2];

  // Broadcast A or B by changing its stride.
  int batch_size_a = BatchCount3D(A);
  int batch_size_b = BatchCount3D(B);
  if (batch_size_a != batch_size_b) {
    if (batch_size_a == 1) {
      A_stride = 0;
    } else if (batch_size_b == 1) {
      B_stride = 0;
    }
  } else {
    ICHECK_EQ(batch_size_a, batch_size);
    ICHECK_EQ(batch_size_b, batch_size);
  }

  cudaDataType_t cuda_in_type = GetCudaDataType(A->dtype);
  cudaDataType_t cuda_out_type = GetCudaDataType(C->dtype);
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  void *alpha_ptr = nullptr, *beta_ptr = nullptr;
  auto alpha_int = static_cast<int32_t>(alpha);
  auto beta_int = static_cast<int32_t>(beta);
  auto alpha_float = static_cast<float>(alpha);
  auto beta_float = static_cast<float>(beta);
  if (C->dtype.code == kDLInt) {
    alpha_ptr = &alpha_int;
    beta_ptr = &beta_int;
  } else if (C->dtype.code == kDLFloat) {
    alpha_ptr = &alpha_float;
    beta_ptr = &beta_float;
  }

  auto A_data = reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<void*>(static_cast<char*>(C->data) + C->byte_offset);
  CHECK_CUBLAS_ERROR(cublasGemmStridedBatchedEx(
      hdl, CUBLASBooleanToTranspose(transb), CUBLASBooleanToTranspose(transa),
      ColumnCount3D(B, transb), RowCount3D(A, transa), ColumnCount3D(A, transa), alpha_ptr, B_data,
      cuda_in_type, ColumnStride3D(B), B_stride, A_data, cuda_in_type, ColumnStride3D(A), A_stride,
      beta_ptr, C_data, cuda_out_type, ColumnStride3D(C), C_stride, batch_size, cuda_out_type,
      algo));
}

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cublas.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* C = args[2];

  CuBlasThreadEntry* entry_ptr = CuBlasThreadEntry::ThreadLocal();

  CUBLASTryEnableTensorCore(entry_ptr->handle);

  if (TypeEqual(A->dtype, C->dtype)) {
    ICHECK(TypeMatch(A->dtype, kDLFloat, 16) || TypeMatch(A->dtype, kDLFloat, 32) ||
           TypeMatch(A->dtype, kDLFloat, 64));
    
    if (TypeMatch(A->dtype, kDLFloat, 16))
      CallGemm(args, ret, CublasHgemmOp(entry_ptr->handle));
    else if (TypeMatch(A->dtype, kDLFloat, 32))
      CallGemm(args, ret, CublasSgemmOp(entry_ptr->handle));
    else
      CallGemm(args, ret, CublasDgemmOp(entry_ptr->handle));
  } else {
    CallGemmEx(args, ret, entry_ptr->handle);
  }
});

#if CUDART_VERSION >= 10010
TVM_REGISTER_GLOBAL("tvm.contrib.cublaslt.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];

  CuBlasThreadEntry* entry_ptr = CuBlasThreadEntry::ThreadLocal();

  CUBLASTryEnableTensorCore(entry_ptr->handle);

  if (TypeMatch(A->dtype, kDLInt, 8)) {
      // Specialized call for int8 single-batch
      ICHECK(TypeMatch(A->dtype, kDLInt, 8)) << "Expects dtype to be int8\n";
      cublasLtHandle_t ltHandle;
      CHECK_CUBLAS_ERROR(cublasLtCreate(&ltHandle));
      auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
      ICHECK(func != nullptr);
      cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
      CallLtIgemm(args, ret, ltHandle, stream);
      CHECK_CUBLAS_ERROR(cublasLtDestroy(ltHandle));
    }
    else {
      // Specialized call for FP8 single-batch
      ICHECK(CheckIsFP8(A->dtype) && CheckIsFP8(B->dtype)) << "Expects dtype to be fp8\n";
      cublasLtHandle_t ltHandle;
      CHECK_CUBLAS_ERROR(cublasLtCreate(&ltHandle));
      auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
      ICHECK(func != nullptr);
      cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
      CallLtFP8Gemm(args, ret, ltHandle, stream);
    }      
});
#endif  // CUDART_VERSION >= 10010

TVM_REGISTER_GLOBAL("tvm.contrib.cublas.batch_matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* C = args[2];

  CuBlasThreadEntry* entry_ptr = CuBlasThreadEntry::ThreadLocal();

  CUBLASTryEnableTensorCore(entry_ptr->handle);
  if (TypeEqual(A->dtype, C->dtype)) {
    ICHECK(TypeMatch(A->dtype, kDLFloat, 16) || TypeMatch(A->dtype, kDLFloat, 32) ||
           TypeMatch(A->dtype, kDLFloat, 64));

    if (TypeMatch(A->dtype, kDLFloat, 16))
      CallBatchGemm(args, ret, CublasHgemmBatchOp(entry_ptr->handle));
    else if (TypeMatch(A->dtype, kDLFloat, 32))
      CallBatchGemm(args, ret, CublasSgemmBatchOp(entry_ptr->handle));
    else
      CallBatchGemm(args, ret, CublasDgemmBatchOp(entry_ptr->handle));
  } else {
    CallBatchGemmEx(args, ret, entry_ptr->handle);
  }
});

}  // namespace contrib
}  // namespace tvm
