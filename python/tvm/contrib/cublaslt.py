# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""External function interface to cuBLASlt libraries."""
import tvm
from tvm import te


def matmul(lhs, rhs, transa=False, transb=False, n=0, m=0, dtype=None, A_scale=None, B_scale=None, D_scale=None, fast_accum=False):
    """Create an extern op that compute matrix mult of lhs and rhs with cuBLASLt.
    Quantized scale factors and fast_accum are only used for FP8 calls, and should be None for int8 calls.
    Quantized scale factors must be scalar (size-1) Tensors of float32 values.
    Batched matmul is supported in clubasLt, but not currently implemented in the integration.

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs
    A_scale: Tensor(float32)
        Scale value for tensor A
    B_scale: Tensor(float32)
        Scale value for tensor B
    D_scale: Tensor(float32)
        Scale value for tensor D
    fast_accum: bool
        If True, use fast (lower-precision) accumulation. If False, use slower (higher-precision) accumulation.

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    if n == 0:
        n = lhs.shape[1] if transa else lhs.shape[0]
    if m == 0:
        m = rhs.shape[0] if transb else rhs.shape[1]
    dtype = dtype if dtype is not None else lhs.dtype
    return te.extern(
        (n, m),
        [lhs, rhs, A_scale, B_scale, D_scale],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cublaslt.matmul", ins[0], ins[1], outs[0], transa, transb, ins[2], ins[3], ins[4], fast_accum
        ),
        dtype=dtype,
        name="C",
    )
