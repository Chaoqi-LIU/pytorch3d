/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <tuple>
#include "utils/pytorch3d_cutils.h"



at::Tensor ChaoqiFarthestPointSamplingCuda(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& R,
    const at::Tensor& K,
    const at::Tensor& start_idxs);


at::Tensor ChaoqiFarthestPointSamplingCpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& R,
    const at::Tensor& K,
    const at::Tensor& start_idxs);


at::Tensor ChaoqiFarthestPointSampling(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& R,
    const at::Tensor& K,
    const at::Tensor& start_idxs) {
  if (points.is_cuda() || lengths.is_cuda() || R.is_cuda() || K.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(lengths);
    CHECK_CUDA(R);
    CHECK_CUDA(K);
    CHECK_CUDA(start_idxs);
    return ChaoqiFarthestPointSamplingCuda(points, lengths, R, K, start_idxs);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return ChaoqiFarthestPointSamplingCpu(points, lengths, R, K, start_idxs);
}