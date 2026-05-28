#ifndef KERNEL_H_
#define KERNEL_H_

#define __HIP_PLATFORM_AMD__

#include <vector>

#include "vendor/hip/hip_runtime_api.h"

namespace math {

class KernelMath {
 public:
  KernelMath();

  void VectorAdd(const std::vector<double>& data);

 private:
};

}  // namespace math

#endif
