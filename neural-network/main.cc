#include "nn.h"

int main() {
  Topology topology;
  topology.topology = {
      3,
      2,
      1,
  };

  Net net(topology);
}
