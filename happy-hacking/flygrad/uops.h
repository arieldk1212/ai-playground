#ifndef UOPS_H_
#define UOPS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// use DAG for the graph of the OPS -> Acts as the unified IR.
// Implement scheduler for the transformation of the ops into a linear dt that
// can fit the GPU kernel.

// How it works:
// The nn ops aren't running immediately, we lazily form them in to a tree or
// graph of UOps.
// Then the scheduler cuts them into subgraphs to create individual GPU kernels.
// Then we flatten in out into a linear instructions before being emitted as the
// actual accelerator code.

namespace uops {

struct Op {
  uint8_t id;
  std::function<void()> compute;
};

struct Node {
  Op* op;
};

struct Graph {
  std::vector<std::unique_ptr<Node>> nodes;
};

struct UOp {
  Graph* op_graph;
};

}  // namespace uops

#endif
