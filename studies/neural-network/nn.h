#ifndef NN_H_
#define NN_H_

#include <cstdint>
#include <iostream>
#include <vector>

struct Topology {
  std::vector<uint8_t> topology;
};

struct Connection {
  double weight;
  double delta_weight;
};

class Neuron {
 public:
  explicit Neuron(uint8_t outputs) {}

 private:
  double output_;
  std::vector<Connection> output_weights_;
};

using Layer = std::vector<Neuron>;

class Net {
 public:
  explicit Net(const Topology& topology) {
    layers_.resize(static_cast<int>(topology.topology.size()));

    for (int i = 0; i < layers_.size(); ++i) {
      uint8_t num_outputs{0};
      if (i < layers_.size() - 1) {
        num_outputs = topology.topology[i + 1];
      }

      for (int j = 0; j <= topology.topology[i]; ++j) {
        layers_[j].emplace_back(Neuron{num_outputs});
      }
    }
  }

  void FeedFoward(const std::vector<double>& neurons) {}
  void BackPropogation(const std::vector<double>& neurons);
  [[nodiscard]] std::vector<double> GetResults() const;

 private:
  std::vector<Layer> layers_;
};

#endif
