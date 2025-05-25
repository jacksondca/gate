#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <atomic>
#include <boost/dynamic_bitset.hpp>
#include <stack>

#include "hnswlib/hnswlib.h"

namespace efanna2e {

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices,
      unsigned *hop_counts,
      std::vector<unsigned> &eps) override;
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);
  void OptimizeGraph(float* data);

  void LoadKmeans(const char *filename);

  void LoadEps(float* eps_load);

  void LoadTwoIterIndex(hnswlib::HierarchicalNSW<float>* cos_index);

  unsigned get_opt_ep_with_opt_graph(const float* query);

  unsigned get_opt_ep(const float* query);

  void set_ep(int index);

  long get_counter() {
    return counter.load();
  }

  long get_navigate_counter() {
    return navigate_counter.load();
  }

  std::vector<long>& get_vertex_visit_counter() {
    return vertex_visit_counter;
  }

  public:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    std::vector<std::vector<float>> iter1_centroids;
    std::vector<std::vector<unsigned>> iter2_eps;
    std::vector<unsigned> ep_counter;
    unsigned iter1_num, iter2_num, dim;

    Index *initializer_ = nullptr;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_ = nullptr;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;
    std::atomic<long> counter;
    std::atomic<long> navigate_counter;

    const float* eps_data_;

    std::vector<long> vertex_visit_counter;

    hnswlib::HierarchicalNSW<float>* two_iter_index = nullptr;
};
}

#endif //EFANNA2E_INDEX_NSG_H
