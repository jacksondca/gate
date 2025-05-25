#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

#include <hnswlib/hnswlib.h>

void load_data(const char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  std::cout << "opening: " << filename << '\n';
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[num * dim * sizeof(float)];
  std::cout << "data dimension: " << dim << ", num: " << num << std::endl;

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(const char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path kmeans_path hop_count_path"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  auto s = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<unsigned> > res;
  res.reserve(query_num);
  std::vector<std::vector<unsigned> > hop_counts;
  long long pre = 0;
  hop_counts.reserve(query_num);
  for (unsigned i = 0; i < query_num; i++) {
    std::vector<unsigned> tmp(K);
    std::vector<unsigned> hop_count(K);
    std::vector<unsigned> eps;
    hop_count[0] = i;

    auto cos_eps = cos_index->searchKnn(query_load + i * dim, 3);
    while (!cos_eps.empty()) {
      unsigned ep_id = cos_eps.top().second;
      cos_eps.pop();
      
      unsigned ep = index.iter2_eps[ep_id / index.iter2_num][ep_id % index.iter2_num];
      eps.push_back(ep);
    }

    index.Search(query_load + i * dim, data_load, K, paras, tmp.data(), hop_count.data(), eps);

    res.push_back(tmp);
    hop_counts.push_back(hop_count);
  }

  save_result(argv[6], res);
  save_result(argv[8], hop_counts);
  
  return 0;
}
