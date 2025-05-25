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

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cout << "params: database_path, query_path, result_path, efc, efs, top_k" << std::endl;
        return -1;
    }

    std::string database_path = argv[1];
    std::string query_path = argv[2];
    std::string result_path = argv[3];
    unsigned efc = (unsigned) atoi(argv[4]);
    unsigned efs = (unsigned) atoi(argv[5]);
    unsigned top_k = (unsigned) atoi(argv[6]);

    float* data_load = NULL;
    unsigned points_num, dim;
    load_data(argv[1], data_load, points_num, dim);

    float* query_load = NULL;
    unsigned query_num, query_dim;
    load_data(argv[2], query_load, query_num, query_dim);
    assert(dim == query_dim);

    hnswlib::CosSpace space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, points_num, efs, efc);

    // build
    for (int i = 0; i < points_num; i++) {
        alg_hnsw->addPoint(data_load + i * dim, i);
    }

    // test
    for (int i = 0; i < query_num; i++) {
        auto result = alg_hnsw->searchKnn(query_load + i * dim, top_k);
    }

    int count = 0;
    for (int i = 0; i < points_num; i++) {
        auto result = alg_hnsw->searchKnn(data_load + i * dim, top_k);
        auto label = result.top().second;
        if (label == i) {
            count++;
        }
    }
    std::cout << "recall: " << 1. * count / points_num << '\n';


    return 0;
}