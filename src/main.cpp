#include "basis_choice.h"
#include <cstdio>
#include <fstream>

struct MtxData {
  Index nrows;
  Index ncols;
  Index nnz;

  std::vector<SparseVector> cols;
};

MtxData ReadColumns(const char *filename) {
  std::ifstream file(filename);

  Index nrows, ncols, nnz;
  file >> nrows;
  file >> ncols;
  file >> nnz;

  std::vector<SparseVector> cols(ncols);

  for (Index k = 0; k < nnz; k++) {
    Index i, j;
    Scalar v;

    file >> i;
    file >> j;
    file >> v;

    cols[j].PushBack(i, v);
  }

  file.close();

  return MtxData{nrows, ncols, nnz, cols};
}

void TestCols(const MtxData &cols) {}

const char *test_files[] = {"./test_data/PRIMAL1.mtx",
                            "./test_data/PRIMAL2.mtx", "./test_data/UBH1.mtx",
                            "./test_data/BOYD1.mtx"};

int main(int argc, const char *argv[]) {
  for (const char *filename : test_files) {
    MtxData data = ReadColumns(filename);
    Scalar sparsity = Scalar(data.nnz) / (data.nrows * data.ncols);

    LOG_INFO("\n\e[32mTesting %s\e[m\n", filename);
    LOG_INFO("nrows=%6d, ncols=%6d, nnz=%8d (%f%%)\n", data.nrows, data.ncols,
             data.nnz, sparsity * 100.0);

    // append an identity matrix
    data.cols.resize(data.ncols + data.nrows);
    for (Index i = 0; i < data.nrows; i++) {
      data.cols[data.ncols + i].PushBack(i, 1.0);
    }
    data.ncols = data.ncols + data.nrows;
    data.nnz += data.nrows;

    sparsity = Scalar(data.nnz) / (data.nrows * data.ncols);
    LOG_INFO(
        "nrows=%6d, ncols=%6d, nnz=%8d (%f%%) (after appending identity)\n",
        data.nrows, data.ncols, data.nnz, sparsity * 100.0);

    TestCols(data);
  }

  return 0;
}
