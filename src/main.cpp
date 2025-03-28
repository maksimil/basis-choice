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

void TestCols(const MtxData &cols) {
  basis_choice::BasisChoice choice(cols.nrows, cols.ncols);
  const std::vector<Scalar> priority;

  Timer factorize_timer;
  choice.Factorize(cols.cols, priority);
  const double elapsed = factorize_timer.Elapsed();
  LOG_INFO("Factorized in %f ms\n", elapsed / 1000.0);

  if (cols.nrows * cols.ncols < 1e8) {
    const bool v = choice.CheckFactorization(cols.cols, 1e-8);
    LOG_INFO("v=%i\n", v);
  } else {
    LOG_INFO("Skip check\n");
  }

  choice.ComputeStats().LogStats();
}

const char *test_files[] = {
    "./test_data/sanity1.mtx", "./test_data/sanity2.mtx",
    "./test_data/sanity3.mtx", "./test_data/PRIMAL1.mtx",
    "./test_data/PRIMAL2.mtx", "./test_data/PRIMAL3.mtx",
    "./test_data/PRIMAL4.mtx", "./test_data/AUG3D.mtx",
    "./test_data/UBH1.mtx",
};

int main(int, const char *[]) {
  for (const char *filename : test_files) {
    MtxData data = ReadColumns(filename);
    Scalar sparsity = Scalar(data.nnz) / (data.nrows * data.ncols);

    LOG_INFO("\n\x1B[32mTesting %s\x1B[m\n", filename);
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
