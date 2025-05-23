#include "basis_choice.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>

#define CHECK_DECOMPOSITION

std::default_random_engine rng;

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

void Normalize1(std::vector<Scalar> &x) {
  Scalar x1 = 0.0;
  for (Index i = 0; i < Index(x.size()); i++) {
    x1 += std::abs(x[i]);
  }
  for (Index i = 0; i < Index(x.size()); i++) {
    x[i] /= x1;
  }
}

struct RhsCheck {
  Scalar err1 = 0;
};

RhsCheck CheckRhs(const MtxData &cols, const basis_choice::BasisChoice &choice,
                  const std::vector<Scalar> &rhs) {
  std::vector<Scalar> res = choice.Solve(rhs);

  std::vector<Scalar> mod_rhs(cols.nrows, 0.0);

  for (Index j = 0; j < cols.ncols; j++) {
    for (SvIterator el(cols.cols[j]); el; ++el) {
      mod_rhs[el.index()] += el.value() * res[j];
    }
  }

  RhsCheck check;

  for (Index i = 0; i < cols.nrows; i++) {
    check.err1 += std::abs(mod_rhs[i] - rhs[i]);
  }

  return check;
}

void TestColsWithPriority(const MtxData &cols,
                          const std::vector<Scalar> &priority) {
  basis_choice::BasisChoice choice(cols.nrows, cols.ncols);

#ifdef USE_EIGEN
  // Just to test conversions
  basis_choice::EigenSparseMatrix eigen_cols =
      basis_choice::MatrixConvertToEigen(cols.cols);
  const std::vector<SparseVector> &mtx_cols =
      basis_choice::MatrixConvertFromEigen(eigen_cols);
#else
  const std::vector<SparseVector> &mtx_cols = cols.cols;
#endif
  Timer factorize_timer;
  choice.Factorize(mtx_cols, priority);
  const double elapsed = factorize_timer.Elapsed();
  LOG_INFO("Factorized in %f ms\n", elapsed / 1000.0);

  // if (cols.nrows * cols.ncols < 1e8) {
  //   const bool v = choice.CheckFactorization(cols.cols, 1e-10);
  //   LOG_INFO("v=%i\n", v);
  // } else {
  //   LOG_INFO("Skip check\n");
  // }

  choice.ComputeStats().LogStats();

  std::vector<Index> vector_order = choice.GetVectorOrder();

  std::sort(vector_order.begin(), vector_order.begin() + cols.nrows,
            [&](const Index &lhs, const Index &rhs) -> bool {
              return priority[lhs] > priority[rhs];
            });

  for (Scalar q : {0.5, 0.75, 0.9, 0.99, 1.0}) {
    const Index k = std::ceil(q * cols.nrows) - 1;
    const Index p = -priority[vector_order[k]];
    const Index min_p = k;
    const Index max_p = cols.ncols - cols.nrows + k;

    LOG_INFO("q=%5.1f%%, priority=%7i, deviation=%7i (<=%7i, %6.2f%%)\n",
             q * 100.0, p, p - min_p, max_p - min_p,
             double(p - min_p) / double(max_p - min_p) * 100.0);
  }

#ifdef CHECK_DECOMPOSITION
  Scalar err_sum = 0.0;
  const Index kCheckRuns = 100;

  std::vector<Scalar> errors;
  errors.reserve(3 * kCheckRuns);

  for (Index k = 0; k < kCheckRuns; k++) {
    Scalar err;

    // LOG_INFO("k=%4d\n", k);
    std::vector<Scalar> rhs(cols.nrows);
    RhsCheck check;

    for (Index i = 0; i < cols.nrows; i++) {
      rhs[i] = std::exponential_distribution<Scalar>(1.0)(rng);
    }
    Normalize1(rhs);
    check = CheckRhs(cols, choice, rhs);
    err = check.err1;
    err_sum += err;
    errors.push_back(err);
    // LOG_INFO("exp(1):       err1=%.4e, relative=%.4e\n", check.err1,
    //          check.err1 / check.rhs1);

    for (Index i = 0; i < cols.nrows; i++) {
      rhs[i] = std::normal_distribution<Scalar>(0.0, 1.0)(rng);
    }
    Normalize1(rhs);
    check = CheckRhs(cols, choice, rhs);
    err = check.err1;
    err_sum += err;
    errors.push_back(err);
    // LOG_INFO("normal(0, 1): err1=%.4e, relative=%.4e\n", check.err1,
    //          check.err1 / check.rhs1);

    for (Index i = 0; i < cols.nrows; i++) {
      rhs[i] = std::normal_distribution<Scalar>(1.0, 1.0)(rng);
    }
    Normalize1(rhs);
    check = CheckRhs(cols, choice, rhs);
    err = check.err1;
    err_sum += err;
    errors.push_back(err);
    // LOG_INFO("normal(1, 1): err1=%.4e, relative=%.4e\n", check.err1,
    //          check.err1 / check.rhs1);
  }

  std::sort(
      errors.begin(), errors.end(),
      [](const Scalar &lhs, const Scalar &rhs) -> bool { return lhs < rhs; });

  LOG_INFO("Solve error: average=%11.4e, "
           "q0=%11.4e, q1=%11.4e, q2=%11.4e, q3=%11.4e, q4=%11.4e\n",
           err_sum / (3 * kCheckRuns), errors[0],
           errors[3 * kCheckRuns * 1 / 4], errors[3 * kCheckRuns * 2 / 4],
           errors[3 * kCheckRuns * 3 / 4], errors[3 * kCheckRuns - 1]);
#endif
}

void TestCols(const MtxData &cols) {
  std::vector<Scalar> priority(cols.ncols);

  for (Index i = 0; i < cols.ncols; i++) {
    priority[i] = -i;
  }

  for (Index k = 0; k < 3; k++) {
    std::shuffle(priority.begin(), priority.end(), rng);

    LOG_INFO("\x1B[34mPriority %2i\x1B[m\n", k);
    TestColsWithPriority(cols, priority);
  }
}

const char *test_files[] = {
    "./test_data/sanity1.mtx",  "./test_data/sanity2.mtx",
    "./test_data/sanity3.mtx",  "./test_data/PRIMAL1.mtx",
    "./test_data/PRIMAL2.mtx",  "./test_data/PRIMAL3.mtx",
    "./test_data/PRIMAL4.mtx",  "./test_data/AUG3D.mtx",
    "./test_data/UBH1.mtx",     "./test_data/CONT-100.mtx",
    "./test_data/CONT-101.mtx", "./test_data/CONT-200.mtx",
    "./test_data/CONT-201.mtx", "./test_data/CONT-300.mtx",
    "./test_data/BOYD1.mtx",
    // "./test_data/BOYD2.mtx"
};

int main(int, const char *[]) {
  rng = std::default_random_engine(42);

  for (const char *filename : test_files) {
    MtxData data = ReadColumns(filename);
    Scalar sparsity =
        Scalar(data.nnz) / (Scalar(data.nrows) * Scalar(data.ncols));

    LOG_INFO("\n\x1B[32mTesting %s\x1B[m\n", filename);
    LOG_INFO("nrows=%6d, ncols=%6d, nnz=%8d (%.4f%%)\n", data.nrows, data.ncols,
             data.nnz, sparsity * 100.0);

    // append an identity matrix
    data.cols.resize(data.ncols + data.nrows);
    for (Index i = 0; i < data.nrows; i++) {
      data.cols[data.ncols + i].PushBack(i, 1.0);
    }
    data.ncols = data.ncols + data.nrows;
    data.nnz += data.nrows;

    sparsity = Scalar(data.nnz) / (Scalar(data.nrows) * Scalar(data.ncols));
    LOG_INFO(
        "nrows=%6d, ncols=%6d, nnz=%8d (%.4f%%) (after appending identity)\n",
        data.nrows, data.ncols, data.nnz, sparsity * 100.0);

    TestCols(data);
  }

  return 0;
}
