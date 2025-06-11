#include "basis_choice.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>

#define CHECK_DECOMPOSITION
#define LOG_TABLE(...) fprintf(stderr, __VA_ARGS__)

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

  std::vector<SparseVector> cols(ncols, SparseVector(nrows));

  for (Index k = 0; k < nnz; k++) {
    Index i, j;
    Scalar v;

    file >> i;
    file >> j;
    file >> v;

    cols[j].PushBack(i, v);
  }

  for (Index j = 0; j < ncols; j++) {
    cols[j].Sort();
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

struct TestReport {
  Scalar factor_time;
  Scalar core_sparsity;
  Scalar q99_deviation;
  Scalar avg_err = -1;
  Scalar max_err = -1;
  Scalar waste;
};

TestReport TestColsWithPriority(const MtxData &cols,
                                const std::vector<Scalar> &priority) {
  TestReport report;
  // for (Index c = 0; c < cols.ncols; c++) {
  //   LOG_INFO("col[%i]:", c);
  //   for (SvIterator el(cols.cols[c]); el; ++el) {
  //     LOG_INFO(" (%i, %f)", el.index(), el.value());
  //   }
  //   LOG_INFO("\n");
  // }

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
  basis_choice::BasisChoice choice(cols.nrows, cols.ncols);
  basis_choice::FactorizeResult r = choice.Factorize(mtx_cols, priority);
  const double elapsed = factorize_timer.Elapsed();
  LOG_INFO("Factorized in %f ms\n", elapsed / 1000.0);
  report.factor_time = elapsed / 1000;

  // if (cols.nrows * cols.ncols < 1e8) {
  //   const bool v = choice.CheckFactorization(cols.cols, 1e-10);
  //   LOG_INFO("v=%i\n", v);
  // } else {
  //   LOG_INFO("Skip check\n");
  // }

  basis_choice::BasisChoiceStats stats = choice.ComputeStats();
  stats.LogStats();
  report.core_sparsity = (stats.l1_sparse + stats.u_sparse) / 2 * 100;
  report.waste = (stats.allocated_size - stats.used_size) /
                 Scalar(stats.allocated_size) * 100;

  if (r == basis_choice::FactorizeResult::kSingular) {
    LOG_INFO("Singular\n");
    return report;
  }

  std::vector<Index> vector_order = choice.GetVectorOrder();

  // LOG_INFO("order:");
  // for (Index i = 0; i < cols.ncols; i++) {
  //   LOG_INFO(" %2i", vector_order[i]);
  // }
  // LOG_INFO("\n");

  std::sort(vector_order.begin(), vector_order.begin() + cols.nrows,
            [&](const Index &lhs, const Index &rhs) -> bool {
              return priority[lhs] > priority[rhs];
            });

  for (Scalar q : {0.5, 0.75, 0.9, 0.99, 1.0}) {
    const Index k = std::min(std::ceil(q * cols.nrows), Scalar(cols.nrows - 1));
    const Index p = -priority[vector_order[k]];
    const Index min_p = k;
    const Index max_p = cols.ncols - cols.nrows + k;

    LOG_INFO("q=%5.1f%% (%i), priority=%7i, deviation=%7i (<=%7i, %6.2f%%)\n",
             q * 100.0, k, p, p - min_p, max_p - min_p,
             double(p - min_p) / double(max_p - min_p) * 100.0);

    if (q == 0.99)
      report.q99_deviation = double(p - min_p) / double(max_p - min_p) * 100.0;
  }

#ifdef CHECK_DECOMPOSITION
  Timer check_timer;

  Scalar err_sum = 0.0;
  const Index kCheckRuns = 10;

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

  LOG_INFO("check took %fms\n", check_timer.Elapsed() / 1000);

  std::sort(
      errors.begin(), errors.end(),
      [](const Scalar &lhs, const Scalar &rhs) -> bool { return lhs < rhs; });

  LOG_INFO("Solve error: average=%11.4e, "
           "q0=%11.4e, q1=%11.4e, q2=%11.4e, q3=%11.4e, q4=%11.4e\n",
           err_sum / (3 * kCheckRuns), errors[0],
           errors[3 * kCheckRuns * 1 / 4], errors[3 * kCheckRuns * 2 / 4],
           errors[3 * kCheckRuns * 3 / 4], errors[3 * kCheckRuns - 1]);

  report.avg_err = err_sum / (3 * kCheckRuns);
  report.max_err = errors[3 * kCheckRuns - 1];
#endif

  return report;
}

struct TestCase {
  const char *filename;

  std::vector<Scalar> priority;

  TestCase(const char *filename_) : filename(filename_), priority() {}

  TestCase(const char *filename_, const std::vector<Scalar> &priority_)
      : filename(filename_), priority(priority_) {}
};

int main(int, const char *[]) {
  rng = std::default_random_engine(42);

  const std::vector<TestCase> tests = {
      TestCase("./test_data/sanity1.mtx"),
      TestCase("./test_data/sanity2.mtx"),
      TestCase("./test_data/sanity3.mtx"),
      TestCase("./test_data/sanity4.mtx", {0.05, 0.05, 0.85, 0.05}),
      TestCase("./test_data/PRIMAL1.mtx"),
      TestCase("./test_data/PRIMAL2.mtx"),
      TestCase("./test_data/PRIMAL3.mtx"),
      TestCase("./test_data/PRIMAL4.mtx"),
      TestCase("./test_data/AUG3D.mtx"),
      TestCase("./test_data/UBH1.mtx"),
      TestCase("./test_data/CONT-100.mtx"),
      TestCase("./test_data/CONT-101.mtx"),
      TestCase("./test_data/CONT-200.mtx"),
      TestCase("./test_data/CONT-201.mtx"),
      TestCase("./test_data/CONT-300.mtx"),
      TestCase("./test_data/BOYD1.mtx"),
      // TestCase("./test_data/BOYD2.mtx"),
  };

#ifdef LOG_TABLE
  LOG_TABLE("%25s; Run; Factor time, ms; Initial sparse, %%; Core sparse, %%; "
            "Q99 deviation, %%; Average error;  Max error; Waste, %%\n",
            "Test");
#endif

  for (const TestCase &test : tests) {
    const char *filename = test.filename;
    std::vector<Scalar> priority = test.priority;

    MtxData data = ReadColumns(filename);
    Scalar sparsity =
        Scalar(data.nnz) / (Scalar(data.nrows) * Scalar(data.ncols));

    LOG_INFO("\n\x1B[32mTesting %s\x1B[m\n", filename);
    LOG_INFO("nrows=%6d, ncols=%6d, nnz=%8d (%.4f%%)\n", data.nrows, data.ncols,
             data.nnz, sparsity * 100.0);

    // append an identity matrix
    data.cols.resize(data.ncols + data.nrows, SparseVector(data.nrows));
    for (Index i = 0; i < data.nrows; i++) {
      data.cols[data.ncols + i].PushBack(i, 1.0);
    }
    data.ncols = data.ncols + data.nrows;
    data.nnz += data.nrows;

    sparsity = Scalar(data.nnz) / (Scalar(data.nrows) * Scalar(data.ncols));
    LOG_INFO(
        "nrows=%6d, ncols=%6d, nnz=%8d (%.4f%%) (after appending identity)\n",
        data.nrows, data.ncols, data.nnz, sparsity * 100.0);

#ifdef LOG_TABLE
    auto process_report = [&](const Index k, const TestReport report) -> void {
      LOG_TABLE("%25s;%4d;%16.4f;%18.4f;%15.4f;%17.4f;%14.4e;%11.4e;%9.4f\n",
                filename, k, report.factor_time, sparsity * 100,
                report.core_sparsity, report.q99_deviation, report.avg_err,
                report.max_err, report.waste);
    };
#else
    auto process_report = [&](const Index, const TestReport) -> void {};
#endif

    if (priority.size() == 0) {
      priority.resize(data.ncols);

      for (Index i = 0; i < data.ncols; i++) {
        priority[i] = -i;
      }

      for (Index k = 0; k < 3; k++) {
        std::shuffle(priority.begin(), priority.end(), rng);

        LOG_INFO("\x1B[34mRandom  Priority %2i\x1B[m\n", k);
        TestReport report = TestColsWithPriority(data, priority);
        process_report(k, report);
      }
    } else {
      assert(Index(priority.size()) == data.ncols);
      LOG_INFO("\x1B[34mDefined Priority\x1B[m\n");
      TestReport report = TestColsWithPriority(data, priority);
      process_report(-1, report);
    }
  }

  return 0;
}
