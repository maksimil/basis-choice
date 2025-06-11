#ifndef __BASIS_CHOICE_H__
#define __BASIS_CHOICE_H__

#include "sparse_vector.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef USE_EIGEN
#include <Eigen/Sparse>
#endif

// #define BASIS_CHOICE_CHECK_SOLVE

#define LOG_INFO(...) fprintf(stdout, __VA_ARGS__)

class Timer {
public:
  using Clock = std::chrono::high_resolution_clock;

  Timer() { this->start_time_ = Clock::now(); }

  double Elapsed() const {
    const Clock::time_point now = Clock::now();
    const std::chrono::duration<double, std::micro> duration =
        now - this->start_time_;
    return duration.count();
  }

private:
  Clock::time_point start_time_;
};

using SparseVector = SparseVectorT<Index, Scalar>;
using SvIterator = SparseVector::Iterator;

inline bool IsSorted(const SparseVector &x) {
  for (Index i = 1; i < x.NonZeros(); i++) {
    if (x.GetIndex()[i - 1] >= x.GetIndex()[i]) {
      return false;
    }
  }

  return true;
}

inline Scalar AllocatedMemory(const SparseVector &x) {
  return Scalar(x.data().capacity()) * (sizeof(Index) + sizeof(Scalar));
}

inline Scalar UsedMemory(const SparseVector &x) {
  return Scalar(x.data().size()) * (sizeof(Index) + sizeof(Scalar));
}

inline Index IndexMax(const SparseVector &x) {
  if (x.NonZeros() == 0) {
    return -1;
  }

  Index mx = x.GetIndex()[0];

  for (Index k = 1; k < x.NonZeros(); k++) {
    if (mx < x.GetIndex()[k]) {
      mx = x.GetIndex()[k];
    }
  }

  return mx;
}

namespace basis_choice {

// cutoff absolute value during factorization
constexpr Scalar kEps = 1e-12;

// tol for pivot choice
constexpr Scalar kPivotTol = 1e-3;

#ifdef USE_EIGEN
using EigenSparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>;

inline std::vector<SparseVector>
MatrixConvertFromEigen(const EigenSparseMatrix &mtx) {
  std::vector<SparseVector> cols(mtx.cols(), SpareSparseVector(mtx.rows()));

  for (Index k = 0; k < mtx.outerSize(); k++) {
    for (EigenSparseMatrix::InnerIterator it(mtx, k); it; ++it) {
      const Index i = it.row();
      const Index j = it.col();
      const Scalar v = it.value();
      cols[j].PushBack(i, v);
    }
  }

  return cols;
}

inline EigenSparseMatrix
MatrixConvertToEigen(const std::vector<SparseVector> &cols) {
  std::vector<Eigen::Triplet<Scalar>> triplets;
  Index nrows = 0;

  for (Index j = 0; j < Index(cols.size()); j++) {
    for (SvIterator el(cols[j]); el; ++el) {
      triplets.emplace_back(el.index(), j, el.value());
      nrows = std::max(nrows, el.index());
    }
  }

  nrows += 1;

  EigenSparseMatrix mtx(nrows, cols.size());
  mtx.setFromTriplets(triplets.begin(), triplets.end());
  return mtx;
}
#endif

inline std::vector<SparseVector>
ComputeRowRepresentation(const std::vector<SparseVector> &cols, Index nrows) {
  std::vector<SparseVector> rows(nrows, SparseVector(cols.size()));

  for (Index ic = 0; ic < Index(cols.size()); ic++) {
    for (SvIterator el(cols[ic]); el; ++el) {
      rows[el.index()].PushBack(ic, el.value());
    }
  }

  return rows;
}

// Permutation coded as a function p such that
// P e_i = e_p(i)
class Permutation {
public:
  Permutation() {};

  Permutation(const Index &dimension)
      : permutation_(dimension), inverse_permutation_(dimension) {}

  Index Dimension() const { return this->permutation_.size(); }

  void Set(const Index &i, const Index &pi) {
    this->permutation_[i] = pi;
    this->inverse_permutation_[pi] = i;
  }

  void SetIdentity() {
    for (Index k = 0; k < this->Dimension(); k++) {
      this->Set(k, k);
    }
  }

  Index Permute(const Index &i) const { return this->permutation_[i]; }
  Index Inverse(const Index &i) const { return this->inverse_permutation_[i]; }

  std::vector<Index> &GetPermutation() { return this->permutation_; }
  const std::vector<Index> &GetPermutation() const {
    return this->permutation_;
  }

  std::vector<Index> &GetInverse() { return this->inverse_permutation_; }
  const std::vector<Index> &GetInverse() const {
    return this->inverse_permutation_;
  }

  void RestoreInverse() {
    const Index dimension = this->Dimension();
    for (Index k = 0; k < dimension; k++) {
      this->inverse_permutation_[this->permutation_[k]] = k;
    }
  }

  void RestorePermutation() {
    const Index dimension = this->Dimension();
    for (Index k = 0; k < dimension; k++) {
      this->permutation_[this->inverse_permutation_[k]] = k;
    }
  }

  void Swap(const Index &i, const Index &j) {
    Index pi = this->Permute(i), pj = this->Permute(j);
    this->Set(i, pj);
    this->Set(j, pi);
  }

  void ConsistentSet(const Index &i, const Index &pi) {
    this->Swap(i, this->Inverse(pi));
  }

  void AssertIntegrity() const {
#ifndef NDEBUG
    for (Index k = 0; k < this->Dimension(); k++) {
      const Index pk = this->Permute(k);
      assert(k == this->Inverse(pk));
    }
#endif
  }

private:
  std::vector<Index> permutation_;
  std::vector<Index> inverse_permutation_;
};

// x' = P x
inline void PermuteDense(std::vector<Scalar> &v, std::vector<Scalar> &swap,
                         const std::vector<Index> &permutation) {
  swap.resize(v.size());

  for (Index k = 0; k < Index(v.size()); k++) {
    std::swap(swap[permutation[k]], v[k]);
  }

  std::swap(swap, v);

  swap.clear();
}

inline void PermuteSparse(SparseVector &v,
                          const std::vector<Index> &permutation) {
  for (Index i = 0; i < v.NonZeros(); i++) {
    v.GetIndex()[i] = permutation[v.GetIndex()[i]];
  }
}

class SharedMemory {
public:
  // n = max(dimension, nvectors)
  // dirty array - capacity n, size 0, values undefined
  // clean array - capactity n, size n, values specified

  SharedMemory(Index dimension, Index nvectors) {
    const Index n = std::max(dimension, nvectors);
    this->dirty_index_.reserve(n);
    this->dirty_scalar_.reserve(n);
    this->clean_index_.resize(n, -1);
    this->clean_scalar_.resize(n, kNaN);

#ifndef NDEBUG
    this->n = n;
#endif
  }

  void AssertClean() const {
#ifndef NDEBUG
    assert(this->dirty_index_.size() == 0);
    assert(this->dirty_scalar_.size() == 0);
    assert(this->dirty_index_.capacity() >= this->n);
    assert(this->dirty_scalar_.capacity() >= this->n);

    assert(this->clean_index_.size() == this->n);
    for (Index k = 0; k < this->n; k++) {
      assert(this->clean_index_[k] == -1);
    }

    assert(this->clean_scalar_.size() == this->n);
    for (Index k = 0; k < this->n; k++) {
      assert(std::isnan(this->clean_scalar_[k]));
    }
#endif
  }

#ifndef NDEBUG
  Scalar n;
#endif

  std::vector<Scalar> dirty_scalar_;
  std::vector<Index> dirty_index_;
  std::vector<Scalar> clean_scalar_; // value = kNan
  std::vector<Index> clean_index_;   // value = -1
};

enum FactorizeResult {
  kOk,
  kSingular,
};

struct BasisChoiceStats {
  Index l_nnz = 0;
  Scalar l_sparse = 0.0;

  Index l1_nnz = 0;
  Scalar l1_sparse = 0.0;

  Index u_nnz = 0;
  Scalar u_sparse = 0.0;

  Index total_nnz = 0;
  Scalar total_sparse = 0.0;

  // memory is computed only for values_ and index_ vectors inside L and U
  // representations
  Scalar used_size = 0;
  Scalar allocated_size = 0;

  void LogStats() const {
    LOG_INFO("nnz(L1)   =%9i (%8.4f%%), nnz(L) =%9i (%8.4f%%), "
             "nnz(U) =%9i (%8.4f%%)\n"
             "nnz(L1+U) =%9i (%8.4f%%), nnz(F) =%9i (%8.4f%%)\n"
             "used =%10.4f Mb, allocated =%10.4f Mb, waste =%8.2f%%\n",
             this->l1_nnz, this->l1_sparse * 100.0, this->l_nnz,
             this->l_sparse * 100.0, this->u_nnz, this->u_sparse * 100.0,
             this->l1_nnz + this->u_nnz,
             (this->l1_sparse + this->u_sparse) / 2.0 * 100.0, this->total_nnz,
             this->total_sparse * 100.0, this->used_size / Scalar(1024 * 1024),
             this->allocated_size / Scalar(1024 * 1024),
             (this->allocated_size - this->used_size) /
                 Scalar(this->allocated_size) * 100);
  }
};

class BasisChoice {
public:
  BasisChoice(Index dimension, Index nvectors)
      : dimension_(dimension), nvectors_(nvectors), row_permutation_(nvectors),
        col_permutation_(dimension), lrows_(nvectors, SparseVector(dimension)),
        lcols_head_(dimension, SparseVector(nvectors)),
        lcols_tail_(dimension, SparseVector(nvectors)),
        urows_(dimension, SparseVector(dimension)), ucols_(dimension),
        udiagonal_(dimension, 0.0), shared_(dimension, nvectors) {}

  // factorize in terms of input vectors
  // priority is defined for each col (smaller value = higher priority)
  FactorizeResult Factorize(const std::vector<SparseVector> &vectors,
                            const std::vector<Scalar> &priority) {
    const std::vector<SparseVector> row_rep =
        ComputeRowRepresentation(vectors, this->dimension_);
    return this->FactorizeCT(vectors, row_rep, priority);
  }

  // factorize in terms of rows of input vectors
  FactorizeResult FactorizeCT(const std::vector<SparseVector> &ct_rows,
                              const std::vector<SparseVector> &ct_cols,
                              const std::vector<Scalar> &priority) {
    // compute col permutation
    this->ComputeQ(ct_rows, ct_cols, priority);

    // compute factorization
    const FactorizeResult r = this->ComputeLU(ct_rows, ct_cols, priority);

    return r;
  }

  void ComputeQ(const std::vector<SparseVector> &ct_rows,
                const std::vector<SparseVector> &ct_cols,
                const std::vector<Scalar> &priority);

  FactorizeResult ComputeLU(const std::vector<SparseVector> &ct_rows,
                            const std::vector<SparseVector> &ct_cols,
                            const std::vector<Scalar> &priority);

  // An array of indices with basis in the beginning:
  // - First m indices are basis vectors
  // - Rest n-m are non-basis
  // m - dimension, n - nvectors
  const std::vector<Index> &GetVectorOrder() const {
    return this->row_permutation_.GetInverse();
  }

  // solve C x' = x
  void SolveInPlace(std::vector<Scalar> &x) const {
    // C' x' = x
    this->SolveBasisInPlace(x);

    // padding (x')^T = (x^T 0 ... 0)
    x.resize(this->nvectors_, 0.0);

    // x' = P^T x
    PermuteDense(x, this->shared_.dirty_scalar_,
                 this->row_permutation_.GetInverse());
  }

  std::vector<Scalar> Solve(const std::vector<Scalar> &y) const {
    std::vector<Scalar> x = y;
    this->SolveInPlace(x);
    return x;
  }

  // solve C' x' = x
  void SolveBasisInPlace(std::vector<Scalar> &x) const {
    // x' = Q^T x
    PermuteDense(x, this->shared_.dirty_scalar_,
                 this->col_permutation_.GetInverse());

    // U^T x' = x
    for (Index i = 0; i < this->dimension_; i++) {
      const Scalar product = this->ucols_[i] * x;
      x[i] = (x[i] - product) / this->udiagonal_[i];
    }

    // L_1^T x' = x
    for (Index ii = 0; ii < this->dimension_; ii++) {
      const Index i = this->dimension_ - 1 - ii;
      const Scalar product = this->lcols_head_[i] * x;
      x[i] = x[i] - product;
    }
  }

  // checking L U = P C^T Q
  bool CheckFactorization(const std::vector<SparseVector> &vectors,
                          Scalar tol = kEps) const;
  BasisChoiceStats ComputeStats() const;

private:
  // L U = P C^T Q
  // L = (L_1^T L_2^T)^T
  // C' = Q U^T L_1^T
  // L_1 diagonal is unit
  // L_1 is lower head
  // L_2 is lower tail

  // dimensions
  Index dimension_; // ncols
  Index nvectors_;  // nrows

  // P, Q permutation matrices
  Permutation row_permutation_; // P
  Permutation col_permutation_; // Q

  // L, U factors
  std::vector<SparseVector> lrows_;
  std::vector<SparseVector> lcols_head_;
  std::vector<SparseVector> lcols_tail_;
  std::vector<SparseVector> urows_;
  std::vector<SparseVector> ucols_;
  std::vector<Scalar> udiagonal_;

  // shared memory for solves
  mutable SharedMemory shared_;
};

// set v[j], v[p] <- 0, v[j]
// if j=p, v[j] <- 0
// v.index[0] >= j
inline void LUTailSwapRemoves(std::vector<SparseVector> &tail,
                              const SparseVector &jrow,
                              const SparseVector &prow, Index p) {
  assert(IsSorted(jrow));
  assert(IsSorted(prow));

  Index jmem = 0;
  Index pmem = 0;

  while (jmem < jrow.NonZeros() && pmem < prow.NonZeros()) {
    const Index jidx = jrow.GetIndex()[jmem];
    const Index pidx = prow.GetIndex()[pmem];

    const Index midx = std::min(jidx, pidx);

    SparseVector &tailcol = tail[midx];
    assert(IsSorted(tailcol));

    const Index mem_p =
        std::lower_bound(tailcol.GetIndex(),
                         tailcol.GetIndex() + tailcol.NonZeros(), p) -
        tailcol.GetIndex();

    if (jidx == pidx) {
      assert(mem_p < tailcol.NonZeros());
      assert(tailcol.GetIndex()[mem_p] == p);
      std::swap(tailcol.GetValues()[0], tailcol.GetValues()[mem_p]);
      tailcol.Erase(SvIterator(tailcol) + 0);
      jmem++;
      pmem++;
    } else if (jidx < pidx) {
      const Scalar value = tailcol.GetValues()[0];

      for (Index k = 0; k < mem_p - 1; k++) {
        tailcol.GetValues()[k] = tailcol.GetValues()[k + 1];
        tailcol.GetIndex()[k] = tailcol.GetIndex()[k + 1];
      }

      tailcol.GetValues()[mem_p - 1] = value;
      tailcol.GetIndex()[mem_p - 1] = p;
      jmem++;
    } else {
      assert(pidx < jidx);
      tailcol.Erase(SvIterator(tailcol) + mem_p);
      pmem++;
    }
  }

  if (jmem < jrow.NonZeros()) {
    assert(pmem == prow.NonZeros());
    while (jmem < jrow.NonZeros()) {
      const Index jidx = jrow.GetIndex()[jmem];
      SparseVector &tailcol = tail[jidx];
      assert(IsSorted(tailcol));

      const Index mem_p =
          std::lower_bound(tailcol.GetIndex(),
                           tailcol.GetIndex() + tailcol.NonZeros(), p) -
          tailcol.GetIndex();

      const Scalar value = tailcol.GetValues()[0];

      for (Index k = 0; k < mem_p - 1; k++) {
        tailcol.GetValues()[k] = tailcol.GetValues()[k + 1];
        tailcol.GetIndex()[k] = tailcol.GetIndex()[k + 1];
      }

      tailcol.GetValues()[mem_p - 1] = value;
      tailcol.GetIndex()[mem_p - 1] = p;
      jmem++;
    }
  } else {
    assert(jmem == jrow.NonZeros());

    while (pmem < prow.NonZeros()) {
      const Index pidx = prow.GetIndex()[pmem];
      SparseVector &tailcol = tail[pidx];
      assert(IsSorted(tailcol));

      const Index mem_p =
          std::lower_bound(tailcol.GetIndex(),
                           tailcol.GetIndex() + tailcol.NonZeros(), p) -
          tailcol.GetIndex();
      tailcol.Erase(SvIterator(tailcol) + mem_p);
      pmem++;
    }
  }
}

inline void LUBVectorCompute(SparseVector &b_vector,
                             const std::vector<SparseVector> &lcols_tail,
                             const SparseVector &ucol, SharedMemory &shared) {
  shared.AssertClean();

  std::vector<Index> &memory_index = shared.clean_index_;

  for (Index k = 0; k < b_vector.NonZeros(); k++) {
    memory_index[b_vector.GetIndex()[k]] = k;
  }

  for (SvIterator u_el(ucol); u_el; ++u_el) {
    // assert(u_el.index() < j);

    for (SvIterator l_el(lcols_tail[u_el.index()]); l_el; ++l_el) {
      const Index i = l_el.index();
      const Scalar v = u_el.value() * l_el.value();

      // assert(i >= j);

      if (memory_index[i] == -1) {
        memory_index[i] = b_vector.NonZeros();
        b_vector.PushBack(i, -v);
      } else {
        b_vector.GetValues()[memory_index[i]] -= v;
      }
    }
  }

  for (Index k = 0; k < b_vector.NonZeros(); k++) {
    memory_index[b_vector.GetIndex()[k]] = -1;
  }

  b_vector.Erase(kEps);
  b_vector.Sort();
  shared.AssertClean();
}

// Solve L x' = x where L is defined by lcols and lrows coincides with lcols
// only in first n rows, where x and lcols are nonzero.
inline void LUFTranL(const std::vector<SparseVector> &lcols,
                     const std::vector<SparseVector> &lrows, SparseVector &x,
                     SharedMemory &shared) {
#ifndef NDEBUG
#ifdef BASIS_CHOICE_CHECK_SOLVE
  const std::vector<Scalar> original_x_dense = x.ToDense();
  assert(original_x_dense.size() == lcols.size());
#endif

  Index n = IndexMax(x);

  for (Index k = 0; k < Index(lcols.size()); k++) {
    n = std::max(n, IndexMax(lcols[k]));
  }

  assert(n < Index(lcols.size()));
#endif

  shared.AssertClean();
  std::vector<Index> &memory_index = shared.clean_index_;
  std::vector<Index> &nodes_stack = shared.dirty_index_;

  // fill-in computation
  for (Index k = 0; k < x.NonZeros(); k++) {
    memory_index[x.GetIndex()[k]] = k;
    nodes_stack.push_back(x.GetIndex()[k]);
  }

  const Index initial_stack_size = nodes_stack.size();

  Index jj = 0;
  while (jj < Index(nodes_stack.size())) {
    const Index j = nodes_stack[jj];

    for (SvIterator el(lcols[j]); el; ++el) {
      const Index i = el.index();
      assert(i <= n);
      assert(i > j);

      if (memory_index[i] == -1) {
        memory_index[i] = nodes_stack.size();
        nodes_stack.push_back(i);
      }
    }

    jj++;
  }

  x.Reserve(nodes_stack.size());
  for (Index jj = initial_stack_size; jj < Index(nodes_stack.size()); jj++) {
    x.PushBack(nodes_stack[jj], 0.0);
  }
  assert(x.NonZeros() == Index(nodes_stack.size()));

  nodes_stack.clear();

  x.Sort();

  // compute the values
  for (Index k = 0; k < x.NonZeros(); k++) {
    const Index i = x.GetIndex()[k];

    memory_index[i] = k;
    Scalar &xi = x.GetValues()[k];

    for (SvIterator el(lrows[i]); el; ++el) {
      assert(el.index() < i);
      assert(memory_index[el.index()] < k);

      if (memory_index[el.index()] != -1) {
        xi -= el.value() * x.GetValues()[memory_index[el.index()]];
      }
    }
  }

  // clean up
  for (Index k = 0; k < x.NonZeros(); k++) {
    memory_index[x.GetIndex()[k]] = -1;
  }

  x.Erase(kEps);
  shared.AssertClean();

  assert(x.Size() == Index(lcols.size()));

  // check
#ifndef NDEBUG
#ifdef BASIS_CHOICE_CHECK_SOLVE
  const std::vector<Scalar> result_x_dense = x.ToDense();

  // first n rows
  for (Index i = 0; i <= n; i++) {
    assert(lrows[i].Size() == Index(lcols.size()));
    const Scalar err =
        original_x_dense[i] - (lrows[i] * result_x_dense + result_x_dense[i]);
    if (std::abs(err) > kEps) {
      LOG_INFO("LUFTranL err i=%7i, err=%11.4e\n", i, err);
    }
  }

  // the rest
  for (Index i = n + 1; i < Index(lcols.size()); i++) {
    assert(result_x_dense[i] == original_x_dense[i]);
  }
#endif
#endif
}

// Performs LU decomposition with partial pivoting according to [1]. Comments
// reference pseudocode the algorithm in [1] as "the algorithm".
//
// [1] J. R. Gilbert and T. Peierls, "Sparse Partial Pivoting in Time
// Proportional to Arithmetic Operations," SIAM Journal on Scientific and
// Statistical Computing, vol. 9, no. 5, pp. 862-874, 1988,
// doi: 10.1137/0909058.
inline FactorizeResult
BasisChoice::ComputeLU(const std::vector<SparseVector> &,
                       const std::vector<SparseVector> &ct_cols,
                       const std::vector<Scalar> &priority) {
  // --- this function assumes ---
  // row_permutation_ is any
  // col_permutation_ contains the final permutation
  // lrows_ is final size with empty elements
  // lcols_head_ is final size with empty elements
  // lcols_tail_ is final size with empty elements
  // urows_ is final size with empty elements
  // ucols_ is final size with empty elements
  // udiagonal_ is final size with zeros
  // -----------------------------

  this->row_permutation_.SetIdentity();

  // --- reserve memory ---
  SparseVector b_vector(this->nvectors_);
  b_vector.Reserve(this->nvectors_);

  for (Index j = 0; j < this->dimension_; j++) {
    const Index reserve_size =
        ct_cols[this->col_permutation_.Permute(j)].NonZeros();

    this->lcols_head_[j].Reserve(reserve_size);
    this->lcols_tail_[j].Reserve(reserve_size);
  }

  for (Index j = 0; j < this->dimension_; j++) {
    // --- get and permute the next column ---

    // get the next column according to the column permutation
    SparseVector &upper_col = this->ucols_[j];
    upper_col = ct_cols[this->col_permutation_.Permute(j)];

    // applying already used partial pivoting row interchanges
    PermuteSparse(upper_col, this->row_permutation_.GetPermutation());
    upper_col.Sort();

    // --- split off and compute the upper part ---

    // split off the b_vector (b_vector is the b in the algorithm)
    Index upper_size = 0;
    while (upper_size < upper_col.NonZeros() &&
           upper_col.GetIndex()[upper_size] < j) {
      upper_size++;
    }

    for (Index k = upper_size; k < upper_col.NonZeros(); k++) {
      b_vector.PushBack(upper_col.GetIndex()[k], upper_col.GetValues()[k]);
    }

    upper_col.data().resize(upper_size);
    // TODO: Inner loop of ConservativeResize is redundant here
    upper_col.ConservativeResize(this->dimension_);
    assert(upper_col.Size() == this->dimension_);

    // line 3 of the algorithm (computing the upper column)
#ifndef NDEBUG
    for (Index i = 0; i < this->nvectors_; i++) {
      assert(this->lrows_[i].Size() == this->dimension_);
    }
#endif
    assert(Index(this->lcols_head_.size()) == this->dimension_);
    assert(Index(this->lrows_.size()) == this->nvectors_);
    LUFTranL(this->lcols_head_, this->lrows_, upper_col, this->shared_);

    // --- compute the b_vector (line 4 of the algorithm) ---

    LUBVectorCompute(b_vector, this->lcols_tail_, upper_col, this->shared_);

    // --- pivot choice (line 5 of the algorithm) ---

    // returning if singular
    if (b_vector.NonZeros() == 0) {
      return FactorizeResult::kSingular;
    }

    // pivot acceptability is defined by threshold partial pivoting, pivot is
    // chosen to be acceptable with the best priority
    Index mem_pivot;
    {
      // getting minimum absolute value of the pivot element
      Scalar b_tol = b_vector.GetValues()[0];
      for (Index k = 1; k < b_vector.NonZeros(); k++) {
        const Scalar b_abs = std::abs(b_vector.GetValues()[k]);
        if (b_abs > b_tol) {
          b_tol = b_abs;
        }
      }
      b_tol = b_tol * kPivotTol;

      // getting the first acceptable pivot
      mem_pivot = 0;
      for (Index k = 0; k < b_vector.NonZeros(); k++) {
        if (std::abs(b_vector.GetValues()[k]) > b_tol) {
          mem_pivot = k;
          break;
        }
      }

      // getting the best acceptable pivot
      for (Index k = mem_pivot + 1; k < b_vector.NonZeros(); k++) {
        if (std::abs(b_vector.GetValues()[k]) <= b_tol) {
          continue;
        }

        if (priority[this->row_permutation_.Inverse(b_vector.GetIndex()[k])] >
            priority[this->row_permutation_.Inverse(
                b_vector.GetIndex()[mem_pivot])]) {
          mem_pivot = k;
        }
      }
    }
    const Index pivot = b_vector.GetIndex()[mem_pivot];
    assert(pivot >= j);

    // --- split off the upper diagonal ---

    // line 6 of the algorithm (setting the diagonal)
    this->udiagonal_[j] = b_vector.GetValues()[mem_pivot];

    // removing the pivot element from b_vector
    if (b_vector.GetIndex()[0] == j) {
      std::swap(b_vector.GetValues()[0], b_vector.GetValues()[mem_pivot]);
      b_vector.Erase(SvIterator(b_vector) + 0);
    } else {
      assert(b_vector.GetIndex()[0] > j);
      b_vector.Erase(SvIterator(b_vector) + mem_pivot);
    }

    // --- update the row permutation (swap j and pivot) ---

    const Index inverse_pivot = this->row_permutation_.Inverse(pivot);
    const Index inverse_j = this->row_permutation_.Inverse(j);
    this->row_permutation_.GetPermutation()[inverse_pivot] = j;
    this->row_permutation_.GetPermutation()[inverse_j] = pivot;
    this->row_permutation_.GetInverse()[j] = inverse_pivot;
    this->row_permutation_.GetInverse()[pivot] = inverse_j;
    this->row_permutation_.AssertIntegrity();

    // --- update lower matrix ---

    // permute lower tail and remove pivot row from it
    LUTailSwapRemoves(this->lcols_tail_, this->lrows_[j], this->lrows_[pivot],
                      pivot);

    // permute lower rows
    std::swap(this->lrows_[j], this->lrows_[pivot]);

    // add the pivot row row to lower head
    for (SvIterator el(this->lrows_[j]); el; ++el) {
      assert(el.index() < j);
      this->lcols_head_[el.index()].PushBack(j, el.value());
    }

    // line 7 of the algorithm (add the new col to lower rows)
    for (SvIterator el(b_vector); el; ++el) {
      assert(el.index() > j);
      this->lrows_[el.index()].PushBack(j, el.value() / this->udiagonal_[j]);
      this->lcols_tail_[j].PushBack(el.index(),
                                    el.value() / this->udiagonal_[j]);
    }

    // --- cleanup ---

    b_vector.data().clear();
  }

  // --- compute upper rows ---
  std::vector<Index> &row_sizes = this->shared_.dirty_index_;
  row_sizes.resize(this->dimension_);

  for (Index i = 0; i < this->dimension_; i++) {
    row_sizes[i] = 0;
  }

  for (Index j = 0; j < this->dimension_; j++) {
    for (SvIterator el(this->ucols_[j]); el; ++el) {
      row_sizes[el.index()]++;
    }
  }

  for (Index i = 0; i < this->dimension_; i++) {
    this->urows_[i].Reserve(row_sizes[i]);
  }

  for (Index j = 0; j < this->dimension_; j++) {
    for (SvIterator el(this->ucols_[j]); el; ++el) {
      assert(el.index() < this->dimension_);
      this->urows_[el.index()].PushBack(j, el.value());
    }
  }
  row_sizes.clear();

  return FactorizeResult::kOk;
}

inline BasisChoiceStats BasisChoice::ComputeStats() const {
  BasisChoiceStats stats;

  const Scalar dimension = Scalar(this->dimension_);
  const Scalar nvectors = Scalar(this->nvectors_);

  // --- compute nnz ---
  const Scalar u_spaces = dimension * (dimension - 1) / 2;

  // U is from ucols
  for (Index i = 0; i < this->dimension_; i++) {
    // stats.u_nnz += i;
    stats.u_nnz += this->ucols_[i].NonZeros();
  }
  stats.u_sparse = Scalar(stats.u_nnz) / u_spaces;

  // L is from lrows
  for (Index i = 0; i < this->nvectors_; i++) {
    // stats.l_nnz += std::min(i, this->dimension_);
    stats.l_nnz += this->lrows_[i].NonZeros();
  }
  stats.l_sparse =
      Scalar(stats.l_nnz) / (nvectors * dimension - u_spaces - dimension);

  // L1 is from lcols
  for (Index i = 0; i < this->dimension_; i++) {
    stats.l1_nnz += this->lcols_head_[i].NonZeros();
  }
  stats.l1_sparse = Scalar(stats.l1_nnz) / u_spaces;

  stats.total_nnz = stats.u_nnz + stats.l_nnz + this->dimension_;
  stats.total_sparse = Scalar(stats.total_nnz) / (nvectors * dimension);

  // --- compute memory ---
  for (Index i = 0; i < this->nvectors_; i++) {
    stats.allocated_size += AllocatedMemory(this->lrows_[i]);
    stats.used_size += UsedMemory(this->lrows_[i]);
  }

  for (Index i = 0; i < this->dimension_; i++) {
    stats.allocated_size += AllocatedMemory(this->lcols_head_[i]) +
                            AllocatedMemory(this->lcols_tail_[i]) +
                            AllocatedMemory(this->ucols_[i]) +
                            AllocatedMemory(this->urows_[i]);

    stats.used_size +=
        UsedMemory(this->lcols_head_[i]) + UsedMemory(this->lcols_tail_[i]) +
        UsedMemory(this->ucols_[i]) + UsedMemory(this->urows_[i]);
  }

  return stats;
}

inline void NoOrdering(const std::vector<SparseVector> &,
                       const std::vector<SparseVector> &,
                       const std::vector<Scalar> &, Permutation &permutation) {
  permutation.SetIdentity();
}

inline void SimpleOrdering(const std::vector<SparseVector> &,
                           const std::vector<SparseVector> &cols,
                           const std::vector<Scalar> &priority,
                           Permutation &permutation) {
  const Index ncols = cols.size();
  std::vector<Scalar> col_metric(ncols);

  for (Index j = 0; j < ncols; j++) {
    const SparseVector &col = cols[j];

    col_metric[j] = priority[col.GetIndex()[0]];

    for (Index k = 0; k < col.NonZeros(); k++) {
      if (priority[col.GetIndex()[k]] > col_metric[j]) {
        col_metric[j] = priority[col.GetIndex()[k]];
      }
    }

    for (SvIterator el(col); el; ++el) {
      if (priority[el.index()] > col_metric[j]) {
        col_metric[j] = priority[el.index()];
      }
    }
  }

  permutation.SetIdentity();
  std::sort(permutation.GetPermutation().begin(),
            permutation.GetPermutation().end(),
            [&](const Index &lhs, const Index &rhs) -> bool {
              std::pair<Scalar, Scalar> lhs_pair, rhs_pair;

              lhs_pair = {-col_metric[lhs], cols[lhs].NonZeros()};
              rhs_pair = {-col_metric[rhs], cols[rhs].NonZeros()};

              return lhs_pair < rhs_pair;
            });
  permutation.RestoreInverse();
}

constexpr Index kMarkowitzNSections = 10;

inline void PriorityMarkowitzOrdering(const std::vector<SparseVector> &rows,
                                      const std::vector<SparseVector> &cols,
                                      const std::vector<Scalar> &priority,
                                      Permutation &permutation) {
  // index of the max priority row of the column
  std::vector<Index> col_max_priority(cols.size());
  std::vector<Index> row_size(rows.size(), 0);

  for (Index j = 0; j < Index(cols.size()); j++) {
    Index i = cols[j].GetIndex()[0];
    row_size[i] += 1;

    for (Index k = 1; k < cols[j].NonZeros(); k++) {
      const Index ip = cols[j].GetIndex()[k];
      row_size[ip] += 1;

      if (priority[i] < priority[ip]) {
        i = ip;
      }
    }

    col_max_priority[j] = i;
  }

  permutation.SetIdentity();

  std::vector<Index> &perm = permutation.GetPermutation();

  std::sort(perm.begin(), perm.end(),
            [&](const Index &lhs, const Index &rhs) -> bool {
              return priority[col_max_priority[lhs]] >
                     priority[col_max_priority[rhs]];
            });

  const Index section_size =
      (cols.size() + kMarkowitzNSections - 1) / kMarkowitzNSections;

  if (section_size == 1) {
    permutation.RestoreInverse();
    return;
  }

  for (Index k = 0; k * section_size < Index(cols.size()); k++) {
    const Index start = k * section_size;
    const Index end = std::min(Index(cols.size()), start + section_size);

    std::sort(perm.begin() + start, perm.begin() + end,
              [&](const Index &lhs, const Index &rhs) -> bool {
                const Index lhs_markowitz =
                    (cols[lhs].NonZeros() - 1) *
                    (row_size[col_max_priority[lhs]] - 1);
                const Index rhs_markowitz =
                    (cols[rhs].NonZeros() - 1) *
                    (row_size[col_max_priority[rhs]] - 1);
                return lhs_markowitz < rhs_markowitz;
              });
  }

  permutation.RestoreInverse();
}

struct COLAMDRowInfo {
  Index start = 0;
  Index end = 0;
};

struct COLAMDColInfo {
  Index start = 0;
  Index end = 0;

  Index score = 0;
  bool chosen = false;

  Index l_size = 0;
};

constexpr Index kMaxTag = std::numeric_limits<Index>::max() / 4;

inline void BasisChoice::ComputeQ(const std::vector<SparseVector> &ct_rows,
                                  const std::vector<SparseVector> &ct_cols,
                                  const std::vector<Scalar> &priority) {
  // NoOrdering(ct_rows, ct_cols, priority, this->col_permutation_);
  // SimpleOrdering(ct_rows, ct_cols, priority, this->col_permutation_);
  PriorityMarkowitzOrdering(ct_rows, ct_cols, priority, this->col_permutation_);
  // COLAMD(ct_rows, ct_cols, priority, this->col_permutation_);

  this->col_permutation_.AssertIntegrity();
}

} // namespace basis_choice

#endif
