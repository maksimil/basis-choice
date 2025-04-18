#ifndef __BASIS_CHOICE_H__
#define __BASIS_CHOICE_H__

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

using Scalar = double;
using Index = int32_t;
constexpr Scalar kNaN = std::numeric_limits<Scalar>::quiet_NaN();

class SparseVector {
public:
  SparseVector() : index_(), values_() {}

  SparseVector(Index index, Scalar value) : index_(), values_() {
    this->PushBack(index, value);
  }

  std::vector<Index> &GetIndex() { return this->index_; }
  const std::vector<Index> &GetIndex() const { return this->index_; }
  std::vector<Scalar> &GetValues() { return this->values_; }
  const std::vector<Scalar> &GetValues() const { return this->values_; }

  Index size() const {
    assert(index_.size() == values_.size());
    return Index(index_.size());
  }

  void PushBack(const Index index, const Scalar value) {
    index_.push_back(index);
    values_.push_back(value);
  }

  void Clear() {
    index_.clear();
    values_.clear();
  }

  void Resize(size_t n) {
    index_.resize(n);
    values_.resize(n);
  }

  void Reserve(size_t n) {
    index_.reserve(n);
    values_.reserve(n);
  }

  void Erase(size_t n) {
    index_.erase(index_.begin() + n);
    values_.erase(values_.begin() + n);
  }

  std::vector<Scalar> ToDense(size_t out_size) const;

  Scalar operator*(const SparseVector &x) const;
  Scalar operator*(const std::vector<Scalar> &x) const;

private:
  std::vector<Index> index_;
  std::vector<Scalar> values_;
};

inline bool IsSorted(const SparseVector &x) {
  for (Index i = 1; i < x.size(); i++) {
    if (x.GetIndex()[i - 1] >= x.GetIndex()[i]) {
      return false;
    }
  }

  return true;
}

inline Scalar AllocatedMemory(const SparseVector &x) {
  return Scalar(x.GetIndex().capacity()) * sizeof(Index) +
         Scalar(x.GetValues().capacity()) * sizeof(Scalar);
}

inline Scalar UsedMemory(const SparseVector &x) {
  return Scalar(x.GetIndex().size()) * sizeof(Index) +
         Scalar(x.GetValues().size()) * sizeof(Scalar);
}

inline Index IndexMax(const SparseVector &x) {
  if (x.size() == 0) {
    return -1;
  }

  Index mx = x.GetIndex()[0];

  for (Index k = 1; k < x.size(); k++) {
    if (mx < x.GetIndex()[k]) {
      mx = x.GetIndex()[k];
    }
  }

  return mx;
}

inline std::vector<Scalar> SparseVector::ToDense(size_t out_size) const {
  std::vector<Scalar> res(out_size, 0.0);
  for (Index i = 0; i < size(); ++i) {
    assert(index_[i] < Index(out_size));
    res[index_[i]] = values_[i];
  }
  return res;
}

inline Scalar SparseVector::operator*(const SparseVector &x) const {
  assert(IsSorted(*this));
  assert(IsSorted(x));

  Scalar res = 0.0;

  Index i_le = 0, i_ri = 0;
  while (i_le < this->size() and i_ri < x.size()) {
    auto pos_le = this->index_[i_le];
    auto pos_ri = x.index_[i_ri];
    if (pos_le < pos_ri) {
      i_le++;
      continue;
    } else if (pos_le > pos_ri) {
      i_ri++;
      continue;
    } else {
      res += this->values_[i_le] * x.values_[i_ri];
      i_le++;
      i_ri++;
    }
  }

  return res;
}

inline Scalar SparseVector::operator*(const std::vector<Scalar> &x) const {
  Scalar dot = 0.0;
  for (Index k = 0; k < size(); ++k) {
    dot += values_[k] * x[index_[k]];
  }
  return dot;
}

class SvIterator {
  // for (SvIterator el(vector); el; ++el) { ... }
public:
  inline explicit SvIterator(const SparseVector &v)
      : index_(v.GetIndex().data()), values_(v.GetValues().data()), ind_(0),
        end_(v.size()) {}

  inline SvIterator &operator++() {
    ind_++;
    return *this;
  }
  inline explicit operator bool() const { return (ind_ < end_); }

  // Access
  inline const Index &index() const { return index_[ind_]; }
  inline const Scalar &value() const { return values_[ind_]; }
  inline Scalar &valueRef() { return const_cast<Scalar &>(values_[ind_]); }

private:
  const Index *index_;
  const Scalar *values_;
  Index ind_;
  const Index end_;
};

namespace basis_choice {

// cutoff absolute value during factorization
constexpr Scalar kEps = 1e-12;

// tol for pivot choice
constexpr Scalar kPivotTol = 1e-3;

#ifdef USE_EIGEN
using EigenSparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>;

inline std::vector<SparseVector>
MatrixConvertFromEigen(const EigenSparseMatrix &mtx) {
  std::vector<SparseVector> cols(mtx.cols());

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
  std::vector<SparseVector> rows(nrows);

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

inline void SortSparse(SparseVector &v, std::vector<Index> &index_swap,
                       std::vector<Scalar> &scalar_swap) {
  index_swap.resize(2 * v.size());

  for (Index i = 0; i < v.size(); i++) {
    index_swap[i] = i;
  }

  std::sort(index_swap.begin(), index_swap.begin() + v.size(),
            [&](const Index &lhs, const Index &rhs) -> bool {
              return v.GetIndex()[lhs] < v.GetIndex()[rhs];
            });

  for (Index i = 0; i < v.size(); i++) {
    scalar_swap[i] = v.GetValues()[index_swap[i]];
    index_swap[v.size() + i] = v.GetIndex()[index_swap[i]];
  }

  for (Index i = 0; i < v.size(); i++) {
    v.GetValues()[i] = scalar_swap[i];
    v.GetIndex()[i] = index_swap[v.size() + i];
  }

#ifndef NDEBUG
  for (Index i = 1; i < v.size(); i++) {
    assert(v.GetIndex()[i - 1] < v.GetIndex()[i]);
  }
#endif

  scalar_swap.clear();
  index_swap.clear();
}

inline void PermuteSparse(SparseVector &v,
                          const std::vector<Index> &permutation) {
  for (Index i = 0; i < v.size(); i++) {
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
    LOG_INFO("nnz(L1) =%9i (%8.4f%%), nnz(L) =%9i (%8.4f%%)\n"
             "nnz(U)  =%9i (%8.4f%%), nnz(F) =%9i (%8.4f%%)\n"
             "used =%10.4f Mb, allocated =%10.4f Mb, waste =%8.2f%%\n",
             this->l1_nnz, this->l1_sparse * 100.0, this->l_nnz,
             this->l_sparse * 100.0, this->u_nnz, this->u_sparse * 100.0,
             this->total_nnz, this->total_sparse * 100.0,
             this->used_size / Scalar(1024 * 1024),
             this->allocated_size / Scalar(1024 * 1024),
             (this->allocated_size - this->used_size) /
                 Scalar(this->allocated_size) * 100);
  }
};

class BasisChoice {
public:
  BasisChoice(Index dimension, Index nvectors)
      : dimension_(dimension), nvectors_(nvectors), row_permutation_(nvectors),
        col_permutation_(dimension), lrows_(nvectors), lcols_head_(dimension),
        lcols_tail_(dimension), urows_(dimension), ucols_(dimension),
        udiagonal_(dimension, 0.0), shared_(dimension, nvectors) {}

  // factorize in terms of input vectors
  // priority is defined for each col (smaller value = higher priority)
  FactorizeResult Factorize(const std::vector<SparseVector> &vectors,
                            const std::vector<Scalar> &priority) {
    const std::vector<SparseVector> row_rep =
        ComputeRowRepresentation(vectors, this->dimension_);
    return this->FactorizeCT(row_rep, priority);
  }

  // factorize in terms of rows of input vectors
  FactorizeResult FactorizeCT(const std::vector<SparseVector> &ct_cols,
                              const std::vector<Scalar> &priority) {
    // compute col permutation
    this->ComputeQ(ct_cols, priority);

    // compute factorization
    const FactorizeResult r = this->ComputeLU(ct_cols, priority);

    return r;
  }

  // TODO: implement COLAMD or a modification
  void ComputeQ(const std::vector<SparseVector> &ct_cols,
                const std::vector<Scalar> &priority);

  FactorizeResult ComputeLU(const std::vector<SparseVector> &ct_cols,
                            const std::vector<Scalar> &priority);

  std::vector<Index> GetBasisVectors() const {
    std::vector<Index> idxs(this->dimension_);
    for (Index i = 0; i < this->dimension_; i++) {
      idxs[i] = this->row_permutation_.Inverse(i);
    }
    return idxs;
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

template <class F> inline void PruneVector(SparseVector &v, F condition) {
  Index l = 0;

  for (Index k = 0; k < v.size(); k++) {
    Scalar value = v.GetValues()[k];
    Index index = v.GetIndex()[k];

    if (!condition(index, value)) {
      v.GetValues()[l] = value;
      v.GetIndex()[l] = index;
      l++;
    }
  }

  v.Resize(l);
}

inline void PruneZeros(SparseVector &v, Scalar tol) {
  PruneVector(v, [&](Index, Scalar x) -> bool { return std::abs(x) < tol; });
}

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

  while (jmem < jrow.size() && pmem < prow.size()) {
    const Index jidx = jrow.GetIndex()[jmem];
    const Index pidx = prow.GetIndex()[pmem];

    const Index midx = std::min(jidx, pidx);

    SparseVector &tailcol = tail[midx];
    assert(IsSorted(tailcol));

    const Index mem_p = std::lower_bound(tailcol.GetIndex().begin(),
                                         tailcol.GetIndex().end(), p) -
                        tailcol.GetIndex().begin();

    if (jidx == pidx) {
      assert(mem_p < tailcol.size());
      assert(tailcol.GetIndex()[mem_p] == p);
      std::swap(tailcol.GetValues()[0], tailcol.GetValues()[mem_p]);
      tailcol.Erase(0);
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
      tailcol.Erase(mem_p);
      pmem++;
    }
  }

  if (jmem < jrow.size()) {
    assert(pmem == prow.size());
    while (jmem < jrow.size()) {
      const Index jidx = jrow.GetIndex()[jmem];
      SparseVector &tailcol = tail[jidx];
      assert(IsSorted(tailcol));

      const Index mem_p = std::lower_bound(tailcol.GetIndex().begin(),
                                           tailcol.GetIndex().end(), p) -
                          tailcol.GetIndex().begin();

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
    assert(jmem == jrow.size());

    while (pmem < prow.size()) {
      const Index pidx = prow.GetIndex()[pmem];
      SparseVector &tailcol = tail[pidx];
      assert(IsSorted(tailcol));

      const Index mem_p = std::lower_bound(tailcol.GetIndex().begin(),
                                           tailcol.GetIndex().end(), p) -
                          tailcol.GetIndex().begin();
      tailcol.Erase(mem_p);
      pmem++;
    }
  }
}

inline void LUBVectorCompute(SparseVector &b_vector,
                             const std::vector<SparseVector> &lcols_tail,
                             const SparseVector &ucol, SharedMemory &shared) {
  shared.AssertClean();

  std::vector<Index> &memory_index = shared.clean_index_;

  for (Index k = 0; k < b_vector.size(); k++) {
    memory_index[b_vector.GetIndex()[k]] = k;
  }

  for (SvIterator u_el(ucol); u_el; ++u_el) {
    // assert(u_el.index() < j);

    for (SvIterator l_el(lcols_tail[u_el.index()]); l_el; ++l_el) {
      const Index i = l_el.index();
      const Scalar v = u_el.value() * l_el.value();

      // assert(i >= j);

      if (memory_index[i] == -1) {
        memory_index[i] = b_vector.size();
        b_vector.PushBack(i, -v);
      } else {
        b_vector.GetValues()[memory_index[i]] -= v;
      }
    }
  }

  for (Index k = 0; k < b_vector.size(); k++) {
    memory_index[b_vector.GetIndex()[k]] = -1;
  }

  PruneZeros(b_vector, kEps);
  SortSparse(b_vector, shared.dirty_index_, shared.dirty_scalar_);
  shared.AssertClean();
}

// Solve L x' = x where L is defined by lcols and lrows coincides with lcols
// only in first n rows, where x and lcols are nonzero.
inline void LUFTranL(const std::vector<SparseVector> &lcols,
                     const std::vector<SparseVector> &lrows, SparseVector &x,
                     SharedMemory &shared) {
#ifndef NDEBUG
#ifdef BASIS_CHOICE_CHECK_SOLVE
  const std::vector<Scalar> original_x_dense = x.ToDense(lrows.size());
#endif

  Index n = IndexMax(x);

  for (Index k = 0; k < Index(lcols.size()); k++) {
    n = std::max(n, IndexMax(lcols[k]));
  }
#endif

  shared.AssertClean();
  std::vector<Index> &memory_index = shared.clean_index_;
  std::vector<Index> &nodes_stack = shared.dirty_index_;

  // fill-in computation
  for (Index k = 0; k < x.size(); k++) {
    memory_index[x.GetIndex()[k]] = k;
    nodes_stack.push_back(x.GetIndex()[k]);
  }

  while (!nodes_stack.empty()) {
    const Index j = nodes_stack.back();
    nodes_stack.pop_back();

    for (SvIterator el(lcols[j]); el; ++el) {
      const Index i = el.index();
      assert(i <= n);
      assert(i > j);

      if (memory_index[i] == -1) {
        memory_index[i] = x.size();
        x.PushBack(i, 0.0);
        nodes_stack.push_back(i);
      }
    }
  }

  SortSparse(x, shared.dirty_index_, shared.dirty_scalar_);

  // compute the values
  for (Index k = 0; k < x.size(); k++) {
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
  for (Index k = 0; k < x.size(); k++) {
    memory_index[x.GetIndex()[k]] = -1;
  }

  PruneZeros(x, kEps);
  shared.AssertClean();

  // check
#ifndef NDEBUG
#ifdef BASIS_CHOICE_CHECK_SOLVE
  const std::vector<Scalar> result_x_dense = x.ToDense(lrows.size());

  // first n rows
  for (Index i = 0; i <= n; i++) {
    const Scalar err =
        original_x_dense[i] - (lrows[i] * result_x_dense + result_x_dense[i]);
    if (std::abs(err) > kEps) {
      LOG_INFO("LUFTranL err i=%7i, err=%11.4e\n", i, err);
    }
  }

  // the rest
  for (Index i = n + 1; i < Index(lrows.size()); i++) {
    assert(result_x_dense[i] == original_x_dense[i]);
  }
#endif
#endif
}

inline FactorizeResult
BasisChoice::ComputeLU(const std::vector<SparseVector> &ct_cols,
                       const std::vector<Scalar> &priority) {
  // --- this function assumes ---
  // row_permutation_ is any
  // cols_permutation_ contains the final permutation
  // lrows_ is final size with empty elements
  // lcols_head_ is final size with empty elements
  // lcols_tail_ is final size with empty elements
  // urows_ is final size with empty elements
  // ucols_ is final size with empty elements
  // udiagonal_ is final size with zeros
  // -----------------------------

  this->row_permutation_.SetIdentity();

  SparseVector b_vector;
  b_vector.Reserve(this->nvectors_);

  for (Index j = 0; j < this->dimension_; j++) {
    const Index reserve_size =
        ct_cols[this->col_permutation_.Permute(j)].size();
    this->lcols_head_[j].Reserve(reserve_size);
    this->lcols_tail_[j].Reserve(reserve_size);
    this->ucols_[j].Reserve(reserve_size);
  }

  for (Index j = 0; j < this->dimension_; j++) {
    // --- get and permute the next column ---

    SparseVector &upper_col = this->ucols_[j];
    upper_col = ct_cols[this->col_permutation_.Permute(j)];

    PermuteSparse(upper_col, this->row_permutation_.GetPermutation());
    SortSparse(upper_col, this->shared_.dirty_index_,
               this->shared_.dirty_scalar_);

    // --- split off and compute the upper part ---

    Index upper_size = 0;
    while (upper_size < upper_col.size() &&
           upper_col.GetIndex()[upper_size] < j) {
      upper_size++;
    }

    for (Index k = upper_size; k < upper_col.size(); k++) {
      b_vector.PushBack(upper_col.GetIndex()[k], upper_col.GetValues()[k]);
    }

    upper_col.Resize(upper_size);

    LUFTranL(this->lcols_head_, this->lrows_, upper_col, this->shared_);

    // --- compute the b_vector ---

    LUBVectorCompute(b_vector, this->lcols_tail_, upper_col, this->shared_);

    if (b_vector.size() == 0) {
      return FactorizeResult::kSingular;
    }

    // --- pivot choice ---

    Index mem_pivot;
    {
      Scalar b_tol = b_vector.GetValues()[0];
      for (Index k = 1; k < b_vector.size(); k++) {
        const Scalar b_abs = std::abs(b_vector.GetValues()[k]);
        if (b_abs > b_tol) {
          b_tol = b_abs;
        }
      }
      b_tol = b_tol * kPivotTol;

      mem_pivot = 0;
      for (Index k = 0; k < b_vector.size(); k++) {
        if (std::abs(b_vector.GetValues()[k]) > b_tol) {
          mem_pivot = k;
          break;
        }
      }

      for (Index k = mem_pivot + 1; k < b_vector.size(); k++) {
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

    this->udiagonal_[j] = b_vector.GetValues()[mem_pivot];

    if (b_vector.GetIndex()[0] == j) {
      std::swap(b_vector.GetValues()[0], b_vector.GetValues()[mem_pivot]);
      b_vector.Erase(0);
    } else {
      assert(b_vector.GetIndex()[0] > j);
      b_vector.Erase(mem_pivot);
    }

    // --- update the row permutation ---

    const Index inverse_pivot = this->row_permutation_.Inverse(pivot);
    const Index inverse_j = this->row_permutation_.Inverse(j);
    this->row_permutation_.GetPermutation()[inverse_pivot] = j;
    this->row_permutation_.GetPermutation()[inverse_j] = pivot;
    this->row_permutation_.GetInverse()[j] = inverse_pivot;
    this->row_permutation_.GetInverse()[pivot] = inverse_j;
    this->row_permutation_.AssertIntegrity();

    // --- update lower matrix ---

    // permute tail and remove pivot row from tail
    LUTailSwapRemoves(this->lcols_tail_, this->lrows_[j], this->lrows_[pivot],
                      pivot);

    // permute lower rows
    std::swap(this->lrows_[j], this->lrows_[pivot]);

    // add the pivot row row to lower cols head
    for (SvIterator el(this->lrows_[j]); el; ++el) {
      assert(el.index() < j);
      this->lcols_head_[el.index()].PushBack(j, el.value());
    }

    // add a new col to lower rows
    for (SvIterator el(b_vector); el; ++el) {
      assert(el.index() > j);
      this->lrows_[el.index()].PushBack(j, el.value() / this->udiagonal_[j]);
      this->lcols_tail_[j].PushBack(el.index(),
                                    el.value() / this->udiagonal_[j]);
    }

    // --- cleanup ---

    b_vector.Clear();
  }

  // compute upper rows
  for (Index j = 0; j < this->dimension_; j++) {
    for (SvIterator el(this->ucols_[j]); el; ++el) {
      assert(el.index() < this->dimension_);
      this->urows_[el.index()].PushBack(j, el.value());
    }
  }

  return FactorizeResult::kOk;
}

inline bool
BasisChoice::CheckFactorization(const std::vector<SparseVector> &vectors,
                                Scalar tol) const {
  bool valid = true;

  for (Index j = 0; j < this->nvectors_; j++) {
    const std::vector<Scalar> dense_input =
        vectors[j].ToDense(this->dimension_);

    const Index pfov = this->row_permutation_.Permute(j);
    const SparseVector &lrow = this->lrows_[pfov];

    for (Index i = 0; i < this->dimension_; i++) {
      const Index qinv = this->col_permutation_.Inverse(i);
      const SparseVector &ucol = this->ucols_[qinv];
      // lu_value = lrows(pfov(j)) * ucols(qinv(i))

      Scalar lu_value = (lrow * ucol) + (SparseVector(pfov, 1.0) * ucol) +
                        (lrow * SparseVector(qinv, this->udiagonal_[qinv])) +
                        (pfov == qinv ? this->udiagonal_[qinv] : 0.0);

      const Scalar c_value = dense_input[i];

      if (std::abs(lu_value - c_value) > tol) {
        LOG_INFO("i=%4i, j=%4i, lu=%.4e, c=%.4e, err=%.4e\n", i, j, lu_value,
                 c_value, c_value - lu_value);
        valid = false;
      }
    }

    if (j % 1000 == 1000 - 1) {
      LOG_INFO("Checked %i/%i cols\n", j + 1, this->nvectors_);
    }
  }

  return valid;
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
    stats.u_nnz += this->ucols_[i].size();
  }
  stats.u_sparse = Scalar(stats.u_nnz) / u_spaces;

  // L is from lrows
  for (Index i = 0; i < this->nvectors_; i++) {
    // stats.l_nnz += std::min(i, this->dimension_);
    stats.l_nnz += this->lrows_[i].size();
  }
  stats.l_sparse =
      Scalar(stats.l_nnz) / (nvectors * dimension - u_spaces - dimension);

  // L1 is from lcols
  for (Index i = 0; i < this->dimension_; i++) {
    stats.l1_nnz += this->lcols_head_[i].size();
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

// inline void COLAMD(const Index &nrows, const Index &ncols,
//                    const std::vector<SparseVector> &cols,
//                    Permutation &permutation) {
//
// }

inline void SimpleOrdering(const std::vector<SparseVector> &cols,
                           const std::vector<Scalar> &priority,
                           Permutation &permutation) {
  const Index ncols = cols.size();
  std::vector<Scalar> col_metric(ncols);

  for (Index j = 0; j < ncols; j++) {
    const SparseVector &col = cols[j];

    col_metric[j] = 0;

    for (SvIterator el(col); el; ++el) {
      if (priority[el.index()] < col_metric[j]) {
        col_metric[j] = priority[el.index()];
      }
    }
  }

  permutation.SetIdentity();
  std::sort(permutation.GetPermutation().begin(),
            permutation.GetPermutation().end(),
            [&](const Index &lhs, const Index &rhs) -> bool {
              if (col_metric[lhs] == col_metric[rhs]) {
                return cols[lhs].size() < cols[rhs].size();
              } else {
                return col_metric[lhs] > col_metric[rhs];
              }
            });
  permutation.RestoreInverse();
}

inline void BasisChoice::ComputeQ(const std::vector<SparseVector> &ct_cols,
                                  const std::vector<Scalar> &priority) {
  SimpleOrdering(ct_cols, priority, this->col_permutation_);

  // COLAMD(this->nvectors_, this->dimension_, ct_cols, this->col_permutation_);

  this->col_permutation_.AssertIntegrity();
}

} // namespace basis_choice

#endif
