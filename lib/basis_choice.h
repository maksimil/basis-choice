#ifndef __BASIS_CHOICE_H__
#define __BASIS_CHOICE_H__

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

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

inline std::vector<Scalar> SparseVector::ToDense(size_t out_size) const {
  std::vector<Scalar> res(out_size, 0.0);
  for (Index i = 0; i < size(); ++i) {
    assert(index_[i] < Index(out_size));
    res[index_[i]] = values_[i];
  }
  return res;
}

inline Scalar SparseVector::operator*(const SparseVector &x) const {
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
                          const std::vector<Index> &permutation,
                          std::vector<Index> &index_swap,
                          std::vector<Scalar> &scalar_swap) {
  for (Index i = 0; i < v.size(); i++) {
    v.GetIndex()[i] = permutation[v.GetIndex()[i]];
  }

  SortSparse(v, index_swap, scalar_swap);
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

  void LogStats() const {
    LOG_INFO("nnz(L) = %i (%f%%), nnz(L1) = %i (%f%%), nnz(U) = %i (%f%%), "
             "nnz(F) = %i (%f%%)\n",
             this->l_nnz, this->l_sparse * 100.0, this->l1_nnz,
             this->l1_sparse * 100.0, this->u_nnz, this->u_sparse * 100.0,
             this->total_nnz, this->total_sparse * 100.0);
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
    this->ComputeQ(ct_cols);

    // compute factorization
    const FactorizeResult r = this->ComputeLU(ct_cols, priority);

    return r;
  }

  void ComputeQ(const std::vector<SparseVector> &ct_cols) {
    // TODO: implement COLAMD
    this->col_permutation_.SetIdentity();
    std::sort(this->col_permutation_.GetPermutation().begin(),
              this->col_permutation_.GetPermutation().end(),
              [&](const Index &lhs, const Index &rhs) -> bool {
                return ct_cols[lhs].size() < ct_cols[rhs].size();
              });
    this->col_permutation_.RestoreInverse();
    this->col_permutation_.AssertIntegrity();
  }

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
    assert(false);

    // L_1^T x' = x
    assert(false);
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
  Index jmem = 0;
  Index pmem = 0;

  while (jmem < jrow.size() && pmem < prow.size()) {
    const Index jidx = jrow.GetIndex()[jmem];
    const Index pidx = prow.GetIndex()[pmem];

    const Index midx = std::min(jidx, pidx);

    SparseVector &tailcol = tail[midx];
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

// Solve the first n rows and cols of L x' = x for sparse LU factorization
// lcols spans only first n rows of L
// lrows spans at least the first n rows L
//
// Works only with the first n elements of x
inline void LUFTranL(const std::vector<SparseVector> &lcols,
                     const std::vector<SparseVector> &lrows, const Index n,
                     SparseVector &x, SharedMemory &shared) {
#ifndef NDEBUG
  const SparseVector original_x = x;
  const std::vector<Scalar> original_x_dense = original_x.ToDense(lrows.size());
#endif
  assert(n < Index(lcols.size()));

  shared.AssertClean();
  std::vector<Index> &memory_index = shared.clean_index_;
  std::vector<Index> &nodes_stack = shared.dirty_index_;

  // fill-in computation
  for (Index k = 0; k < x.size() && x.GetIndex()[k] < n; k++) {
    memory_index[x.GetIndex()[k]] = k;
    nodes_stack.push_back(x.GetIndex()[k]);
  }

  while (!nodes_stack.empty()) {
    const Index j = nodes_stack.back();
    nodes_stack.pop_back();

    for (SvIterator el(lcols[j]); el; ++el) {
      const Index i = el.index();
      assert(i < n);
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
  for (Index k = 0; k < x.size() && x.GetIndex()[k] < n; k++) {
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
  for (Index k = 0; k < x.size() && x.GetIndex()[k] < n; k++) {
    memory_index[x.GetIndex()[k]] = -1;
  }

  PruneZeros(x, kEps);
  shared.AssertClean();

  // check
#ifndef NDEBUG
  const std::vector<Scalar> result_x_dense = x.ToDense(lrows.size());

  // first n rows
  for (Index i = 0; i < n; i++) {
    const Scalar err =
        original_x_dense[i] - (lrows[i] * result_x_dense + result_x_dense[i]);
    if (std::abs(err) > kEps) {
      LOG_INFO("FTranL err i=%i, err=%f\n", i, err);
    }
  }

  // the rest
  for (Index i = n; i < Index(lrows.size()); i++) {
    assert(result_x_dense[i] == original_x_dense[i]);
  }
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

  for (Index j = 0; j < this->dimension_; j++) {
    // --- get the next column ---

    SparseVector &upper_col = this->ucols_[j];
    upper_col = ct_cols[this->col_permutation_.Permute(j)];

    PermuteSparse(upper_col, this->row_permutation_.GetPermutation(),
                  this->shared_.dirty_index_, this->shared_.dirty_scalar_);

    LUFTranL(this->lcols_head_, this->lrows_, j, upper_col, this->shared_);

    // --- split off the upper part ---
    SparseVector b_vector;
    b_vector.Reserve(upper_col.size());

    Index upper_size = 0;
    while (upper_size < upper_col.size() &&
           upper_col.GetIndex()[upper_size] < j) {
      upper_size++;
    }

    for (Index k = upper_size; k < upper_col.size(); k++) {
      b_vector.PushBack(upper_col.GetIndex()[k], upper_col.GetValues()[k]);
    }

    upper_col.Resize(upper_size);

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
        if (std::abs(b_vector.GetValues()[k]) <= b_tol) {
          continue;
        }

        if (priority[b_vector.GetIndex()[k]] <
            priority[b_vector.GetIndex()[mem_pivot]]) {
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

    // permute tail and remove pivot row
    LUTailSwapRemoves(this->lcols_tail_, this->lrows_[j], this->lrows_[pivot],
                      pivot);

    // permute lower rows
    std::swap(this->lrows_[j], this->lrows_[pivot]);

    // add the pivot row row to lower cols
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

  const Index u_spaces = this->dimension_ * (this->dimension_ - 1) / 2;

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
  stats.l_sparse = Scalar(stats.l_nnz) / (this->nvectors_ * this->dimension_ -
                                          u_spaces - this->dimension_);

  // L1 is from lcols
  for (Index i = 0; i < this->dimension_; i++) {
    stats.l1_nnz += this->lcols_head_[i].size();
  }
  stats.l1_sparse = Scalar(stats.l1_nnz) / u_spaces;

  stats.total_nnz = stats.u_nnz + stats.l_nnz + this->dimension_;
  stats.total_sparse =
      Scalar(stats.total_nnz) / (this->nvectors_ * this->dimension_);

  return stats;
}

} // namespace basis_choice

#endif
