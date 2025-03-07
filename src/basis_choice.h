#ifndef __BASIS_CHOICE_H__
#define __BASIS_CHOICE_H__

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define LOG_INFO(fmt, args...) fprintf(stdout, fmt, ##args)

using Scalar = double;
using Index = int32_t;

class SparseVector {
public:
  SparseVector(Index n) : index_(n), values_(n) {}

  SparseVector() {}

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

private:
  std::vector<Index> index_;
  std::vector<Scalar> values_;
};

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

inline std::vector<SparseVector>
ComputeRowRepresentation(const std::vector<SparseVector> &cols, Index nrows) {
  std::vector<SparseVector> rows(nrows);

  for (Index ic = 0; ic < cols.size(); ic++) {
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
#if DEBUG == 1
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

inline void SortSparse(SparseVector &v, Index dimension,
                       std::vector<Index> &index_swap,
                       std::vector<Scalar> &scalar_swap) {
  scalar_swap.resize(dimension);

  for (SvIterator el(v); el; ++el) {
    scalar_swap[el.index()] = el.value();
    index_swap.push_back(el.index());
  }

  std::sort(index_swap.begin(), index_swap.end());
  v.Clear();

  for (Index i = 0; i < index_swap.size(); i++) {
    v.PushBack(index_swap[i], scalar_swap[index_swap[i]]);
  }

  index_swap.clear();
  scalar_swap.clear();
}

inline void PermuteSparse(SparseVector &v,
                          const std::vector<Index> &permutation,
                          std::vector<Index> &index_swap,
                          std::vector<Scalar> &scalar_swap) {
  for (Index i = 0; i < v.size(); i++) {
    v.GetIndex()[i] = permutation[v.GetIndex()[i]];
  }

  SortSparse(v, permutation.size(), index_swap, scalar_swap);
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
  }

  std::vector<Scalar> dirty_scalar_;
  std::vector<Index> dirty_index_;
  std::vector<Index> clean_index_; // value = -1
};

enum FactorizeResult {
  kOk,
  kSingular,
};

class BasisChoice {
public:
  BasisChoice(Index dimension, Index nvectors)
      : row_permutation_(nvectors), col_permutation_(dimension),
        lower_triangular_rows_(nvectors), lower_triangular_cols_(dimension),
        upper_triangular_rows_(dimension), upper_triangular_cols_(dimension),
        upper_diagonal_(dimension), shared_(dimension, nvectors) {}

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
    std::sort(this->col_permutation_.GetPermutation().begin(),
              this->col_permutation_.GetPermutation().end(),
              [&](const Index &lhs, const Index &rhs) -> bool {
                return ct_cols[lhs].size() <= ct_cols[rhs].size();
              });

    this->col_permutation_.RestoreInverse();
    this->col_permutation_.AssertIntegrity();
  }

  FactorizeResult ComputeLU(const std::vector<SparseVector> &ct_cols,
                            const std::vector<Scalar> &priority);

  // solve Cx'=x
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
                 this->row_permutation_.GetInverse());

    // U^T x' = x
    assert(false);

    // L_1^T x' = x
    assert(false);
  }

private:
  // LU = PC^TQ
  // L = (L_1^T L_2^T)^T
  // C' = Q U^T L_1^T

  // dimensions
  Index dimension_; // ncols
  Index nvectors_;  // nrows

  // P, Q permutation matrices
  Permutation row_permutation_; // P
  Permutation col_permutation_; // Q

  // L_1, U factors
  std::vector<SparseVector> lower_triangular_rows_;
  std::vector<SparseVector> lower_triangular_cols_;
  std::vector<SparseVector> upper_triangular_rows_;
  std::vector<SparseVector> upper_triangular_cols_;
  std::vector<Scalar> upper_diagonal_;

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

// Solve L x' = x in LU only for the first n cols
// lcols is assumed to span only first n rows and all the cols
// lrows is assumed to span the whole matrix
inline void FactorizationFTranL(const std::vector<SparseVector> &lcols,
                                const std::vector<SparseVector> &lrows,
                                const Index n, SparseVector &x,
                                SharedMemory &shared) {
  std::vector<Index> &memory_index = shared.clean_index_;

  // fill-in computation
  for (Index k = 0; k < x.size(); k++) {
    memory_index[x.GetIndex()[k]] = k;
  }

  Index k = 0;
  while (k < x.size()) {
    const Index j = x.GetIndex()[k];

    for (SvIterator el(lcols[j]); el; ++el) {
      const Index i = el.index();

      if (memory_index[i] != -1) {
        memory_index[el.index()] = x.size();
        x.PushBack(el.index(), 0.0);
      }
    }

    k++;
  }

  SortSparse(x, lcols.size(), shared.dirty_index_, shared.dirty_scalar_);

  for (Index k = 0; k < x.size(); k++) {
    memory_index[x.GetIndex()[k]] = k;
  }

  // compute the values
  for (Index k = 0; k < x.size(); k++) {
    const Index i = x.GetIndex()[k];
    Scalar &xi = x.GetValues()[k];

    for (SvIterator el(lrows[i]); el; ++el) {
      xi -= el.value() * x.GetValues()[memory_index[el.index()]];
    }
  }

  // clean up
  for (Index k = 0; k < x.size(); k++) {
    memory_index[x.GetIndex()[k]] = -1;
  }

  PruneVector(x, [&](Index _, Scalar v) -> bool { return std::abs(v) < kEps; });
}

inline FactorizeResult
BasisChoice::ComputeLU(const std::vector<SparseVector> &ct_cols,
                       const std::vector<Scalar> &priority) {
  // --- this function assumes ---
  // row_permutation_ is empty (all zeros, not a correct permutation)
  // cols_permutation_ contains the final permutation
  // lower_triangular_rows_ is final size with empty elements
  // lower_triangular_cols_ is final size with empty elements
  // upper_triangular_rows_ is final size with empty elements
  // upper_triangular_cols_ is final size with empty elements
  // upper_diagonal_ is final size with undefined elements
  // -----------------------------

  this->row_permutation_.SetIdentity();

  for (Index j = 0; j < dimension_; j++) {
    SparseVector &upper_col = this->upper_triangular_cols_[j];
    upper_col = ct_cols[this->col_permutation_.Permute(j)];

    PermuteSparse(upper_col, this->row_permutation_.GetPermutation(),
                  this->shared_.dirty_index_, this->shared_.dirty_scalar_);

    FactorizationFTranL(this->lower_triangular_cols_,
                        this->lower_triangular_rows_, j, upper_col,
                        this->shared_);

    // split off the upper factor

    SparseVector &lower_col = this->lower_triangular_cols_[j];
    lower_col.Reserve(upper_col.size());

    Index k = 0;
    while (upper_col.GetIndex()[k] < j) {
      assert(k < upper_col.size());
      k++;
    }

    for (; k < upper_col.size(); k++) {
      lower_col.PushBack(upper_col.GetIndex()[k], upper_col.GetValues()[k]);
    }

    // pivot choice

    Index mem_pivot = 0;
    Index pivot = lower_col.GetIndex()[0];

    if (lower_col.size() == 1) {
      mem_pivot = 0;
    } else {
      mem_pivot = 1;
    }

    pivot = lower_col.GetIndex()[mem_pivot];

    // setting diagonal, lower factors and permutation

    this->upper_diagonal_[j] = lower_col.GetValues()[mem_pivot];

    if (lower_col.GetIndex()[0] == j) {
      std::swap(lower_col.GetValues()[0], lower_col.GetValues()[mem_pivot]);
      lower_col.GetValues().erase(lower_col.GetValues().begin());
      lower_col.GetIndex().erase(lower_col.GetIndex().begin());
    } else {
      assert(false);
    }

    const Index inverse_pivot = this->row_permutation_.Inverse(pivot);
    const Index inverse_j = this->row_permutation_.Inverse(j);
    this->row_permutation_.GetPermutation()[inverse_pivot] = j;
    this->row_permutation_.GetPermutation()[inverse_j] = pivot;
    this->row_permutation_.GetInverse()[j] = inverse_pivot;
    this->row_permutation_.GetInverse()[pivot] = inverse_j;
  }

  return FactorizeResult::kOk;
}

} // namespace basis_choice

#endif
