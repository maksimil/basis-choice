#ifndef __BASIS_CHOICE_H__
#define __BASIS_CHOICE_H__

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
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
  for (Index k = 0; k < Index(v.size()); k++) {
    std::swap(swap[permutation[k]], v[k]);
  }

  std::swap(swap, v);
}

class SharedMemory {
public:
  SharedMemory(Index dimension, Index nvectors)
      : swap_(std::max(dimension, nvectors)) {}

  std::vector<Scalar> swap_;
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
  FactorizeResult Factorize(const std::vector<SparseVector> &vectors) {
    const std::vector<SparseVector> row_rep =
        ComputeRowRepresentation(vectors, this->dimension_);
    return this->FactorizeCT(row_rep);
  }

  // factorize in terms of rows of input vectors
  FactorizeResult FactorizeCT(const std::vector<SparseVector> &ct_cols) {
    // compute col permutation
    this->ComputeQ(ct_cols);

    // compute factorization
    const FactorizeResult r = this->ComputeLU(ct_cols);

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

  FactorizeResult ComputeLU(const std::vector<SparseVector> &ct_cols) {
    assert(false);
  }

  // solve Cx'=x
  void SolveInPlace(std::vector<Scalar> &x) const {
    // C' x' = x
    this->SolveBasisInPlace(x);

    // padding (x')^T = (x^T 0 ... 0)
    x.resize(this->nvectors_, 0.0);

    // x' = P^T x
    PermuteDense(x, this->shared_.swap_, this->row_permutation_.GetInverse());
  }

  std::vector<Scalar> Solve(const std::vector<Scalar> &y) const {
    std::vector<Scalar> x = y;
    this->SolveInPlace(x);
    return x;
  }

  // solve C' x' = x
  void SolveBasisInPlace(std::vector<Scalar> &x) const {
    // x' = Q^T x
    PermuteDense(x, this->shared_.swap_, this->row_permutation_.GetInverse());

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

} // namespace basis_choice

#endif
