#[cfg(test)]
mod test;

#[cfg(test)]
use super::test::ErrorTensor;

use std::{
    array::from_fn,
    fmt::{self, Display, Formatter},
    iter::Sum,
    mem::transmute,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use super::{
    Hessian, Rank2, SquareMatrix, Tensor, TensorArray, Vector,
    rank_0::TensorRank0,
    rank_1::TensorRank1,
    rank_2::TensorRank2,
    rank_3::{TensorRank3, get_identity_1010_parts},
};

pub mod list;

/// A *d*-dimensional tensor of rank 4.
///
/// `D` is the dimension, `I`, `J`, `K`, `L` are the configurations.
#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRank4<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
>([TensorRank3<D, J, K, L>; D]);

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Default
    for TensorRank4<D, I, J, K, L>
{
    fn default() -> Self {
        Self::zero()
    }
}

pub const IDENTITY_1010: TensorRank4<3, 1, 0, 1, 0> = TensorRank4(get_identity_1010_parts());

impl<const I: usize, const J: usize, const K: usize> From<TensorRank4<3, I, J, K, 0>>
    for TensorRank4<3, I, J, K, 2>
{
    fn from(tensor_rank_4: TensorRank4<3, I, J, K, 0>) -> Self {
        unsafe {
            transmute::<TensorRank4<3, I, J, K, 0>, TensorRank4<3, I, J, K, 2>>(tensor_rank_4)
        }
    }
}

impl<const J: usize, const L: usize> From<TensorRank4<3, 1, J, 1, L>>
    for TensorRank4<3, 2, J, 2, L>
{
    fn from(tensor_rank_4: TensorRank4<3, 1, J, 1, L>) -> Self {
        unsafe {
            transmute::<TensorRank4<3, 1, J, 1, L>, TensorRank4<3, 2, J, 2, L>>(tensor_rank_4)
        }
    }
}

impl<const I: usize, const K: usize> From<TensorRank4<3, I, 0, K, 0>>
    for TensorRank4<3, I, 2, K, 2>
{
    fn from(tensor_rank_4: TensorRank4<3, I, 0, K, 0>) -> Self {
        unsafe {
            transmute::<TensorRank4<3, I, 0, K, 0>, TensorRank4<3, I, 2, K, 2>>(tensor_rank_4)
        }
    }
}

impl<const K: usize> From<TensorRank4<3, 0, 0, K, 0>> for TensorRank4<3, 2, 2, K, 2> {
    fn from(tensor_rank_4: TensorRank4<3, 0, 0, K, 0>) -> Self {
        unsafe {
            transmute::<TensorRank4<3, 0, 0, K, 0>, TensorRank4<3, 2, 2, K, 2>>(tensor_rank_4)
        }
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    From<Vec<Vec<Vec<Vec<TensorRank0>>>>> for TensorRank4<D, I, J, K, L>
{
    fn from(vec_rank_4: Vec<Vec<Vec<Vec<TensorRank0>>>>) -> Self {
        vec_rank_4
            .into_iter()
            .map(|vec_rank_3| {
                vec_rank_3
                    .into_iter()
                    .map(|vec_rank_2| {
                        vec_rank_2
                            .into_iter()
                            .map(|vec_rank_1| vec_rank_1.into_iter().collect())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    From<TensorRank4<D, I, J, K, L>> for Vec<Vec<Vec<Vec<TensorRank0>>>>
{
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L>) -> Self {
        tensor_rank_4
            .iter()
            .map(|tensor_rank_3| {
                tensor_rank_3
                    .iter()
                    .map(|tensor_rank_2| {
                        tensor_rank_2
                            .iter()
                            .map(|tensor_rank_1| tensor_rank_1.iter().copied().collect())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    From<TensorRank4<D, I, J, K, L>> for Vec<TensorRank0>
{
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L>) -> Self {
        tensor_rank_4
            .iter()
            .flat_map(|tensor_rank_3| {
                tensor_rank_3.iter().flat_map(|tensor_rank_2| {
                    tensor_rank_2
                        .iter()
                        .flat_map(|tensor_rank_1| tensor_rank_1.iter().copied())
                })
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    From<TensorRank4<D, I, J, K, L>> for Vector
{
    fn from(tensor_rank_4: TensorRank4<D, I, J, K, L>) -> Self {
        tensor_rank_4
            .iter()
            .flat_map(|tensor_rank_3| {
                tensor_rank_3.iter().flat_map(|tensor_rank_2| {
                    tensor_rank_2
                        .iter()
                        .flat_map(|tensor_rank_1| tensor_rank_1.iter().copied())
                })
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Display
    for TensorRank4<D, I, J, K, L>
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[")?;
        self.iter()
            .enumerate()
            .try_for_each(|(i, entry)| write!(f, "{entry},\n\x1B[u\x1B[{}B\x1B[2D", i + 1))?;
        write!(f, "\x1B[u\x1B[1A\x1B[{}C]", 16 * D + 2)
    }
}

#[cfg(test)]
impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> ErrorTensor
    for TensorRank4<D, I, J, K, L>
{
    fn error_fd(&self, comparator: &Self, epsilon: TensorRank0) -> Option<(bool, usize)> {
        let error_count = self
            .iter()
            .zip(comparator.iter())
            .map(|(self_i, comparator_i)| {
                self_i
                    .iter()
                    .zip(comparator_i.iter())
                    .map(|(self_ij, comparator_ij)| {
                        self_ij
                            .iter()
                            .zip(comparator_ij.iter())
                            .map(|(self_ijk, comparator_ijk)| {
                                self_ijk
                                    .iter()
                                    .zip(comparator_ijk.iter())
                                    .filter(|&(&self_ijkl, &comparator_ijkl)| {
                                        (self_ijkl / comparator_ijkl - 1.0).abs() >= epsilon
                                            && (self_ijkl.abs() >= epsilon
                                                || comparator_ijkl.abs() >= epsilon)
                                    })
                                    .count()
                            })
                            .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum();
        if error_count > 0 {
            let auxiliary = self
                .iter()
                .zip(comparator.iter())
                .map(|(self_i, comparator_i)| {
                    self_i
                        .iter()
                        .zip(comparator_i.iter())
                        .map(|(self_ij, comparator_ij)| {
                            self_ij
                                .iter()
                                .zip(comparator_ij.iter())
                                .map(|(self_ijk, comparator_ijk)| {
                                    self_ijk
                                        .iter()
                                        .zip(comparator_ijk.iter())
                                        .filter(|&(&self_ijkl, &comparator_ijkl)| {
                                            (self_ijkl / comparator_ijkl - 1.0).abs() >= epsilon
                                                && (self_ijkl - comparator_ijkl).abs() >= epsilon
                                                && (self_ijkl.abs() >= epsilon
                                                    || comparator_ijkl.abs() >= epsilon)
                                        })
                                        .count()
                                })
                                .sum::<usize>()
                        })
                        .sum::<usize>()
                })
                .sum::<usize>()
                > 0;
            Some((auxiliary, error_count))
        } else {
            None
        }
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    TensorRank4<D, I, J, K, L>
{
    pub fn dyad_ij_kl(
        tensor_rank_2_a: &TensorRank2<D, I, J>,
        tensor_rank_2_b: &TensorRank2<D, K, L>,
    ) -> Self {
        tensor_rank_2_a
            .iter()
            .map(|tensor_rank_2_a_i| {
                tensor_rank_2_a_i
                    .iter()
                    .map(|tensor_rank_2_a_ij| tensor_rank_2_b * tensor_rank_2_a_ij)
                    .collect()
            })
            .collect()
    }
    pub fn dyad_ik_jl(
        tensor_rank_2_a: &TensorRank2<D, I, K>,
        tensor_rank_2_b: &TensorRank2<D, J, L>,
    ) -> Self {
        tensor_rank_2_a
            .iter()
            .map(|tensor_rank_2_a_i| {
                tensor_rank_2_b
                    .iter()
                    .map(|tensor_rank_2_b_j| {
                        tensor_rank_2_a_i
                            .iter()
                            .map(|tensor_rank_2_a_ik| tensor_rank_2_b_j * tensor_rank_2_a_ik)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
    pub fn dyad_il_jk(
        tensor_rank_2_a: &TensorRank2<D, I, L>,
        tensor_rank_2_b: &TensorRank2<D, J, K>,
    ) -> Self {
        tensor_rank_2_a
            .iter()
            .map(|tensor_rank_2_a_i| {
                tensor_rank_2_b
                    .iter()
                    .map(|tensor_rank_2_b_j| {
                        tensor_rank_2_b_j
                            .iter()
                            .map(|tensor_rank_2_b_jk| tensor_rank_2_a_i * tensor_rank_2_b_jk)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
    pub fn dyad_il_kj(
        tensor_rank_2_a: &TensorRank2<D, I, L>,
        tensor_rank_2_b: &TensorRank2<D, K, J>,
    ) -> Self {
        Self::dyad_il_jk(tensor_rank_2_a, &(tensor_rank_2_b.transpose()))
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Hessian
    for TensorRank4<D, I, J, K, L>
{
    fn fill_into(self, square_matrix: &mut SquareMatrix) {
        self.into_iter().enumerate().for_each(|(i, self_i)| {
            self_i.into_iter().enumerate().for_each(|(j, self_ij)| {
                self_ij.into_iter().enumerate().for_each(|(k, self_ijk)| {
                    self_ijk
                        .into_iter()
                        .enumerate()
                        .for_each(|(l, self_ijkl)| square_matrix[D * i + j][D * k + l] = self_ijkl)
                })
            })
        })
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Tensor
    for TensorRank4<D, I, J, K, L>
{
    type Item = TensorRank3<D, J, K, L>;
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        self.0.iter_mut()
    }
    fn len(&self) -> usize {
        D
    }
    fn size(&self) -> usize {
        D * D * D * D
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> IntoIterator
    for TensorRank4<D, I, J, K, L>
{
    type Item = TensorRank3<D, J, K, L>;
    type IntoIter = std::array::IntoIter<Self::Item, D>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> TensorArray
    for TensorRank4<D, I, J, K, L>
{
    type Array = [[[[TensorRank0; D]; D]; D]; D];
    type Item = TensorRank3<D, J, K, L>;
    fn as_array(&self) -> Self::Array {
        let mut array = [[[[0.0; D]; D]; D]; D];
        array
            .iter_mut()
            .zip(self.iter())
            .for_each(|(entry_rank_3, tensor_rank_3)| *entry_rank_3 = tensor_rank_3.as_array());
        array
    }
    fn identity() -> Self {
        Self::dyad_ij_kl(&TensorRank2::identity(), &TensorRank2::identity())
    }
    fn new(array: Self::Array) -> Self {
        array.into_iter().map(Self::Item::new).collect()
    }
    fn zero() -> Self {
        Self(from_fn(|_| Self::Item::zero()))
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    FromIterator<TensorRank3<D, J, K, L>> for TensorRank4<D, I, J, K, L>
{
    fn from_iter<Ii: IntoIterator<Item = TensorRank3<D, J, K, L>>>(into_iterator: Ii) -> Self {
        let mut tensor_rank_4 = Self::zero();
        tensor_rank_4
            .iter_mut()
            .zip(into_iterator)
            .for_each(|(tensor_rank_4_i, value_i)| *tensor_rank_4_i = value_i);
        tensor_rank_4
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Index<usize>
    for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank3<D, J, K, L>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> IndexMut<usize>
    for TensorRank4<D, I, J, K, L>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Sum
    for TensorRank4<D, I, J, K, L>
{
    fn sum<Ii>(iter: Ii) -> Self
    where
        Ii: Iterator<Item = Self>,
    {
        iter.reduce(|mut acc, item| {
            acc += item;
            acc
        })
        .unwrap_or_else(Self::default)
    }
}

pub trait ContractAllIndicesWithFirstIndicesOf<TIM, TJN, TKO, TLP> {
    type Output;
    fn contract_all_indices_with_first_indices_of(
        self,
        tensor_rank_2_a: TIM,
        tensor_rank_2_b: TJN,
        tensor_rank_2_c: TKO,
        tensor_rank_2_d: TLP,
    ) -> Self::Output;
}

impl<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
>
    ContractAllIndicesWithFirstIndicesOf<
        &TensorRank2<D, I, M>,
        &TensorRank2<D, J, N>,
        &TensorRank2<D, K, O>,
        &TensorRank2<D, L, P>,
    > for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, M, N, O, P>;
    fn contract_all_indices_with_first_indices_of(
        self,
        tensor_rank_2_a: &TensorRank2<D, I, M>,
        tensor_rank_2_b: &TensorRank2<D, J, N>,
        tensor_rank_2_c: &TensorRank2<D, K, O>,
        tensor_rank_2_d: &TensorRank2<D, L, P>,
    ) -> Self::Output {
        let mut output = TensorRank4::zero();
        self.into_iter().zip(tensor_rank_2_a.iter()).for_each(|(self_m, tensor_rank_2_a_m)|
            self_m.into_iter().zip(tensor_rank_2_b.iter()).for_each(|(self_mn, tensor_rank_2_b_n)|
                self_mn.into_iter().zip(tensor_rank_2_c.iter()).for_each(|(self_mno, tensor_rank_2_c_o)|
                    self_mno.into_iter().zip(tensor_rank_2_d.iter()).for_each(|(self_mnop, tensor_rank_2_d_p)|
                        output.iter_mut().zip(tensor_rank_2_a_m.iter()).for_each(|(output_i, tensor_rank_2_a_mi)|
                            output_i.iter_mut().zip(tensor_rank_2_b_n.iter()).for_each(|(output_ij, tensor_rank_2_b_nj)|
                                output_ij.iter_mut().zip(tensor_rank_2_c_o.iter()).for_each(|(output_ijk, tensor_rank_2_c_ok)|
                                    output_ijk.iter_mut().zip(tensor_rank_2_d_p.iter()).for_each(|(output_ijkl, tensor_rank_2_dp)|
                                        *output_ijkl += self_mnop * tensor_rank_2_a_mi * tensor_rank_2_b_nj * tensor_rank_2_c_ok * tensor_rank_2_dp
                                    )
                                )
                            )
                        )
                    )
                )
            )
        );
        output
    }
}

pub trait ContractFirstThirdFourthIndicesWithFirstIndicesOf<TIM, TKO, TLP> {
    type Output;
    fn contract_first_third_fourth_indices_with_first_indices_of(
        self,
        tensor_rank_2_a: TIM,
        tensor_rank_2_c: TKO,
        tensor_rank_2_d: TLP,
    ) -> Self::Output;
}

impl<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
    const M: usize,
    const O: usize,
    const P: usize,
>
    ContractFirstThirdFourthIndicesWithFirstIndicesOf<
        &TensorRank2<D, I, M>,
        &TensorRank2<D, K, O>,
        &TensorRank2<D, L, P>,
    > for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, M, J, O, P>;
    fn contract_first_third_fourth_indices_with_first_indices_of(
        self,
        tensor_rank_2_a: &TensorRank2<D, I, M>,
        tensor_rank_2_b: &TensorRank2<D, K, O>,
        tensor_rank_2_c: &TensorRank2<D, L, P>,
    ) -> Self::Output {
        let mut output = TensorRank4::zero();
        self.into_iter().zip(tensor_rank_2_a.iter()).for_each(|(self_q, tensor_rank_2_a_q)|
            output.iter_mut().zip(tensor_rank_2_a_q.iter()).for_each(|(output_i, tensor_rank_2_a_qi)|
                output_i.iter_mut().zip(self_q.iter()).for_each(|(output_ij, self_qj)|
                    self_qj.iter().zip(tensor_rank_2_b.iter()).for_each(|(self_qjm, tensor_rank_2_b_m)|
                        self_qjm.iter().zip(tensor_rank_2_c.iter()).for_each(|(self_qjmn, tensor_rank_2_c_n)|
                            output_ij.iter_mut().zip(tensor_rank_2_b_m.iter()).for_each(|(output_ijk, tensor_rank_2_b_mk)|
                                output_ijk.iter_mut().zip(tensor_rank_2_c_n.iter()).for_each(|(output_ijkl, tensor_rank_2_c_nl)|
                                    *output_ijkl += self_qjmn*tensor_rank_2_a_qi*tensor_rank_2_b_mk*tensor_rank_2_c_nl
                                )
                            )
                        )
                    )
                )
            )
        );
        output
    }
}

pub trait ContractSecondIndexWithFirstIndexOf<TJN> {
    type Output;
    fn contract_second_index_with_first_index_of(self, tensor_rank_2: TJN) -> Self::Output;
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const N: usize>
    ContractSecondIndexWithFirstIndexOf<&TensorRank2<D, J, N>> for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, I, N, K, L>;
    fn contract_second_index_with_first_index_of(
        self,
        tensor_rank_2: &TensorRank2<D, J, N>,
    ) -> Self::Output {
        let mut output = TensorRank4::zero();
        output.iter_mut().zip(self).for_each(|(output_i, self_i)| {
            self_i
                .into_iter()
                .zip(tensor_rank_2.iter())
                .for_each(|(self_is, tensor_rank_2_s)| {
                    output_i.iter_mut().zip(tensor_rank_2_s.iter()).for_each(
                        |(output_ij, tensor_rank_2_sj)| *output_ij += &self_is * tensor_rank_2_sj,
                    )
                })
        });
        output
    }
}

pub trait ContractSecondFourthIndicesWithFirstIndicesOf<TJ, TL> {
    type Output;
    fn contract_second_fourth_indices_with_first_indices_of(
        &self,
        object_a: TJ,
        object_b: TL,
    ) -> Self::Output;
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    ContractSecondFourthIndicesWithFirstIndicesOf<&TensorRank1<D, J>, &TensorRank1<D, L>>
    for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank2<D, I, K>;
    fn contract_second_fourth_indices_with_first_indices_of(
        &self,
        tensor_rank_1_a: &TensorRank1<D, J>,
        tensor_rank_1_b: &TensorRank1<D, L>,
    ) -> Self::Output {
        self.iter()
            .map(|self_i| {
                self_i
                    .iter()
                    .zip(tensor_rank_1_a.iter())
                    .map(|(self_ij, tensor_rank_1_a_j)| {
                        self_ij
                            .iter()
                            .map(|self_ijk| self_ijk * (tensor_rank_1_b * tensor_rank_1_a_j))
                            .collect()
                    })
                    .sum()
            })
            .collect()
    }
}

pub trait ContractThirdIndexWithFirstIndexOf<TKL> {
    type Output;
    fn contract_third_index_with_first_index_of(&self, tensor: TKL) -> Self::Output;
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    ContractThirdIndexWithFirstIndexOf<&TensorRank2<D, M, K>> for TensorRank4<D, I, J, M, L>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn contract_third_index_with_first_index_of(
        &self,
        tensor_rank_2: &TensorRank2<D, M, K>,
    ) -> Self::Output {
        self.iter()
            .map(|self_i| {
                self_i
                    .iter()
                    .map(|self_ij| {
                        self_ij
                            .iter()
                            .zip(tensor_rank_2.iter())
                            .map(|(self_ijm, tensor_rank_2_m)| {
                                (tensor_rank_2_m, self_ijm).into()
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

pub trait ContractThirdFourthIndicesWithFirstSecondIndicesOf<TKL> {
    type Output;
    fn contract_third_fourth_indices_with_first_second_indices_of(
        self,
        tensor: TKL,
    ) -> Self::Output;
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    ContractThirdFourthIndicesWithFirstSecondIndicesOf<&TensorRank2<D, K, L>>
    for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank2<D, I, J>;
    fn contract_third_fourth_indices_with_first_second_indices_of(
        self,
        tensor_rank_2: &TensorRank2<D, K, L>,
    ) -> Self::Output {
        self.into_iter()
            .map(|self_i| {
                self_i
                    .into_iter()
                    .map(|self_ij| self_ij.full_contraction(tensor_rank_2))
                    .collect()
            })
            .collect()
    }
}

impl<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
    const M: usize,
    const N: usize,
> ContractThirdFourthIndicesWithFirstSecondIndicesOf<&TensorRank4<D, K, L, M, N>>
    for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, I, J, M, N>;
    fn contract_third_fourth_indices_with_first_second_indices_of(
        self,
        tensor: &TensorRank4<D, K, L, M, N>,
    ) -> Self::Output {
        self.into_iter()
            .map(|self_i| {
                self_i
                    .into_iter()
                    .map(|self_ij| {
                        self_ij
                            .into_iter()
                            .zip(tensor.iter())
                            .map(|(self_ijk, tensor_k)| {
                                self_ijk
                                    .into_iter()
                                    .zip(tensor_k.iter())
                                    .map(|(self_ijkl, tensor_kl)| tensor_kl * self_ijkl)
                                    .sum()
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

pub trait ContractFirstSecondIndicesWithSecondIndicesOf<TI, TJ> {
    type Output;
    fn contract_first_second_indices_with_second_indices_of(
        self,
        object_a: TI,
        object_b: TJ,
    ) -> Self::Output;
}

impl<
    const D: usize,
    const I: usize,
    const J: usize,
    const K: usize,
    const L: usize,
    const M: usize,
    const N: usize,
> ContractFirstSecondIndicesWithSecondIndicesOf<&TensorRank2<D, I, M>, &TensorRank2<D, J, N>>
    for TensorRank4<D, M, N, K, L>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn contract_first_second_indices_with_second_indices_of(
        self,
        tensor_rank_2_a: &TensorRank2<D, I, M>,
        tensor_rank_2_b: &TensorRank2<D, J, N>,
    ) -> Self::Output {
        let mut output = TensorRank4::zero();
        output
            .iter_mut()
            .zip(tensor_rank_2_a.iter())
            .for_each(|(output_i, tensor_rank_2_a_i)| {
                output_i.iter_mut().zip(tensor_rank_2_b.iter()).for_each(
                    |(output_ij, tensor_rank_2_b_j)| {
                        self.iter().zip(tensor_rank_2_a_i.iter()).for_each(
                            |(self_m, tensor_rank_2_a_im)| {
                                self_m.iter().zip(tensor_rank_2_b_j.iter()).for_each(
                                    |(self_mn, tensor_rank_2_b_jn)| {
                                        *output_ij +=
                                            self_mn * tensor_rank_2_a_im * tensor_rank_2_b_jn
                                    },
                                )
                            },
                        )
                    },
                )
            });
        output
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    Div<TensorRank0> for TensorRank4<D, I, J, K, L>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self /= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    Div<TensorRank0> for &TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn div(self, tensor_rank_0: TensorRank0) -> Self::Output {
        self.iter().map(|self_i| self_i / tensor_rank_0).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    Div<&TensorRank0> for TensorRank4<D, I, J, K, L>
{
    type Output = Self;
    fn div(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self /= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    DivAssign<TensorRank0> for TensorRank4<D, I, J, K, L>
{
    fn div_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    DivAssign<&TensorRank0> for TensorRank4<D, I, J, K, L>
{
    fn div_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i /= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    Mul<TensorRank0> for TensorRank4<D, I, J, K, L>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: TensorRank0) -> Self::Output {
        self *= &tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    Mul<&TensorRank0> for TensorRank4<D, I, J, K, L>
{
    type Output = Self;
    fn mul(mut self, tensor_rank_0: &TensorRank0) -> Self::Output {
        self *= tensor_rank_0;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    MulAssign<TensorRank0> for TensorRank4<D, I, J, K, L>
{
    fn mul_assign(&mut self, tensor_rank_0: TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= &tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    MulAssign<&TensorRank0> for TensorRank4<D, I, J, K, L>
{
    fn mul_assign(&mut self, tensor_rank_0: &TensorRank0) {
        self.iter_mut().for_each(|self_i| *self_i *= tensor_rank_0);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<TensorRank2<D, L, M>> for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, I, J, K, M>;
    fn mul(self, tensor_rank_2: TensorRank2<D, L, M>) -> Self::Output {
        self.into_iter()
            .map(|self_i| {
                self_i
                    .into_iter()
                    .map(|self_ij| self_ij * &tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<&TensorRank2<D, L, M>> for TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, I, J, K, M>;
    fn mul(self, tensor_rank_2: &TensorRank2<D, L, M>) -> Self::Output {
        self.into_iter()
            .map(|self_i| {
                self_i
                    .into_iter()
                    .map(|self_ij| self_ij * tensor_rank_2)
                    .collect()
            })
            .collect()
    }
}

impl<const D: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<TensorRank4<D, M, J, K, L>> for TensorRank1<D, M>
{
    type Output = TensorRank3<D, J, K, L>;
    fn mul(self, tensor_rank_4: TensorRank4<D, M, J, K, L>) -> Self::Output {
        self.into_iter()
            .zip(tensor_rank_4)
            .map(|(self_m, tensor_rank_4_m)| tensor_rank_4_m * self_m)
            .sum()
    }
}

impl<const D: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<&TensorRank4<D, M, J, K, L>> for TensorRank1<D, M>
{
    type Output = TensorRank3<D, J, K, L>;
    fn mul(self, tensor_rank_4: &TensorRank4<D, M, J, K, L>) -> Self::Output {
        let foo = 1; // get rid of clone by doing other impl
        self.into_iter()
            .zip(tensor_rank_4.iter())
            .map(|(self_m, tensor_rank_4_m)| tensor_rank_4_m.clone() * self_m)
            .sum()
    }
}

impl<const D: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<TensorRank4<D, M, J, K, L>> for &TensorRank1<D, M>
{
    type Output = TensorRank3<D, J, K, L>;
    fn mul(self, tensor_rank_4: TensorRank4<D, M, J, K, L>) -> Self::Output {
        self.iter()
            .zip(tensor_rank_4)
            .map(|(self_m, tensor_rank_4_m)| tensor_rank_4_m * self_m)
            .sum()
    }
}

impl<const D: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<&TensorRank4<D, M, J, K, L>> for &TensorRank1<D, M>
{
    type Output = TensorRank3<D, J, K, L>;
    fn mul(self, tensor_rank_4: &TensorRank4<D, M, J, K, L>) -> Self::Output {
        let foo = 1; // get rid of clone by doing other impl
        self.iter()
            .zip(tensor_rank_4.iter())
            .map(|(self_m, tensor_rank_4_m)| tensor_rank_4_m.clone() * self_m)
            .sum()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<TensorRank4<D, M, J, K, L>> for TensorRank2<D, I, M>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn mul(self, tensor_rank_4: TensorRank4<D, M, J, K, L>) -> Self::Output {
        self.into_iter()
            .map(|self_i| self_i * &tensor_rank_4)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<&TensorRank4<D, M, J, K, L>> for TensorRank2<D, I, M>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn mul(self, tensor_rank_4: &TensorRank4<D, M, J, K, L>) -> Self::Output {
        self.into_iter()
            .map(|self_i| self_i * tensor_rank_4)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<TensorRank4<D, M, J, K, L>> for &TensorRank2<D, I, M>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn mul(self, tensor_rank_4: TensorRank4<D, M, J, K, L>) -> Self::Output {
        self.iter().map(|self_i| self_i * &tensor_rank_4).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize>
    Mul<&TensorRank4<D, M, J, K, L>> for &TensorRank2<D, I, M>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn mul(self, tensor_rank_4: &TensorRank4<D, M, J, K, L>) -> Self::Output {
        self.iter().map(|self_i| self_i * tensor_rank_4).collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Add
    for TensorRank4<D, I, J, K, L>
{
    type Output = Self;
    fn add(mut self, tensor_rank_4: Self) -> Self::Output {
        self += tensor_rank_4;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Add<&Self>
    for TensorRank4<D, I, J, K, L>
{
    type Output = Self;
    fn add(mut self, tensor_rank_4: &Self) -> Self::Output {
        self += tensor_rank_4;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    Add<TensorRank4<D, I, J, K, L>> for &TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn add(self, mut tensor_rank_4: TensorRank4<D, I, J, K, L>) -> Self::Output {
        tensor_rank_4 += self;
        tensor_rank_4
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> AddAssign
    for TensorRank4<D, I, J, K, L>
{
    fn add_assign(&mut self, tensor_rank_4: Self) {
        self.iter_mut()
            .zip(tensor_rank_4)
            .for_each(|(self_i, tensor_rank_4_i)| *self_i += tensor_rank_4_i);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    AddAssign<&Self> for TensorRank4<D, I, J, K, L>
{
    fn add_assign(&mut self, tensor_rank_4: &Self) {
        self.iter_mut()
            .zip(tensor_rank_4.iter())
            .for_each(|(self_i, tensor_rank_4_i)| *self_i += tensor_rank_4_i);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Sub
    for TensorRank4<D, I, J, K, L>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_4: Self) -> Self::Output {
        self -= tensor_rank_4;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Sub<&Self>
    for TensorRank4<D, I, J, K, L>
{
    type Output = Self;
    fn sub(mut self, tensor_rank_4: &Self) -> Self::Output {
        self -= tensor_rank_4;
        self
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> Sub
    for &TensorRank4<D, I, J, K, L>
{
    type Output = TensorRank4<D, I, J, K, L>;
    fn sub(self, tensor_rank_4: Self) -> Self::Output {
        tensor_rank_4
            .iter()
            .zip(self.iter())
            .map(|(tensor_rank_4_i, self_i)| self_i - tensor_rank_4_i)
            .collect()
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize> SubAssign
    for TensorRank4<D, I, J, K, L>
{
    fn sub_assign(&mut self, tensor_rank_4: Self) {
        self.iter_mut()
            .zip(tensor_rank_4)
            .for_each(|(self_i, tensor_rank_4_i)| *self_i -= tensor_rank_4_i);
    }
}

impl<const D: usize, const I: usize, const J: usize, const K: usize, const L: usize>
    SubAssign<&Self> for TensorRank4<D, I, J, K, L>
{
    fn sub_assign(&mut self, tensor_rank_4: &Self) {
        self.iter_mut()
            .zip(tensor_rank_4.iter())
            .for_each(|(self_i, tensor_rank_4_i)| *self_i -= tensor_rank_4_i);
    }
}
