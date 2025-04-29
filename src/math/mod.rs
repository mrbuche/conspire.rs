//! Mathematics library.

#[cfg(test)]
pub mod test;

/// Special functions.
pub mod special;

/// Integration and ODEs.
pub mod integrate;

/// Interpolation schemes.
pub mod interpolate;

/// Optimization and root finding.
pub mod optimize;

mod matrix;
mod tensor;

pub use matrix::{Matrix, square::SquareMatrix, vector::Vector};
pub use tensor::{
    Hessian, Rank2, Tensor, TensorArray, TensorVec,
    rank_0::{
        TensorRank0,
        list::{TensorRank0List, tensor_rank_0_list},
    },
    rank_1::{
        TensorRank1,
        list::{TensorRank1List, tensor_rank_1_list},
        list_2d::{TensorRank1List2D, tensor_rank_1_list_2d},
        sparse::TensorRank1Sparse,
        tensor_rank_1,
        vec::TensorRank1Vec,
        zero as tensor_rank_1_zero,
    },
    rank_2::{
        IDENTITY, IDENTITY_00, IDENTITY_10, TensorRank2, ZERO, ZERO_10,
        list::{TensorRank2List, tensor_rank_2_list},
        list_2d::TensorRank2List2D,
        tensor_rank_2,
        vec::TensorRank2Vec,
        vec_2d::TensorRank2Vec2D,
    },
    rank_3::{
        LEVI_CIVITA, TensorRank3, levi_civita, list::TensorRank3List, list_2d::TensorRank3List2D,
        list_3d::TensorRank3List3D,
    },
    rank_4::{
        ContractAllIndicesWithFirstIndicesOf, ContractFirstSecondIndicesWithSecondIndicesOf,
        ContractFirstThirdFourthIndicesWithFirstIndicesOf,
        ContractSecondFourthIndicesWithFirstIndicesOf, ContractSecondIndexWithFirstIndexOf,
        ContractThirdFourthIndicesWithFirstSecondIndicesOf, IDENTITY_1010, TensorRank4,
        list::TensorRank4List,
    },
};

use std::fmt;

fn write_tensor_rank_0(f: &mut fmt::Formatter, tensor_rank_0: &TensorRank0) -> fmt::Result {
    let num = if tensor_rank_0.abs() > 1e-1 {
        (tensor_rank_0 * 1e6).round() / 1e6
    } else {
        *tensor_rank_0
    };
    let num_abs = num.abs();
    if num.is_nan() {
        write!(f, "{:>11}, ", num)
    } else if num == 0.0 || num_abs == 1.0 {
        let temp_1 = format!("{:>11.6e}, ", num).to_string();
        let mut temp_2 = temp_1.split("e");
        let a = temp_2.next().unwrap();
        let b = temp_2.next().unwrap();
        write!(f, "{}e+00{}", a, b)
    } else if num_abs <= 1e-100 {
        write!(f, "{:>14.6e}, ", num)
    } else if num_abs >= 1e100 {
        let temp_1 = format!("{:>13.6e}, ", num).to_string();
        let mut temp_2 = temp_1.split("e");
        let a = temp_2.next().unwrap();
        let b = temp_2.next().unwrap();
        write!(f, "{}e+{}", a, b)
    } else if num_abs <= 1e-10 {
        let temp_1 = format!("{:>13.6e}, ", num).to_string();
        let mut temp_2 = temp_1.split("e");
        let a = temp_2.next().unwrap();
        let b = temp_2.next().unwrap();
        let mut c = b.split("-");
        c.next();
        let e = c.next().unwrap();
        write!(f, "{}e-0{}", a, e)
    } else if num_abs >= 1e10 {
        let temp_1 = format!("{:>12.6e}, ", num).to_string();
        let mut temp_2 = temp_1.split("e");
        let a = temp_2.next().unwrap();
        let b = temp_2.next().unwrap();
        write!(f, "{}e+0{}", a, b)
    } else if num_abs <= 1e0 {
        let temp_1 = format!("{:>12.6e}, ", num).to_string();
        let mut temp_2 = temp_1.split("e");
        let a = temp_2.next().unwrap();
        let b = temp_2.next().unwrap();
        let mut c = b.split("-");
        c.next();
        let e = c.next().unwrap();
        write!(f, "{}e-00{}", a, e)
    } else {
        let temp_1 = format!("{:>11.6e}, ", num).to_string();
        let mut temp_2 = temp_1.split("e");
        let a = temp_2.next().unwrap();
        let b = temp_2.next().unwrap();
        write!(f, "{}e+00{}", a, b)
    }
}
