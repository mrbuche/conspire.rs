//! Mathematics library.

#[cfg(test)]
pub mod test;

/// Special functions.
pub mod special;

/// Integration and ODEs.
pub mod integrate;

/// Optimization and root finding.
pub mod optimize;

mod matrix;
mod tensor;

pub const FOUR_THIRDS: TensorRank0 = 4.0 / 3.0;
pub const FIVE_THIRDS: TensorRank0 = 5.0 / 3.0;
pub const NINE_FIFTHS: TensorRank0 = 9.0 / 5.0;
pub const ONE_SIXTH: TensorRank0 = 1.0 / 6.0;
pub const ONE_TWENTY_FOURTH: TensorRank0 = 1.0 / 24.0;
pub const SEVEN_THIRDS: TensorRank0 = 7.0 / 3.0;
pub const TWO_THIRDS: TensorRank0 = 2.0 / 3.0;

pub use matrix::{square::SquareMatrix, vector::Vector};
pub use tensor::{
    rank_0::{list::TensorRank0List, TensorRank0},
    rank_1::{
        list::TensorRank1List, list_2d::TensorRank1List2D, vec::TensorRank1Vec,
        zero as tensor_rank_1_zero, TensorRank1,
    },
    rank_2::{
        list::TensorRank2List, list_2d::TensorRank2List2D, vec::TensorRank2Vec,
        vec_2d::TensorRank2Vec2D, TensorRank2,
    },
    rank_3::{
        levi_civita, list::TensorRank3List, list_2d::TensorRank3List2D, list_3d::TensorRank3List3D,
        TensorRank3,
    },
    rank_4::{
        list::TensorRank4List, ContractAllIndicesWithFirstIndicesOf,
        ContractFirstSecondIndicesWithSecondIndicesOf,
        ContractFirstThirdFourthIndicesWithFirstIndicesOf,
        ContractSecondFourthIndicesWithFirstIndicesOf, ContractSecondIndexWithFirstIndexOf,
        ContractThirdFourthIndicesWithFirstSecondIndicesOf, TensorRank4,
    },
    Convert, Hessian, Rank2, Tensor, TensorArray, TensorVec,
};

use std::fmt;

fn write_tensor_rank_0(f: &mut fmt::Formatter, tensor_rank_0: &TensorRank0) -> fmt::Result {
    let num = (tensor_rank_0 * 1e6).round() / 1e6;
    let num_abs = num.abs();
    if num.is_nan() {
        write!(f, "{:>14}, ", num)
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
    } else if num_abs <= 1e10 {
        let temp_1 = format!("{:>11.6e}, ", num).to_string();
        let mut temp_2 = temp_1.split("e");
        let a = temp_2.next().unwrap();
        let b = temp_2.next().unwrap();
        write!(f, "{}e+00{}", a, b)
    } else {
        println!("{}", num);
        panic!("\n\n{}\n\n", num)
    }
}
