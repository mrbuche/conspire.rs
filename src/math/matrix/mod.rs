pub mod square;
pub mod vector;

use crate::math::{TensorVec, TensorRank0};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};
use vector::Vector;
