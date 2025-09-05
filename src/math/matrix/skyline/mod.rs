
use crate::{
    ABS_TOL,
    math::{
        Hessian, Rank2, Scalar, Tensor, TensorRank0, TensorRank2Vec2D, TensorVec, Vector,
        write_tensor_rank_0,
    },
};
use std::{
    collections::VecDeque,
    fmt::{self, Display, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    vec::IntoIter,
};

/// A skyline matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SkylineMatrix{
    values: Vec<f64>,
    col_heights: Vec<usize>,
    col_offsets: Vec<usize>,
}
