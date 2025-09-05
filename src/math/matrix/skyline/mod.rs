use crate::math::Scalar;

/// A skyline matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SkylineMatrix {
    values: Vec<Scalar>,
    col_heights: Vec<usize>,
    col_offsets: Vec<usize>,
}
