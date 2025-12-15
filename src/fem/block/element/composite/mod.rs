mod tetrahedron;

pub use tetrahedron::Tetrahedron;

use crate::math::{TensorRank1List, TensorRank2, TensorRank2List};

pub type NormalizedProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
pub type ParametricGradientOperators<const P: usize> = TensorRank2List<3, 0, 0, P>;
pub type ProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
pub type ShapeFunctionIntegrals<const P: usize, const Q: usize> = TensorRank1List<Q, 9, P>;
pub type ShapeFunctionIntegralsProducts<const P: usize, const Q: usize> =
    TensorRank2List<Q, 9, 9, P>;
