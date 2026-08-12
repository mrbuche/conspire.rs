use crate::math::{Projection, Reference};
mod tetrahedron;

pub use tetrahedron::Tetrahedron;

use crate::{
    fem::block::element::Element,
    math::{TensorRank1List, TensorRank2, TensorRank2List, unit::Length},
};

pub type CompositeElement<const G: usize, const N: usize> = Element<3, G, N, 0>;

pub type NormalizedProjectionMatrix<const Q: usize> = TensorRank2<Q, Projection, Projection>;
pub type ParametricGradientOperators<const P: usize> =
    TensorRank2List<3, Reference, Reference, P, Length>;
pub type ProjectionMatrix<const Q: usize> = TensorRank2<Q, Projection, Projection>;
pub type ShapeFunctionIntegrals<const P: usize, const Q: usize> = TensorRank1List<Q, Projection, P>;
pub type ShapeFunctionIntegralsProducts<const P: usize, const Q: usize> =
    TensorRank2List<Q, Projection, Projection, P>;
