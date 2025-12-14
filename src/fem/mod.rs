//! Finite element library.

pub mod block;

use crate::math::{
    Scalar, Tensor, TensorRank1List, TensorRank1Vec, TensorRank1Vec2D, TensorRank2,
    TensorRank2List, TensorRank2Vec2D,
};

pub type NodalCoordinates = TensorRank1Vec<3, 1>;
pub type NodalCoordinatesHistory = TensorRank1Vec2D<3, 1>;
pub type NodalForcesSolid = TensorRank1Vec<3, 1>;
pub type NodalReferenceCoordinates = TensorRank1Vec<3, 0>;
pub type NodalStiffnessesSolid = TensorRank2Vec2D<3, 1, 1>;
pub type NodalVelocities = TensorRank1Vec<3, 1>;
pub type NodalVelocitiesHistory = TensorRank1Vec2D<3, 1>;

type NormalizedProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
type ParametricGradientOperators<const P: usize> = TensorRank2List<3, 0, 0, P>;
type ProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
type ShapeFunctionIntegrals<const P: usize, const Q: usize> = TensorRank1List<Q, 9, P>;
type ShapeFunctionIntegralsProducts<const P: usize, const Q: usize> = TensorRank2List<Q, 9, 9, P>;
type ShapeFunctionsAtIntegrationPoints<const G: usize, const Q: usize> = TensorRank1List<Q, 9, G>;
