//! Finite element library.

pub mod block;

use crate::{
    math::{
        Scalar, Tensor, TensorRank1List, TensorRank1List2D, TensorRank1Vec, TensorRank1Vec2D,
        TensorRank2, TensorRank2List, TensorRank2List2D, TensorRank2Vec2D,
    },
    mechanics::{Forces, Stiffnesses, Vectors, Vectors2D},
};

pub type Connectivity<const N: usize> = Vec<[usize; N]>;
pub type NodalCoordinates = TensorRank1Vec<3, 1>;
pub type NodalVelocitiesBlock = TensorRank1Vec<3, 1>;
pub type NodalForcesSolid = TensorRank1Vec<3, 1>;
pub type NodalStiffnessesSolid = TensorRank2Vec2D<3, 1, 1>;
pub type ReferenceNodalCoordinates = TensorRank1Vec<3, 0>;

pub type NodalCoordinatesHistory = TensorRank1Vec2D<3, 1>;
pub type NodalVelocitiesHistory = TensorRank1Vec2D<3, 1>;

type Bases<const I: usize, const P: usize> = TensorRank1List2D<3, I, 2, P>;
type GradientVectors<const G: usize, const N: usize> = Vectors2D<0, N, G>;
type ElementNodalForcesSolid<const D: usize> = Forces<D>;
type ElementNodalStiffnessesSolid<const D: usize> = Stiffnesses<D>;
type Normals<const P: usize> = Vectors<1, P>;
type NormalGradients<const O: usize, const P: usize> = TensorRank2List2D<3, 1, 1, O, P>;
type NormalRates<const P: usize> = Vectors<1, P>;
type NormalizedProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
type ParametricGradientOperators<const P: usize> = TensorRank2List<3, 0, 0, P>;
type ProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
type ReferenceNormals<const P: usize> = Vectors<0, P>;
type ShapeFunctionIntegrals<const P: usize, const Q: usize> = TensorRank1List<Q, 9, P>;
type ShapeFunctionIntegralsProducts<const P: usize, const Q: usize> = TensorRank2List<Q, 9, 9, P>;
type ShapeFunctionsAtIntegrationPoints<const G: usize, const Q: usize> = TensorRank1List<Q, 9, G>;
type StandardGradientOperators<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 0, O, P>;
type StandardGradientOperatorsTransposed<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 0, P, O>;
