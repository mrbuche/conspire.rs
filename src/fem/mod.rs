//! Finite element library.

mod block;

pub use block::{
    ElasticFiniteElementBlock, ElementBlock, FiniteElementBlock, HyperelasticFiniteElementBlock,
    HyperviscoelasticFiniteElementBlock, SurfaceFiniteElementBlock, ViscoelasticFiniteElementBlock,
    element::{
        ElasticFiniteElement, FiniteElement, FiniteElementMethods, HyperelasticFiniteElement,
        HyperviscoelasticFiniteElement, SurfaceFiniteElement, ViscoelasticFiniteElement,
        composite::tetrahedron::Tetrahedron as CompositeTetrahedron,
        linear::{
            tetrahedron::Tetrahedron as LinearTetrahedron, triangle::Triangle as LinearTriangle,
        },
    },
};

use crate::{
    constitutive::{
        Constitutive, ConstitutiveError, Parameters,
        solid::{
            elastic::Elastic, elastic_hyperviscous::ElasticHyperviscous,
            hyperelastic::Hyperelastic, hyperviscoelastic::Hyperviscoelastic,
            viscoelastic::Viscoelastic,
        },
    },
    math::{
        ContractSecondFourthIndicesWithFirstIndicesOf, Tensor, TensorRank1, TensorRank1List,
        TensorRank1List2D, TensorRank1Vec, TensorRank2, TensorRank2List, TensorRank2List2D,
        TensorRank2Vec2D, TensorVec,
    },
    mechanics::{
        Coordinates, CurrentCoordinates, DeformationGradient, DeformationGradientRate,
        DeformationGradientRates, DeformationGradients, FirstPiolaKirchhoffRateTangentStiffnesses,
        FirstPiolaKirchhoffStresses, FirstPiolaKirchhoffTangentStiffnesses, Forces,
        ReferenceCoordinates, Scalar, Scalars, Stiffnesses, Vectors, Vectors2D,
    },
};

pub type Connectivity<const N: usize> = Vec<[usize; N]>;
pub type NodalCoordinatesBlock = TensorRank1Vec<3, 1>;
pub type NodalForcesBlock = TensorRank1Vec<3, 1>;
pub type ReferenceNodalCoordinatesBlock = TensorRank1Vec<3, 0>;

type NodalVelocitiesBlock = TensorRank1Vec<3, 1>;
type NodalStiffnessesBlock = TensorRank2Vec2D<3, 1, 1>;

type Bases<const I: usize, const P: usize> = TensorRank1List2D<3, I, 2, P>;
type GradientVectors<const G: usize, const N: usize> = Vectors2D<0, N, G>;
type NodalCoordinates<const D: usize> = CurrentCoordinates<D>;
type NodalForces<const D: usize> = Forces<D>;
type NodalStiffnesses<const D: usize> = Stiffnesses<D>;
type NodalVelocities<const D: usize> = CurrentCoordinates<D>;
type Normals<const P: usize> = Vectors<1, P>;
type NormalGradients<const O: usize, const P: usize> = TensorRank2List2D<3, 1, 1, O, P>;
type NormalRates<const P: usize> = Vectors<1, P>;
type NormalizedProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
type ParametricGradientOperators<const P: usize> = TensorRank2List<3, 0, 9, P>;
type ProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
type ReferenceNodalCoordinates<const D: usize> = ReferenceCoordinates<D>;
type ReferenceNormals<const P: usize> = Vectors<0, P>;
type ShapeFunctionIntegrals<const P: usize, const Q: usize> = TensorRank1List<Q, 9, P>;
type ShapeFunctionIntegralsProducts<const P: usize, const Q: usize> = TensorRank2List<Q, 9, 9, P>;
type ShapeFunctionsAtIntegrationPoints<const G: usize, const Q: usize> = TensorRank1List<Q, 9, G>;
type StandardGradientOperators<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 9, O, P>;
type StandardGradientOperatorsTransposed<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 9, P, O>;
