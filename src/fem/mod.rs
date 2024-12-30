//! Finite element library.

mod block;

pub use block::{
    element::{
        composite::{
            localization::{
                wedge::Wedge as CompositeWedgeLocalization, CompositeLocalizationElement,
            },
            surface::{triangle::Triangle as CompositeTriangle, CompositeSurfaceElement},
            tetrahedron::Tetrahedron as CompositeTetrahedron,
            CompositeElement, ElasticCompositeElement, ElasticHyperviscousCompositeElement,
            HyperelasticCompositeElement, HyperviscoelasticCompositeElement,
            ViscoelasticCompositeElement,
        },
        linear::{
            cohesive::{wedge::Wedge as LinearWedgeCohesive, LinearCohesiveElement},
            localization::{wedge::Wedge as LinearWedgeLocalization, LinearLocalizationElement},
            surface::{triangle::Triangle as LinearTriangle, LinearSurfaceElement},
            tetrahedron::Tetrahedron as LinearTetrahedron,
            ElasticHyperviscousLinearElement, ElasticLinearElement, HyperelasticLinearElement,
            HyperviscoelasticLinearElement, LinearElement, ViscoelasticLinearElement,
        },
        CohesiveElement, ElasticFiniteElement, FiniteElement, HyperelasticFiniteElement,
        HyperviscoelasticFiniteElement, SurfaceElement, ViscoelasticFiniteElement,
    },
    BasicFiniteElementBlock, ElasticBlock, ElasticFiniteElementBlock, FiniteElementBlock,
    HyperelasticFiniteElementBlock, HyperviscoelasticFiniteElementBlock, SurfaceElementBlock,
    ViscoelasticBlock, ViscoelasticFiniteElementBlock,
};

use crate::{
    constitutive::{
        cohesive::Cohesive,
        solid::{
            elastic::Elastic, elastic_hyperviscous::ElasticHyperviscous,
            hyperelastic::Hyperelastic, hyperviscoelastic::Hyperviscoelastic,
            viscoelastic::Viscoelastic,
        },
        Constitutive, ConstitutiveError, Parameters,
    },
    math::{
        tensor_rank_1_zero, ContractSecondFourthIndicesWithFirstIndicesOf, Tensor, TensorArray,
        TensorRank1, TensorRank1List, TensorRank1List2D, TensorRank1Vec, TensorRank2,
        TensorRank2List, TensorRank2List2D, TensorRank2Vec2D, TensorRank3, TensorRank3List,
        TensorRank3List2D, TensorRank3List3D, TensorVec, ONE_SIXTH, ONE_TWENTY_FOURTH,
    },
    mechanics::{
        Coordinates, CurrentCoordinates, DeformationGradient, DeformationGradientRate,
        DeformationGradientRates, DeformationGradients, DeformationGradientss, Displacement,
        FirstPiolaKirchoffRateTangentStiffnesses, FirstPiolaKirchoffStresses,
        FirstPiolaKirchoffTangentStiffnesses, Forces, ReferenceCoordinates, Scalar, Scalars,
        Stiffnesses, Vector, Vectors, Vectors2D, IDENTITY, LEVI_CIVITA, ZERO_VECTOR,
    },
};

type NodalCoordinatesBlock = TensorRank1Vec<3, 1>;
type ReferenceNodalCoordinatesBlock = TensorRank1Vec<3, 0>;
type NodalVelocitiesBlock = TensorRank1Vec<3, 1>;
type NodalForcesBlock = TensorRank1Vec<3, 1>;
type NodalStiffnessesBlock = TensorRank2Vec2D<3, 1, 1>;

type Basis<const I: usize> = Vectors<I, 2>;
type Bases<const I: usize, const P: usize> = TensorRank1List2D<3, I, 2, P>;
type Connectivity<const E: usize, const N: usize> = [[usize; N]; E];
type GradientVectors<const N: usize> = Vectors<0, N>;
type NodalCoordinates<const D: usize> = CurrentCoordinates<D>;
type NodalForces<const D: usize> = Forces<D>;
type NodalStiffnesses<const D: usize> = Stiffnesses<D>;
type NodalVelocities<const D: usize> = CurrentCoordinates<D>;
type Normal = Vector<1>;
type Normals<const P: usize> = Vectors<1, P>;
type NormalGradients<const O: usize> = TensorRank2List<3, 1, 1, O>;
type NormalGradientss<const P: usize, const O: usize> = TensorRank2List2D<3, 1, 1, O, P>;
type NormalizedProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
type NormalRate = Vector<1>;
type NormalRates<const P: usize> = Vectors<1, P>;
type NormalTangents<const O: usize> = TensorRank3List2D<3, 1, 1, 1, O, O>;
type NormalTangentss<const P: usize, const O: usize> = TensorRank3List3D<3, 1, 1, 1, O, O, P>;
type ParametricGradientOperators<const P: usize> = TensorRank2List<3, 0, 9, P>;
type ProjectedGradientVectors<const G: usize, const N: usize> = Vectors2D<0, N, G>;
type ProjectionMatrix<const Q: usize> = TensorRank2<Q, 9, 9>;
type ReferenceNodalCoordinates<const D: usize> = ReferenceCoordinates<D>;
type ReferenceNormal = Vector<0>;
type ReferenceNormals<const P: usize> = Vectors<0, P>;
type ScaledReferenceNormals<const G: usize, const P: usize> = TensorRank1List2D<3, 0, P, G>;
type ShapeFunctionIntegrals<const P: usize, const Q: usize> = TensorRank1List<Q, 9, P>;
type ShapeFunctionIntegralsProducts<const P: usize, const Q: usize> = TensorRank2List<Q, 9, 9, P>;
type ShapeFunctionIntegralsProductsMixed<const O: usize, const P: usize> =
    TensorRank1List2D<3, 9, P, O>;
type ShapeFunctionsAtIntegrationPoints<const G: usize, const Q: usize> = TensorRank1List<Q, 9, G>;
type StandardGradientOperator<const M: usize, const O: usize> = TensorRank1List<M, 9, O>;
type StandardGradientOperators<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 9, O, P>;
type StandardGradientOperatorsTransposed<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 9, P, O>;
