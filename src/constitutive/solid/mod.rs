//! Solid constitutive models.

pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod thermoelastic;
pub mod thermohyperelastic;
pub mod viscoelastic;

const TWO_THIRDS: Scalar = 2.0 / 3.0;
const FIVE_THIRDS: Scalar = 5.0 / 3.0;

use crate::{
    constitutive::{Constitutive, ConstitutiveError},
    math::{
        ContractFirstSecondIndicesWithSecondIndicesOf, ContractSecondIndexWithFirstIndexOf,
        IDENTITY, IDENTITY_00, Rank2, Tensor, TensorArray, ZERO_10,
    },
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, CauchyTangentStiffness, Deformation,
        DeformationGradient, DeformationGradientRate, DeformationGradientRates,
        DeformationGradients, FirstPiolaKirchhoffRateTangentStiffness, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Scalar, SecondPiolaKirchhoffRateTangentStiffness,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness, Times,
    },
};
use std::fmt::Debug;

impl<C> Constitutive for C where C: Solid {}

/// Required methods for solid constitutive models.
pub trait Solid
where
    Self: Debug,
{
    /// Returns the bulk modulus.
    fn bulk_modulus(&self) -> Scalar;
    /// Returns the shear modulus.
    fn shear_modulus(&self) -> Scalar;
}
