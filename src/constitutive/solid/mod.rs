//! Solid constitutive models.

pub mod elastic;
pub mod elastic_hyperviscous;
pub mod hyperelastic;
pub mod hyperviscoelastic;
pub mod thermoelastic;
pub mod thermohyperelastic;
pub mod viscoelastic;

const TWO_THIRDS: Scalar = 2.0 / 3.0;
const FIVE_THIRDS: Scalar = 5.0 / 3.0;

use super::{Constitutive, Parameters};
use crate::{
    constitutive::ConstitutiveError,
    math::{
        ContractFirstSecondIndicesWithSecondIndicesOf, ContractSecondIndexWithFirstIndexOf,
        IDENTITY, IDENTITY_00, IDENTITY_10, IDENTITY_1010, Rank2, Tensor, TensorArray, TensorRank4,
        ZERO_10,
    },
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, CauchyStresses, CauchyTangentStiffness,
        Deformation, DeformationGradient, DeformationGradientRate, DeformationGradients,
        FirstPiolaKirchhoffRateTangentStiffness, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Scalar, SecondPiolaKirchhoffRateTangentStiffness,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
};

/// Possible applied loads.
pub enum AppliedLoad {
    /// Uniaxial stress given $`F_{11}`$.
    UniaxialStress(Scalar),
    /// Biaxial stress given $`F_{11}`$ and $`F_{22}`$.
    BiaxialStress(Scalar, Scalar),
}

/// Required methods for solid constitutive models.
pub trait Solid<'a>
where
    Self: Constitutive<'a>,
{
    /// Returns the bulk modulus.
    fn bulk_modulus(&self) -> &Scalar;
    /// Returns the shear modulus.
    fn shear_modulus(&self) -> &Scalar;
}
