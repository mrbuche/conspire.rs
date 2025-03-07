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
        ContractFirstSecondIndicesWithSecondIndicesOf, ContractSecondIndexWithFirstIndexOf, Rank2,
        Tensor, TensorArray, TensorRank4, IDENTITY, IDENTITY_00, IDENTITY_10, IDENTITY_1010,
        ZERO_10,
    },
    mechanics::{
        CauchyRateTangentStiffness, CauchyStress, CauchyStresses, CauchyTangentStiffness,
        DeformationGradient, DeformationGradientRate, DeformationGradients,
        FirstPiolaKirchhoffRateTangentStiffness, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, LeftCauchyGreenDeformation,
        RightCauchyGreenDeformation, Scalar, SecondPiolaKirchhoffRateTangentStiffness,
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
    /// Calculates and returns the left Cauchy-Green deformation.
    ///
    /// ```math
    /// \mathbf{B} = \mathbf{F}\cdot\mathbf{F}^T
    /// ```
    fn left_cauchy_green_deformation(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> LeftCauchyGreenDeformation {
        deformation_gradient
            .iter()
            .map(|deformation_gradient_i| {
                deformation_gradient
                    .iter()
                    .map(|deformation_gradient_j| deformation_gradient_i * deformation_gradient_j)
                    .collect()
            })
            .collect()
    }
    /// Calculates and returns the right Cauchy-Green deformation.
    ///
    /// ```math
    /// \mathbf{C} = \mathbf{F}^T\cdot\mathbf{F}
    /// ```
    fn right_cauchy_green_deformation(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> RightCauchyGreenDeformation {
        let deformation_gradient_transpose = deformation_gradient.transpose();
        deformation_gradient_transpose
            .iter()
            .map(|deformation_gradient_transpose_i| {
                deformation_gradient_transpose
                    .iter()
                    .map(|deformation_gradient_transpose_j| {
                        deformation_gradient_transpose_i * deformation_gradient_transpose_j
                    })
                    .collect()
            })
            .collect()
    }
    /// Returns the bulk modulus.
    fn bulk_modulus(&self) -> &Scalar;
    /// Returns the shear modulus.
    fn shear_modulus(&self) -> &Scalar;
}
