//! Hyperelastic constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(test)]
pub mod test;

mod arruda_boyce;
mod fung;
mod gent;
mod mooney_rivlin;
mod neo_hookean;
mod saint_venant_kirchhoff;
mod yeoh;

pub use self::{
    arruda_boyce::ArrudaBoyce, fung::Fung, gent::Gent, mooney_rivlin::MooneyRivlin,
    neo_hookean::NeoHookean, saint_venant_kirchhoff::SaintVenantKirchhoff, yeoh::Yeoh,
};
use super::{
    elastic::{AppliedLoad, Elastic},
    *,
};
use crate::math::{
    Matrix, TensorVec, Vector,
    optimize::{
        EqualityConstraint, FirstOrderOptimization, OptimizeError, SecondOrderOptimization,
    },
};

/// Required methods for hyperelastic constitutive models.
pub trait Hyperelastic
where
    Self: Elastic,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError>;
}

/// First-order minimization methods for elastic constitutive models.
pub trait FirstOrderMinimize {
    /// Solve for the unknown components of the deformation gradient under an applied load.
    ///
    /// ```math
    /// \Pi(\mathbf{F},\boldsymbol{\lambda}) = a(\mathbf{F}) - \boldsymbol{\lambda}:(\mathbf{F} - \mathbf{F}_0) - \mathbf{P}_0:\mathbf{F}
    /// ```
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderOptimization<Scalar, DeformationGradient>,
    ) -> Result<DeformationGradient, OptimizeError>;
}

/// Second-order minimization methods for elastic constitutive models.
pub trait SecondOrderMinimize {
    /// Solve for the unknown components of the deformation gradient under an applied load.
    ///
    /// ```math
    /// \Pi(\mathbf{F},\boldsymbol{\lambda}) = a(\mathbf{F}) - \boldsymbol{\lambda}:(\mathbf{F} - \mathbf{F}_0) - \mathbf{P}_0:\mathbf{F}
    /// ```
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl SecondOrderOptimization<
            Scalar,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<DeformationGradient, OptimizeError>;
}

impl<T> FirstOrderMinimize for T
where
    T: Hyperelastic,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl FirstOrderOptimization<Scalar, DeformationGradient>,
    ) -> Result<DeformationGradient, OptimizeError> {
        match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_11) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                vector[0] = deformation_gradient_11;
                solver.minimize(
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.helmholtz_free_energy_density(deformation_gradient)?)
                    },
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.first_piola_kirchhoff_stress(deformation_gradient)?)
                    },
                    DeformationGradient::identity(),
                    EqualityConstraint::Linear(matrix, vector),
                )
            }
            AppliedLoad::BiaxialStress(deformation_gradient_11, deformation_gradient_22) => {
                let mut matrix = Matrix::zero(5, 9);
                let mut vector = Vector::zero(5);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                matrix[4][4] = 1.0;
                vector[0] = deformation_gradient_11;
                vector[4] = deformation_gradient_22;
                solver.minimize(
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.helmholtz_free_energy_density(deformation_gradient)?)
                    },
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.first_piola_kirchhoff_stress(deformation_gradient)?)
                    },
                    DeformationGradient::identity(),
                    EqualityConstraint::Linear(matrix, vector),
                )
            }
        }
    }
}

impl<T> SecondOrderMinimize for T
where
    T: Hyperelastic,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        solver: impl SecondOrderOptimization<
            Scalar,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<DeformationGradient, OptimizeError> {
        match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_11) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                vector[0] = deformation_gradient_11;
                solver.minimize(
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.helmholtz_free_energy_density(deformation_gradient)?)
                    },
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.first_piola_kirchhoff_stress(deformation_gradient)?)
                    },
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?)
                    },
                    DeformationGradient::identity(),
                    EqualityConstraint::Linear(matrix, vector),
                    None,
                )
            }
            AppliedLoad::BiaxialStress(deformation_gradient_11, deformation_gradient_22) => {
                let mut matrix = Matrix::zero(5, 9);
                let mut vector = Vector::zero(5);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                matrix[4][4] = 1.0;
                vector[0] = deformation_gradient_11;
                vector[4] = deformation_gradient_22;
                solver.minimize(
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.helmholtz_free_energy_density(deformation_gradient)?)
                    },
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.first_piola_kirchhoff_stress(deformation_gradient)?)
                    },
                    |deformation_gradient: &DeformationGradient| {
                        Ok(self.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?)
                    },
                    DeformationGradient::identity(),
                    EqualityConstraint::Linear(matrix, vector),
                    None,
                )
            }
        }
    }
}
