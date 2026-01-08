//! Mechanics library.

#[cfg(test)]
pub mod test;

use crate::{
    defeat_message,
    math::{
        Rank2, Tensor, TensorRank1, TensorRank1List, TensorRank1List2D, TensorRank1RefVec,
        TensorRank1Vec, TensorRank1Vec2D, TensorRank2, TensorRank2List, TensorRank2List2D,
        TensorRank2Vec, TensorRank2Vec2D, TensorRank4, TensorRank4List, TensorRank4Vec,
    },
};
use std::fmt::{self, Debug, Display, Formatter};

pub use crate::math::Scalar;

/// Possible errors for deformation gradients.
pub enum DeformationError {
    InvalidJacobian(Scalar),
}

impl Debug for DeformationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InvalidJacobian(jacobian) => {
                format!("\x1b[1;91mInvalid Jacobian: {jacobian:.6e}.\x1b[0;91m")
            }
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for DeformationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InvalidJacobian(jacobian) => {
                format!("\x1b[1;91mInvalid Jacobian: {jacobian:.6e}.\x1b[0;91m")
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

/// Methods for deformation gradients.
pub trait Deformation<const I: usize, const J: usize> {
    /// Calculates and returns the Jacobian.
    ///
    /// ```math
    /// J = \mathrm{det}(\mathbf{F})
    /// ```
    fn jacobian(&self) -> Result<Scalar, DeformationError>;
    /// Calculates and returns the left Cauchy-Green deformation.
    ///
    /// ```math
    /// \mathbf{B} = \mathbf{F}\cdot\mathbf{F}^T
    /// ```
    fn left_cauchy_green(&self) -> TensorRank2<3, I, I>;
    /// Calculates and returns the right Cauchy-Green deformation.
    ///
    /// ```math
    /// \mathbf{C} = \mathbf{F}^T\cdot\mathbf{F}
    /// ```
    fn right_cauchy_green(&self) -> TensorRank2<3, J, J>;
}

impl<const I: usize, const J: usize> Deformation<I, J> for DeformationGradientGeneral<I, J> {
    fn jacobian(&self) -> Result<Scalar, DeformationError> {
        let jacobian = self.determinant();
        if jacobian > 0.0 {
            Ok(jacobian)
        } else {
            Err(DeformationError::InvalidJacobian(jacobian))
        }
    }
    fn left_cauchy_green(&self) -> TensorRank2<3, I, I> {
        self.iter()
            .map(|deformation_gradient_i| {
                self.iter()
                    .map(|deformation_gradient_j| deformation_gradient_i * deformation_gradient_j)
                    .collect()
            })
            .collect()
    }
    fn right_cauchy_green(&self) -> TensorRank2<3, J, J> {
        let deformation_gradient_transpose = self.transpose();
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
}

/// A basis.
pub type Basis = TensorRank1List<3, 1, 3>;

/// A list of bases.
pub type Bases<const N: usize> = TensorRank1List2D<3, 1, 3, N>;

/// The Cauchy stress $`\boldsymbol{\sigma}`$.
pub type CauchyStress = TensorRank2<3, 1, 1>;

/// A list of Cauchy stresses.
pub type CauchyStresses<const W: usize> = TensorRank2List<3, 1, 1, W>;

/// The tangent stiffness associated with the Cauchy stress $`\boldsymbol{\mathcal{T}}`$.
pub type CauchyTangentStiffness = TensorRank4<3, 1, 1, 1, 0>;

/// The tangent stiffness associated with the Cauchy stress $`\boldsymbol{\mathcal{T}}_1`$.
pub type CauchyTangentStiffness1 = TensorRank4<3, 1, 1, 1, 2>;

/// The tangent stiffness associated with the elastic Cauchy stress $`\boldsymbol{\mathcal{T}}_\mathrm{e}`$.
pub type CauchyTangentStiffnessElastic = TensorRank4<3, 1, 1, 1, 2>;

/// The rate tangent stiffness associated with the Cauchy stress $`\boldsymbol{\mathcal{V}}`$.
pub type CauchyRateTangentStiffness = TensorRank4<3, 1, 1, 1, 0>;

/// A coordinate.
pub type Coordinate<const I: usize> = TensorRank1<3, I>;

/// A list of coordinates.
pub type CoordinateList<const I: usize, const N: usize> = TensorRank1List<3, I, N>;

/// A vector of coordinates.
pub type Coordinates<const I: usize> = TensorRank1Vec<3, I>;

/// A vector of references to coordinates.
pub type CoordinatesRef<'a, const I: usize> = TensorRank1RefVec<'a, 3, I>;

/// A coordinate in the current configuration.
pub type CurrentCoordinate = TensorRank1<3, 1>;

/// A list of coordinates in the current configuration.
pub type CurrentCoordinates<const W: usize> = TensorRank1List<3, 1, W>;

/// A vector of references to current coordinates.
pub type CurrentCoordinatesRef<'a> = TensorRank1RefVec<'a, 3, 1>;

/// A velocity in the current configuration.
pub type CurrentVelocity = TensorRank1<3, 1>;

/// The deformation gradient $`\mathbf{F}`$.
pub type DeformationGradient = TensorRank2<3, 1, 0>;

/// The second deformation gradient $`\mathbf{F}_2`$.
pub type DeformationGradient2 = TensorRank2<3, 2, 0>;

/// The elastic deformation gradient $`\mathbf{F}_\mathrm{e}`$.
pub type DeformationGradientElastic = TensorRank2<3, 1, 2>;

/// A general deformation gradient.
pub type DeformationGradientGeneral<const I: usize, const J: usize> = TensorRank2<3, I, J>;

/// The plastic deformation gradient $`\mathbf{F}_\mathrm{p}`$.
pub type DeformationGradientPlastic = TensorRank2<3, 2, 0>;

/// The deformation gradient rate $`\dot{\mathbf{F}}`$.
pub type DeformationGradientRate = TensorRank2<3, 1, 0>;

/// The plastic deformation gradient rate $`\dot{\mathbf{F}}_\mathrm{p}`$.
pub type DeformationGradientRatePlastic = TensorRank2<3, 2, 0>;

/// A list of deformation gradients.
pub type DeformationGradientList<const W: usize> = TensorRank2List<3, 1, 0, W>;

/// A list of deformation gradient rates.
pub type DeformationGradientRateList<const W: usize> = TensorRank2List<3, 1, 0, W>;

/// A vector of deformation gradients.
pub type DeformationGradients = TensorRank2Vec<3, 1, 0>;

/// A vector of plastic deformation gradients.
pub type DeformationGradientsPlastic = TensorRank2Vec<3, 2, 0>;

/// A vector of deformation gradient rates.
pub type DeformationGradientRates = TensorRank2Vec<3, 1, 0>;

/// A vector of plastic deformation gradient rates.
pub type DeformationGradientRatesPlastic = TensorRank2Vec<3, 2, 0>;

/// A displacement.
pub type Displacement = TensorRank1<3, 1>;

/// The first Piola-Kirchhoff stress $`\mathbf{P}`$.
pub type FirstPiolaKirchhoffStress = TensorRank2<3, 1, 0>;

/// The first Piola-Kirchhoff stress $`\mathbf{P}_1`$.
pub type FirstPiolaKirchhoffStress1 = TensorRank2<3, 1, 2>;

/// The first Piola-Kirchhoff stress $`\mathbf{P}_2`$.
pub type FirstPiolaKirchhoffStress2 = TensorRank2<3, 2, 0>;

/// The elastic first Piola-Kirchhoff stress $`\mathbf{P}_\mathrm{e}`$.
pub type FirstPiolaKirchhoffStressElastic = FirstPiolaKirchhoffStress1;

/// A list of first Piola-Kirchhoff stresses.
pub type FirstPiolaKirchhoffStressList<const N: usize> = TensorRank2List<3, 1, 0, N>;

/// A vector of first Piola-Kirchhoff stresses.
pub type FirstPiolaKirchhoffStresses = TensorRank2Vec<3, 1, 0>;

/// The tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{C}}`$.
pub type FirstPiolaKirchhoffTangentStiffness = TensorRank4<3, 1, 0, 1, 0>;

/// The first tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{C}}_1`$.
pub type FirstPiolaKirchhoffTangentStiffness1 = TensorRank4<3, 1, 2, 1, 2>;

/// The second tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{C}}_2`$.
pub type FirstPiolaKirchhoffTangentStiffness2 = TensorRank4<3, 2, 0, 2, 0>;

/// The elastic tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{C}}_\mathrm{e}`$.
pub type FirstPiolaKirchhoffTangentStiffnessElastic = FirstPiolaKirchhoffTangentStiffness1;

/// A list of first Piola-Kirchhoff tangent stiffnesses.
pub type FirstPiolaKirchhoffTangentStiffnessList<const N: usize> =
    TensorRank4List<3, 1, 0, 1, 0, N>;

/// A vector of first Piola-Kirchhoff tangent stiffnesses.
pub type FirstPiolaKirchhoffTangentStiffnesses = TensorRank4Vec<3, 1, 0, 1, 0>;

/// The rate tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{U}}`$.
pub type FirstPiolaKirchhoffRateTangentStiffness = TensorRank4<3, 1, 0, 1, 0>;

/// A list of first Piola-Kirchhoff rate tangent stiffnesses.
pub type FirstPiolaKirchhoffRateTangentStiffnesses<const W: usize> =
    TensorRank4List<3, 1, 0, 1, 0, W>;

/// A force.
pub type Force = TensorRank1<3, 1>;

/// A list of forces.
pub type ForceList<const N: usize> = TensorRank1List<3, 1, N>;

/// A vector of forces.
pub type Forces = TensorRank1Vec<3, 1>;

/// The frame spin $`\mathbf{\Omega}=\dot{\mathbf{Q}}\cdot\mathbf{Q}^T`$.
pub type FrameSpin = TensorRank2<3, 1, 1>;

/// The heat flux.
pub type HeatFlux = TensorRank1<3, 0>;

/// A list of heat fluxes.
pub type HeatFluxes<const N: usize> = TensorRank1List<3, 0, N>;

/// The heat flux tangent.
pub type HeatFluxTangent = TensorRank2<3, 0, 0>;

/// A list of heat flux tangents.
pub type HeatFluxTangents<const N: usize> = TensorRank2List<3, 0, 0, N>;

/// The left Cauchy-Green deformation $`\mathbf{B}`$.
pub type LeftCauchyGreenDeformation = TensorRank2<3, 1, 1>;

/// The Mandel stress $`\mathbf{M}`$.
pub type MandelStress = TensorRank2<3, 0, 0>;

/// The elastic stress $`\mathbf{M}_e`$.
pub type MandelStressElastic = TensorRank2<3, 2, 2>;

/// A normal.
pub type Normal = TensorRank1<3, 1>;

/// A list of normals.
pub type Normals<const N: usize> = TensorRank1List<3, 1, N>;

/// A list of normal gradients.
pub type NormalGradients<const O: usize, const P: usize> = TensorRank2List2D<3, 1, 1, O, P>;

/// A normal rate.
pub type NormalRate = TensorRank1<3, 1>;

/// A list of normal rates.
pub type NormalRates<const N: usize> = TensorRank1List<3, 1, N>;

/// A coordinate in the reference configuration.
pub type ReferenceCoordinate = TensorRank1<3, 0>;

/// A list of coordinates in the reference configuration.
pub type ReferenceCoordinates<const W: usize> = TensorRank1List<3, 0, W>;

/// A reference normal.
pub type ReferenceNormal = TensorRank1<3, 0>;

/// A list of reference normals.
pub type ReferenceNormals<const N: usize> = TensorRank1List<3, 0, N>;

/// The right Cauchy-Green deformation $`\mathbf{C}`$.
pub type RightCauchyGreenDeformation = TensorRank2<3, 0, 0>;

/// The rotation of the current configuration $`\mathbf{Q}`$.
pub type RotationCurrentConfiguration = TensorRank2<3, 1, 1>;

/// The rate of rotation of the current configuration $`\dot{\mathbf{Q}}`$.
pub type RotationRateCurrentConfiguration = TensorRank2<3, 1, 1>;

/// The rotation of the reference configuration $`\mathbf{Q}_0`$.
pub type RotationReferenceConfiguration = TensorRank2<3, 0, 0>;

/// A separation.
pub type Separation = Displacement;

/// The second Piola-Kirchhoff stress $`\mathbf{S}`$.
pub type SecondPiolaKirchhoffStress = TensorRank2<3, 0, 0>;

/// The elastic second Piola-Kirchhoff stress $`\mathbf{S}`$.
pub type SecondPiolaKirchhoffStressElastic = TensorRank2<3, 2, 2>;

/// The tangent stiffness associated with the second Piola-Kirchhoff stress $`\boldsymbol{\mathcal{G}}`$.
pub type SecondPiolaKirchhoffTangentStiffness = TensorRank4<3, 0, 0, 1, 0>;

/// The elastic tangent stiffness associated with the second Piola-Kirchhoff stress $`\boldsymbol{\mathcal{G}}_\mathrm{e}`$.
pub type SecondPiolaKirchhoffTangentStiffnessElastic = TensorRank4<3, 2, 2, 1, 2>;

/// The rate tangent stiffness associated with the second Piola-Kirchhoff stress $`\boldsymbol{\mathcal{W}}`$.
pub type SecondPiolaKirchhoffRateTangentStiffness = TensorRank4<3, 0, 0, 1, 0>;

/// A stiffness resulting from a force.
pub type Stiffness = TensorRank2<3, 1, 1>;

/// A list of stiffnesses.
pub type StiffnessList<const N: usize> = TensorRank2List2D<3, 1, 1, N, N>;

/// A vector of stiffnesses.
pub type Stiffnesses = TensorRank2Vec2D<3, 1, 1>;

/// The stretching rate $`\mathbf{D}`$.
pub type StretchingRate = TensorRank2<3, 1, 1>;

/// The plastic stretching rate $`\mathbf{D}^\mathrm{p}`$.
pub type StretchingRatePlastic = TensorRank2<3, 2, 2>;

/// A surface basis.
pub type SurfaceBasis<const I: usize> = TensorRank1List<3, I, 2>;

/// A list of surface bases.
pub type SurfaceBases<const I: usize, const N: usize> = TensorRank1List2D<3, I, 2, N>;

/// The temperature gradient.
pub type TemperatureGradient = TensorRank1<3, 0>;

/// A list of temperature gradients.
pub type TemperatureGradients<const N: usize> = TensorRank1List<3, 0, N>;

/// A vector of times.
pub type Times = crate::math::Vector;

/// A traction.
pub type Traction = TensorRank1<3, 1>;

/// A list of tractions.
pub type TractionList<const N: usize> = TensorRank1List<3, 1, N>;

/// A vector.
pub type Vector<const I: usize> = TensorRank1<3, I>;

/// A list of vectors.
pub type VectorList<const I: usize, const W: usize> = TensorRank1List<3, I, W>;

/// A 2D list of vectors.
pub type VectorList2D<const I: usize, const W: usize, const X: usize> =
    TensorRank1List2D<3, I, W, X>;

/// A vector of vectors.
pub type Vectors<const I: usize> = TensorRank1Vec<3, I>;

/// A 2D vector of vectors.
pub type Vectors2D<const I: usize> = TensorRank1Vec2D<3, I>;
