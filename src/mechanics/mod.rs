//! Mechanics library.

use crate::math::{
    Current, Intermediate, Reference,
    unit::{
        ForcePerLength, ForcePerVelocity, Length, PowerPerArea, PowerPerLengthTemperature, Rate,
        ReciprocalLength, Stress, TemperaturePerLength, Velocity, Viscosity,
    },
};
#[cfg(test)]
pub mod test;

use crate::math::{
    Rank2, Style, StyledError, Tensor, TensorRank1, TensorRank1List, TensorRank1List2D,
    TensorRank1RefVec, TensorRank1Vec, TensorRank1Vec2D, TensorRank2, TensorRank2List,
    TensorRank2List2D, TensorRank2Vec, TensorRank2Vec2D, TensorRank4, TensorRank4List,
    TensorRank4Vec, styled_error,
};

pub use crate::math::Scalar;

/// Possible errors for deformation gradients.
pub enum DeformationError {
    InvalidJacobian(Scalar),
}

impl StyledError for DeformationError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::InvalidJacobian(jacobian) => {
                format!("{h}Invalid Jacobian: {jacobian:.6e}.{c}")
            }
        }
    }
}

styled_error!(DeformationError);

/// Methods for deformation gradients.
pub trait Deformation<I, J> {
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

impl<I, J> Deformation<I, J> for DeformationGradientGeneral<I, J> {
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
pub type Basis = TensorRank1List<3, Current, 3, Length>;

/// A list of bases.
pub type Bases<const N: usize> = TensorRank1List2D<3, Current, 3, N, Length>;

/// The Cauchy stress $`\boldsymbol{\sigma}`$.
pub type CauchyStress = TensorRank2<3, Current, Current, Stress>;

/// A list of Cauchy stresses.
pub type CauchyStresses<const W: usize> = TensorRank2List<3, Current, Current, W, Stress>;

/// The tangent stiffness associated with the Cauchy stress $`\boldsymbol{\mathcal{T}}`$.
pub type CauchyTangentStiffness = TensorRank4<3, Current, Current, Current, Reference, Stress>;

/// The tangent stiffness associated with the Cauchy stress $`\boldsymbol{\mathcal{T}}_1`$.
pub type CauchyTangentStiffness1 = TensorRank4<3, Current, Current, Current, Intermediate, Stress>;

/// The tangent stiffness associated with the elastic Cauchy stress $`\boldsymbol{\mathcal{T}}_\mathrm{e}`$.
pub type CauchyTangentStiffnessElastic =
    TensorRank4<3, Current, Current, Current, Intermediate, Stress>;

/// The rate tangent stiffness associated with the Cauchy stress $`\boldsymbol{\mathcal{V}}`$.
pub type CauchyRateTangentStiffness =
    TensorRank4<3, Current, Current, Current, Reference, Viscosity>;

/// A coordinate.
pub type Coordinate<I> = TensorRank1<3, I, Length>;

/// A list of coordinates.
pub type CoordinateList<I, const N: usize> = TensorRank1List<3, I, N, Length>;

/// A vector of coordinates.
pub type Coordinates<I> = TensorRank1Vec<3, I, Length>;

/// A vector of references to coordinates.
pub type CoordinatesRef<'a, I> = TensorRank1RefVec<'a, 3, I, Length>;

/// A coordinate in the current configuration.
pub type CurrentCoordinate = TensorRank1<3, Current, Length>;

/// A list of coordinates in the current configuration.
pub type CurrentCoordinates<const W: usize> = TensorRank1List<3, Current, W, Length>;

/// A vector of references to current coordinates.
pub type CurrentCoordinatesRef<'a> = TensorRank1RefVec<'a, 3, Current, Length>;

/// A velocity in the current configuration.
pub type CurrentVelocity = TensorRank1<3, Current, Velocity>;

/// A list of velocities in the current configuration.
pub type CurrentVelocities<const W: usize> = TensorRank1List<3, Current, W, Velocity>;

/// The deformation gradient $`\mathbf{F}`$.
pub type DeformationGradient = TensorRank2<3, Current, Reference>;

/// The second deformation gradient $`\mathbf{F}_2`$.
pub type DeformationGradient2 = TensorRank2<3, Intermediate, Reference>;

/// The elastic deformation gradient $`\mathbf{F}_\mathrm{e}`$.
pub type DeformationGradientElastic = TensorRank2<3, Current, Intermediate>;

/// A general deformation gradient.
pub type DeformationGradientGeneral<I, J> = TensorRank2<3, I, J>;

/// The plastic deformation gradient $`\mathbf{F}_\mathrm{p}`$.
pub type DeformationGradientPlastic = TensorRank2<3, Intermediate, Reference>;

/// The deformation gradient rate $`\dot{\mathbf{F}}`$.
pub type DeformationGradientRate = TensorRank2<3, Current, Reference, Rate>;

/// The plastic deformation gradient rate $`\dot{\mathbf{F}}_\mathrm{p}`$.
pub type DeformationGradientRatePlastic = TensorRank2<3, Intermediate, Reference, Rate>;

/// A list of deformation gradients.
pub type DeformationGradientList<const W: usize> = TensorRank2List<3, Current, Reference, W>;

/// A list of deformation gradient rates.
pub type DeformationGradientRateList<const W: usize> =
    TensorRank2List<3, Current, Reference, W, Rate>;

/// A vector of deformation gradients.
pub type DeformationGradients = TensorRank2Vec<3, Current, Reference>;

/// A vector of plastic deformation gradients.
pub type DeformationGradientsPlastic = TensorRank2Vec<3, Intermediate, Reference>;

/// A vector of deformation gradient rates.
pub type DeformationGradientRates = TensorRank2Vec<3, Current, Reference, Rate>;

/// A vector of plastic deformation gradient rates.
pub type DeformationGradientRatesPlastic = TensorRank2Vec<3, Intermediate, Reference, Rate>;

/// A displacement.
pub type Displacement = TensorRank1<3, Current, Length>;

/// The first Piola-Kirchhoff stress $`\mathbf{P}`$.
pub type FirstPiolaKirchhoffStress = TensorRank2<3, Current, Reference, Stress>;

/// The first Piola-Kirchhoff stress $`\mathbf{P}_1`$.
pub type FirstPiolaKirchhoffStress1 = TensorRank2<3, Current, Intermediate, Stress>;

/// The first Piola-Kirchhoff stress $`\mathbf{P}_2`$.
pub type FirstPiolaKirchhoffStress2 = TensorRank2<3, Intermediate, Reference, Stress>;

/// The elastic first Piola-Kirchhoff stress $`\mathbf{P}_\mathrm{e}`$.
pub type FirstPiolaKirchhoffStressElastic = FirstPiolaKirchhoffStress1;

/// A list of first Piola-Kirchhoff stresses.
pub type FirstPiolaKirchhoffStressList<const N: usize> =
    TensorRank2List<3, Current, Reference, N, Stress>;

/// A vector of first Piola-Kirchhoff stresses.
pub type FirstPiolaKirchhoffStresses = TensorRank2Vec<3, Current, Reference, Stress>;

/// The tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{C}}`$.
pub type FirstPiolaKirchhoffTangentStiffness =
    TensorRank4<3, Current, Reference, Current, Reference, Stress>;

/// The first tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{C}}_1`$.
pub type FirstPiolaKirchhoffTangentStiffness1 =
    TensorRank4<3, Current, Intermediate, Current, Intermediate, Stress>;

/// The second tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{C}}_2`$.
pub type FirstPiolaKirchhoffTangentStiffness2 =
    TensorRank4<3, Intermediate, Reference, Intermediate, Reference, Stress>;

/// The elastic tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{C}}_\mathrm{e}`$.
pub type FirstPiolaKirchhoffTangentStiffnessElastic = FirstPiolaKirchhoffTangentStiffness1;

/// A list of first Piola-Kirchhoff tangent stiffnesses.
pub type FirstPiolaKirchhoffTangentStiffnessList<const N: usize> =
    TensorRank4List<3, Current, Reference, Current, Reference, N, Stress>;

/// A vector of first Piola-Kirchhoff tangent stiffnesses.
pub type FirstPiolaKirchhoffTangentStiffnesses =
    TensorRank4Vec<3, Current, Reference, Current, Reference, Stress>;

/// The rate tangent stiffness associated with the first Piola-Kirchhoff stress $`\boldsymbol{\mathcal{U}}`$.
pub type FirstPiolaKirchhoffRateTangentStiffness =
    TensorRank4<3, Current, Reference, Current, Reference, Viscosity>;

/// A list of first Piola-Kirchhoff rate tangent stiffnesses.
pub type FirstPiolaKirchhoffRateTangentStiffnesses<const W: usize> =
    TensorRank4List<3, Current, Reference, Current, Reference, W, Viscosity>;

/// A force.
pub type Force = TensorRank1<3, Current, crate::math::unit::Force>;

/// A list of forces.
pub type ForceList<const N: usize> = TensorRank1List<3, Current, N, crate::math::unit::Force>;

/// A vector of forces.
pub type Forces = TensorRank1Vec<3, Current, crate::math::unit::Force>;

/// The frame spin $`\mathbf{\Omega}=\dot{\mathbf{Q}}\cdot\mathbf{Q}^T`$.
pub type FrameSpin = TensorRank2<3, Current, Current, Rate>;

/// The heat flux.
pub type HeatFlux = TensorRank1<3, Reference, PowerPerArea>;

/// A list of heat fluxes.
pub type HeatFluxes<const N: usize> = TensorRank1List<3, Reference, N, PowerPerArea>;

/// The heat flux tangent.
pub type HeatFluxTangent = TensorRank2<3, Reference, Reference, PowerPerLengthTemperature>;

/// A list of heat flux tangents.
pub type HeatFluxTangents<const N: usize> =
    TensorRank2List<3, Reference, Reference, N, PowerPerLengthTemperature>;

/// The left Cauchy-Green deformation $`\mathbf{B}`$.
pub type LeftCauchyGreenDeformation = TensorRank2<3, Current, Current>;

/// The Mandel stress $`\mathbf{M}`$.
pub type MandelStress = TensorRank2<3, Reference, Reference, Stress>;

/// The elastic stress $`\mathbf{M}_e`$.
pub type MandelStressElastic = TensorRank2<3, Intermediate, Intermediate, Stress>;

/// A normal.
pub type Normal = TensorRank1<3, Current>;

/// A list of normals.
pub type Normals<const N: usize> = TensorRank1List<3, Current, N>;

/// A list of normal gradients.
pub type NormalGradients<const O: usize, const P: usize> =
    TensorRank2List2D<3, Current, Current, O, P, ReciprocalLength>;

/// A normal rate.
pub type NormalRate = TensorRank1<3, Current, Rate>;

/// A list of normal rates.
pub type NormalRates<const N: usize> = TensorRank1List<3, Current, N, Rate>;

/// A coordinate in the reference configuration.
pub type ReferenceCoordinate = TensorRank1<3, Reference, Length>;

/// A list of coordinates in the reference configuration.
pub type ReferenceCoordinates<const W: usize> = TensorRank1List<3, Reference, W, Length>;

/// A reference normal.
pub type ReferenceNormal = TensorRank1<3, Reference>;

/// A list of reference normals.
pub type ReferenceNormals<const N: usize> = TensorRank1List<3, Reference, N>;

/// The right Cauchy-Green deformation $`\mathbf{C}`$.
pub type RightCauchyGreenDeformation = TensorRank2<3, Reference, Reference>;

/// The rotation of the current configuration $`\mathbf{Q}`$.
pub type RotationCurrentConfiguration = TensorRank2<3, Current, Current>;

/// A list of rotations of the current configuration.
pub type RotationCurrentConfigurationList<const N: usize> = TensorRank2List<3, Current, Current, N>;

/// The rate of rotation of the current configuration $`\dot{\mathbf{Q}}`$.
pub type RotationRateCurrentConfiguration = TensorRank2<3, Current, Current, Rate>;

/// The rotation of the reference configuration $`\mathbf{Q}_0`$.
pub type RotationReferenceConfiguration = TensorRank2<3, Reference, Reference>;

/// A separation.
pub type Separation = Displacement;

/// The second Piola-Kirchhoff stress $`\mathbf{S}`$.
pub type SecondPiolaKirchhoffStress = TensorRank2<3, Reference, Reference, Stress>;

/// The elastic second Piola-Kirchhoff stress $`\mathbf{S}`$.
pub type SecondPiolaKirchhoffStressElastic = TensorRank2<3, Intermediate, Intermediate, Stress>;

/// The tangent stiffness associated with the second Piola-Kirchhoff stress $`\boldsymbol{\mathcal{G}}`$.
pub type SecondPiolaKirchhoffTangentStiffness =
    TensorRank4<3, Reference, Reference, Current, Reference, Stress>;

/// The elastic tangent stiffness associated with the second Piola-Kirchhoff stress $`\boldsymbol{\mathcal{G}}_\mathrm{e}`$.
pub type SecondPiolaKirchhoffTangentStiffnessElastic =
    TensorRank4<3, Intermediate, Intermediate, Current, Intermediate, Stress>;

/// The rate tangent stiffness associated with the second Piola-Kirchhoff stress $`\boldsymbol{\mathcal{W}}`$.
pub type SecondPiolaKirchhoffRateTangentStiffness =
    TensorRank4<3, Reference, Reference, Current, Reference, Viscosity>;

/// A stiffness resulting from a force.
pub type Stiffness = TensorRank2<3, Current, Current, ForcePerLength>;

/// A list of stiffnesses.
pub type StiffnessList<const N: usize> = TensorRank2List<3, Current, Current, N, ForcePerLength>;

/// A 2D list of stiffnesses.
pub type StiffnessList2D<const N: usize> =
    TensorRank2List2D<3, Current, Current, N, N, ForcePerLength>;

/// A damping resulting from a force per unit rate.
pub type Damping = TensorRank2<3, Current, Current, ForcePerVelocity>;

/// A list of two-dimensional lists of dampings.
pub type DampingList2D<const N: usize> =
    TensorRank2List2D<3, Current, Current, N, N, ForcePerVelocity>;

/// A vector of two-dimensional vectors of dampings.
pub type Dampings = TensorRank2Vec2D<3, Current, Current, ForcePerVelocity>;

/// A vector of stiffnesses.
pub type Stiffnesses = TensorRank2Vec2D<3, Current, Current, ForcePerLength>;

/// The stretching rate $`\mathbf{D}`$.
pub type StretchingRate = TensorRank2<3, Current, Current, Rate>;

/// The plastic stretching rate $`\mathbf{D}^\mathrm{p}`$.
pub type StretchingRatePlastic = TensorRank2<3, Intermediate, Intermediate, Rate>;

/// A surface basis.
pub type SurfaceBasis<I> = TensorRank1List<3, I, 2, Length>;

/// A list of surface bases.
pub type SurfaceBases<I, const N: usize> = TensorRank1List2D<3, I, 2, N, Length>;

/// A surface dual basis.
pub type SurfaceDualBasis<I> = TensorRank1List<3, I, 2, ReciprocalLength>;

/// A list of surface dual bases.
pub type SurfaceDualBases<I, const N: usize> = TensorRank1List2D<3, I, 2, N, ReciprocalLength>;

/// The temperature gradient.
pub type TemperatureGradient = TensorRank1<3, Reference, TemperaturePerLength>;

/// A list of temperature gradients.
pub type TemperatureGradients<const N: usize> =
    TensorRank1List<3, Reference, N, TemperaturePerLength>;

/// A vector of times.
pub type Times = crate::math::integrate::Times;

/// A traction.
pub type Traction = TensorRank1<3, Current, Stress>;

/// A list of tractions.
pub type TractionList<const N: usize> = TensorRank1List<3, Current, N, Stress>;

/// A vector.
pub type Vector<I> = TensorRank1<3, I>;

/// A list of vectors.
pub type VectorList<I, const W: usize> = TensorRank1List<3, I, W>;

/// A 2D list of vectors.
pub type VectorList2D<I, const W: usize, const X: usize> = TensorRank1List2D<3, I, W, X>;

/// A vector of vectors.
pub type Vectors<I> = TensorRank1Vec<3, I>;

/// A 2D vector of vectors.
pub type Vectors2D<I> = TensorRank1Vec2D<3, I>;
