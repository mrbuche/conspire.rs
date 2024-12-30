//! Mechanics library.

#[cfg(test)]
pub mod test;

use crate::math::{
    tensor_rank_1_zero, TensorRank0, TensorRank0List, TensorRank1, TensorRank1List,
    TensorRank1List2D, TensorRank2, TensorRank2List, TensorRank2List2D, TensorRank3, TensorRank4,
    TensorRank4List,
};

pub const IDENTITY: TensorRank2<3, 1, 1> = TensorRank2([
    TensorRank1([1.0, 0.0, 0.0]),
    TensorRank1([0.0, 1.0, 0.0]),
    TensorRank1([0.0, 0.0, 1.0]),
]);

pub const IDENTITY_00: TensorRank2<3, 0, 0> = TensorRank2([
    TensorRank1([1.0, 0.0, 0.0]),
    TensorRank1([0.0, 1.0, 0.0]),
    TensorRank1([0.0, 0.0, 1.0]),
]);

pub const IDENTITY_10: TensorRank2<3, 1, 0> = TensorRank2([
    TensorRank1([1.0, 0.0, 0.0]),
    TensorRank1([0.0, 1.0, 0.0]),
    TensorRank1([0.0, 0.0, 1.0]),
]);

pub const IDENTITY_1010: TensorRank4<3, 1, 0, 1, 0> = TensorRank4([
    TensorRank3([
        TensorRank2([
            TensorRank1([1.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            TensorRank1([0.0, 1.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            TensorRank1([0.0, 0.0, 1.0]),
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
        ]),
    ]),
    TensorRank3([
        TensorRank2([
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([1.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 1.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 1.0]),
            TensorRank1([0.0, 0.0, 0.0]),
        ]),
    ]),
    TensorRank3([
        TensorRank2([
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([1.0, 0.0, 0.0]),
        ]),
        TensorRank2([
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 1.0, 0.0]),
        ]),
        TensorRank2([
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 0.0]),
            TensorRank1([0.0, 0.0, 1.0]),
        ]),
    ]),
]);

pub const LEVI_CIVITA: TensorRank3<3, 1, 1, 1> = TensorRank3([
    TensorRank2([
        TensorRank1([0.0, 0.0, 0.0]),
        TensorRank1([0.0, 0.0, 1.0]),
        TensorRank1([0.0, -1.0, 0.0]),
    ]),
    TensorRank2([
        TensorRank1([0.0, 0.0, -1.0]),
        TensorRank1([0.0, 0.0, 0.0]),
        TensorRank1([1.0, 0.0, 0.0]),
    ]),
    TensorRank2([
        TensorRank1([0.0, 1.0, 0.0]),
        TensorRank1([-1.0, 0.0, 0.0]),
        TensorRank1([0.0, 0.0, 0.0]),
    ]),
]);

pub const ZERO: TensorRank2<3, 1, 1> = TensorRank2([
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
]);

pub const ZERO_10: TensorRank2<3, 1, 0> = TensorRank2([
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
    tensor_rank_1_zero(),
]);

pub const ZERO_VECTOR: TensorRank1<3, 1> = tensor_rank_1_zero();

/// The Cauchy stress $`\boldsymbol{\sigma}`$.
pub type CauchyStress = TensorRank2<3, 1, 1>;

/// A list of Cauchy stresses.
pub type CauchyStresses<const W: usize> = TensorRank2List<3, 1, 1, W>;

/// The tangent stiffness associated with the Cauchy stress $`\boldsymbol{\mathcal{T}}`$.
pub type CauchyTangentStiffness = TensorRank4<3, 1, 1, 1, 0>;

/// The rate tangent stiffness associated with the Cauchy stress $`\boldsymbol{\mathcal{V}}`$.
pub type CauchyRateTangentStiffness = TensorRank4<3, 1, 1, 1, 0>;

/// A list of coordinates.
pub type Coordinates<const I: usize, const W: usize> = TensorRank1List<3, I, W>;

/// A coordinate in the current configuration.
pub type CurrentCoordinate = TensorRank1<3, 1>;

/// A list of coordinates in the current configuration.
pub type CurrentCoordinates<const W: usize> = TensorRank1List<3, 1, W>;

/// A velocity in the current configuration.
pub type CurrentVelocity = TensorRank1<3, 1>;

/// The deformation gradient $`\mathbf{F}`$.
pub type DeformationGradient = TensorRank2<3, 1, 0>;

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
pub type DeformationGradients<const W: usize> = TensorRank2List<3, 1, 0, W>;

/// A 2D list of deformation gradients.
pub type DeformationGradientss<const W: usize, const X: usize> = TensorRank2List2D<3, 1, 0, W, X>;

/// A list of deformation gradient rates.
pub type DeformationGradientRates<const W: usize> = TensorRank2List<3, 1, 0, W>;

/// A displacement.
pub type Displacement = TensorRank1<3, 1>;

/// The first Piola-Kirchoff stress $`\mathbf{P}`$.
pub type FirstPiolaKirchoffStress = TensorRank2<3, 1, 0>;

/// A list of first Piola-Kirchoff stresses.
pub type FirstPiolaKirchoffStresses<const W: usize> = TensorRank2List<3, 1, 0, W>;

/// The tangent stiffness associated with the first Piola-Kirchoff stress $`\boldsymbol{\mathcal{C}}`$.
pub type FirstPiolaKirchoffTangentStiffness = TensorRank4<3, 1, 0, 1, 0>;

/// A list of first Piola-Kirchoff tangent stiffnesses.
pub type FirstPiolaKirchoffTangentStiffnesses<const W: usize> = TensorRank4List<3, 1, 0, 1, 0, W>;

/// The rate tangent stiffness associated with the first Piola-Kirchoff stress $`\boldsymbol{\mathcal{U}}`$.
pub type FirstPiolaKirchoffRateTangentStiffness = TensorRank4<3, 1, 0, 1, 0>;

/// A list of first Piola-Kirchoff rate tangent stiffnesses.
pub type FirstPiolaKirchoffRateTangentStiffnesses<const W: usize> =
    TensorRank4List<3, 1, 0, 1, 0, W>;

/// A force.
pub type Force = TensorRank1<3, 1>;

/// A list of forces.
pub type Forces<const W: usize> = TensorRank1List<3, 1, W>;

/// The frame spin $`\mathbf{\Omega}=\dot{\mathbf{Q}}\cdot\mathbf{Q}^T`$.
pub type FrameSpin = TensorRank2<3, 1, 1>;

/// The heat flux.
pub type HeatFlux = TensorRank1<3, 1>;

/// The left Cauchy-Green deformation $`\mathbf{B}`$.
pub type LeftCauchyGreenDeformation = TensorRank2<3, 1, 1>;

/// The Mandel stress $`\mathbf{M}`$.
pub type MandelStress = TensorRank2<3, 2, 2>;

/// A normal.
pub type Normal = TensorRank1<3, 1>;

/// A coordinate in the reference configuration.
pub type ReferenceCoordinate = TensorRank1<3, 0>;

/// A list of coordinates in the reference configuration.
pub type ReferenceCoordinates<const W: usize> = TensorRank1List<3, 0, W>;

/// The right Cauchy-Green deformation $`\mathbf{C}`$.
pub type RightCauchyGreenDeformation = TensorRank2<3, 0, 0>;

/// The rotation of the current configuration $`\mathbf{Q}`$.
pub type RotationCurrentConfiguration = TensorRank2<3, 1, 1>;

/// The rate of rotation of the current configuration $`\dot{\mathbf{Q}}`$.
pub type RotationRateCurrentConfiguration = TensorRank2<3, 1, 1>;

/// The rotation of the reference configuration $`\mathbf{Q}_0`$.
pub type RotationReferenceConfiguration = TensorRank2<3, 0, 0>;

/// A scalar.
pub type Scalar = TensorRank0;

/// A list of scalars.
pub type Scalars<const W: usize> = TensorRank0List<W>;

/// The second Piola-Kirchoff stress $`\mathbf{S}`$.
pub type SecondPiolaKirchoffStress = TensorRank2<3, 0, 0>;

/// The tangent stiffness associated with the second Piola-Kirchoff stress $`\boldsymbol{\mathcal{G}}`$.
pub type SecondPiolaKirchoffTangentStiffness = TensorRank4<3, 0, 0, 1, 0>;

/// The rate tangent stiffness associated with the second Piola-Kirchoff stress $`\boldsymbol{\mathcal{W}}`$.
pub type SecondPiolaKirchoffRateTangentStiffness = TensorRank4<3, 0, 0, 1, 0>;

/// A stiffness resulting from a force.
pub type Stiffness = TensorRank2<3, 1, 1>;

/// A list of stiffnesses.
pub type Stiffnesses<const W: usize> = TensorRank2List2D<3, 1, 1, W, W>;

/// The stretching rate $`\mathbf{D}`$.
pub type StretchingRate = TensorRank2<3, 1, 1>;

/// The plastic stretching rate $`\mathbf{D}^\mathrm{p}`$.
pub type StretchingRatePlastic = TensorRank2<3, 2, 2>;

/// The temperature gradient.
pub type TemperatureGradient = TensorRank1<3, 1>;

/// A traction.
pub type Traction = TensorRank1<3, 1>;

/// A vector.
pub type Vector<const I: usize> = TensorRank1<3, I>;

/// A list of vectors.
pub type Vectors<const I: usize, const W: usize> = TensorRank1List<3, I, W>;

/// A 2D list of vectors.
pub type Vectors2D<const I: usize, const W: usize, const X: usize> = TensorRank1List2D<3, I, W, X>;
