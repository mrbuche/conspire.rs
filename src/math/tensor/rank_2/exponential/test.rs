use crate::math::Current;
use crate::math::assert::{Assert, AssertionError};
use crate::math::{Rank2, TensorArray, TensorRank2};
use crate::units::Dimensionless;

fn rotation() -> TensorRank2<3, Current, Current> {
    [
        [
            0.781_639_173_907_025,
            -0.482_929_284_214_212_2,
            0.394_739_798_173_799_8,
        ],
        [
            0.550_117_230_704_358_4,
            0.832_030_133_774_634_6,
            -0.071_392_499_417_875_86,
        ],
        [
            -0.29395787843858057,
            0.27295633888831433,
            0.916_015_066_887_317_3,
        ],
    ]
    .into()
}

fn from_eigenvalues(eigenvalues: [f64; 3]) -> TensorRank2<3, Current, Current> {
    let rotation = rotation();
    let diagonal = TensorRank2::<3, Current, Current, Dimensionless>::from([
        [eigenvalues[0], 0.0, 0.0],
        [0.0, eigenvalues[1], 0.0],
        [0.0, 0.0, eigenvalues[2]],
    ]);
    let tensor = &(&rotation * &diagonal) * &rotation.transpose();
    (tensor.clone() + tensor.transpose()) * 0.5
}

const TIGHT: Assert = Assert {
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    fd_tol: 3e-6,
};

#[test]
fn expm_identity_is_e() -> Result<(), AssertionError> {
    TIGHT.eq_within_tols(
        &TensorRank2::<3, Current, Current>::identity().expm()?,
        &(TensorRank2::<3, Current, Current>::identity() * std::f64::consts::E),
    )
}

#[test]
fn expm_zero_is_identity() -> Result<(), AssertionError> {
    TIGHT.eq_within_tols(
        &TensorRank2::<3, Current, Current>::zero().expm()?,
        &TensorRank2::identity(),
    )
}

#[test]
fn expm_diagonal() -> Result<(), AssertionError> {
    let tensor = TensorRank2::<3, Current, Current>::from([
        [0.7, 0.0, 0.0],
        [0.0, -0.4, 0.0],
        [0.0, 0.0, 0.2],
    ]);
    TIGHT.eq_within_tols(
        &tensor.expm()?,
        &TensorRank2::from([
            [0.7_f64.exp(), 0.0, 0.0],
            [0.0, (-0.4_f64).exp(), 0.0],
            [0.0, 0.0, 0.2_f64.exp()],
        ]),
    )
}

#[test]
fn expm_symmetric_matches_eigenvalue_exponentials() -> Result<(), AssertionError> {
    TIGHT.eq_within_tols(
        &from_eigenvalues([0.9, -0.3, 0.15]).expm()?,
        &from_eigenvalues([0.9_f64.exp(), (-0.3_f64).exp(), 0.15_f64.exp()]),
    )
}

#[test]
fn expm_series_branch_matches_eigenvalue_branch() -> Result<(), AssertionError> {
    // norm below 1e-2 takes the truncated series; compare against the spectral result.
    TIGHT.eq_within_tols(
        &from_eigenvalues([3.0e-3, -2.0e-3, 1.0e-3]).expm()?,
        &from_eigenvalues([(3.0e-3_f64).exp(), (-2.0e-3_f64).exp(), (1.0e-3_f64).exp()]),
    )
}

#[test]
fn expm_is_inverse_of_logm() -> Result<(), AssertionError> {
    let tensor = from_eigenvalues([1.7, 0.6, 1.1]);
    TIGHT.eq_within_tols(&tensor.logm()?.expm()?, &tensor)
}

#[test]
fn expm_repeated_eigenvalue() -> Result<(), AssertionError> {
    TIGHT.eq_within_tols(
        &from_eigenvalues([0.4, 0.4, -0.2]).expm()?,
        &from_eigenvalues([0.4_f64.exp(), 0.4_f64.exp(), (-0.2_f64).exp()]),
    )
}

#[test]
fn expm_deviatoric_has_unit_determinant() -> Result<(), AssertionError> {
    // exp of a trace-free tensor is unimodular.
    let deviatoric = from_eigenvalues([0.5, -0.3, -0.2]);
    Assert::default().eq_within_tols(deviatoric.expm()?.determinant(), &1.0)
}

#[test]
#[should_panic(expected = "Matrix exponential only implemented for symmetric cases")]
fn expm_non_symmetric_panics() {
    let _ = TensorRank2::<3, Current, Current>::from([
        [1.0, 4.0, 6.0],
        [7.0, 2.0, 5.0],
        [9.0, 8.0, 3.0],
    ])
    .expm();
}
