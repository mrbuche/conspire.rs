use crate::math::Current;
use crate::math::assert::{Assert, AssertionError};
use crate::math::{Quantity, Rank2, TensorArray, TensorRank2, TensorRank4};
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

fn contract_third_fourth_indices(
    rank_4: &TensorRank4<3, Current, Current, Current, Current>,
    tensor: &TensorRank2<3, Current, Current>,
) -> TensorRank2<3, Current, Current> {
    let mut result = TensorRank2::zero();
    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            result[i][j] = (0..3)
                .map(|k| {
                    (0..3)
                        .map(|l| rank_4[i][j][k][l] * tensor[k][l])
                        .sum::<Quantity>()
                })
                .sum();
        })
    });
    result
}

fn dexpm_matches_finite_difference(
    tensor: &TensorRank2<3, Current, Current>,
    tolerance: f64,
) -> Result<(), AssertionError> {
    let dexpm = tensor.dexpm()?;
    let epsilon = 1e-6;
    let directions = [
        TensorRank2::from([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        TensorRank2::from([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        TensorRank2::from([[0.3, 0.2, 0.1], [0.2, -0.4, 0.05], [0.1, 0.05, 0.1]]),
    ];
    for direction in directions.iter() {
        let perturbation = direction * epsilon;
        let finite_difference = ((tensor.clone() + perturbation.clone()).expm()?
            - (tensor.clone() - perturbation).expm()?)
            / (2.0 * epsilon);
        Assert {
            abs_tol: tolerance,
            rel_tol: tolerance,
            ..Default::default()
        }
        .eq_within_tols(
            &finite_difference,
            &contract_third_fourth_indices(&dexpm, direction),
        )?
    }
    Ok(())
}

#[test]
fn dexpm_diagonal_matches_finite_difference_of_expm() -> Result<(), AssertionError> {
    dexpm_matches_finite_difference(
        &TensorRank2::from([[0.7, 0.0, 0.0], [0.0, -0.4, 0.0], [0.0, 0.0, 0.2]]),
        1e-6,
    )
}

#[test]
fn dexpm_symmetric_matches_finite_difference_of_expm() -> Result<(), AssertionError> {
    dexpm_matches_finite_difference(&from_eigenvalues([0.9, -0.3, 0.15]), 1e-6)
}

#[test]
fn dexpm_repeated_eigenvalue_matches_finite_difference_of_expm() -> Result<(), AssertionError> {
    // the finite difference itself degrades where the eigenvectors are not unique.
    dexpm_matches_finite_difference(&from_eigenvalues([0.4, 0.4, -0.2]), 1e-3)?;
    dexpm_matches_finite_difference(&from_eigenvalues([0.4, -0.2, -0.2]), 1e-3)
}

#[test]
fn dexpm_series_branch_matches_finite_difference_of_expm() -> Result<(), AssertionError> {
    dexpm_matches_finite_difference(&from_eigenvalues([3.0e-3, -2.0e-3, 1.0e-3]), 1e-6)
}

#[test]
fn dexpm_non_symmetric_matches_finite_difference_of_expm() -> Result<(), AssertionError> {
    // norm ~0.35, materially non-symmetric: the scaling-and-squaring branch
    dexpm_matches_finite_difference(
        &TensorRank2::from([[0.1, 0.25, -0.05], [-0.2, 0.05, 0.15], [0.1, -0.1, -0.15]]),
        1e-6,
    )
}

#[test]
fn dexpm_zero_is_the_fourth_order_identity() -> Result<(), AssertionError> {
    let dexpm = TensorRank2::<3, Current, Current>::zero().dexpm()?;
    let mut identity = TensorRank4::<3, Current, Current, Current, Current>::zero();
    (0..3).for_each(|i| (0..3).for_each(|j| identity[i][j][i][j] = Quantity::new(1.0)));
    TIGHT.eq_within_tols(&dexpm, &identity)
}

#[test]
fn dexpm_inverts_dlogm() -> Result<(), AssertionError> {
    // d(exp)|_{log B} composed with d(log)|_B is the fourth-order identity.
    let tensor = from_eigenvalues([1.7, 0.6, 1.1]);
    let dlogm = tensor.dlogm()?;
    let dexpm = tensor.logm()?.dexpm()?;
    let mut composition = TensorRank4::<3, Current, Current, Current, Current>::zero();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    composition[i][j][k][l] = (0..3)
                        .map(|m| {
                            (0..3)
                                .map(|n| dexpm[i][j][m][n] * dlogm[m][n][k][l])
                                .sum::<Quantity>()
                        })
                        .sum()
                }
            }
        }
    }
    let mut identity = TensorRank4::<3, Current, Current, Current, Current>::zero();
    (0..3).for_each(|i| (0..3).for_each(|j| identity[i][j][i][j] = Quantity::new(1.0)));
    Assert {
        abs_tol: 1e-8,
        rel_tol: 1e-8,
        ..Default::default()
    }
    .eq_within_tols(&composition, &identity)
}

#[test]
fn expm_non_symmetric_matches_a_high_order_taylor_reference() -> Result<(), AssertionError> {
    let a = TensorRank2::<3, Current, Current>::from([
        [1.2, 3.3, -2.1],
        [-2.7, 0.6, 1.8],
        [1.5, -0.9, -1.8],
    ]);
    let mut reference = TensorRank2::identity() + &a;
    let mut power = a.clone();
    let mut factorial = 1.0;
    (2..40).for_each(|k| {
        power = &power * &a;
        factorial *= k as f64;
        reference += &power / factorial;
    });
    Assert {
        abs_tol: 1e-10,
        rel_tol: 1e-10,
        ..Default::default()
    }
    .eq_within_tols(a.expm().unwrap(), &reference)
}

fn rate() -> TensorRank2<3, Current, Current> {
    [[1.0, -0.5, 0.3], [0.2, 0.7, -0.1], [-0.4, 0.6, 0.9]].into()
}

#[test]
fn dexpinv_at_zero_is_the_identity() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        TensorRank2::<3, Current, Current>::zero().dexpinv(&rate()),
        &rate(),
    )
}

#[test]
fn dexpinv_is_the_right_trivialized_inverse() -> Result<(), AssertionError> {
    // The Fréchet derivative and the right-trivialized differential are related by
    // `dexpm(σ)[H] = dexp_σ(H)·exp(σ)`, and `dexpinv` inverts `dexp_σ`, so
    // `dexpm(σ) : dexpinv_σ(A) = A·exp(σ)`.
    //
    // small norm so the truncated dexpinv is exact enough for the identity
    let sigma = TensorRank2::<3, Current, Current>::from([
        [0.0, 0.5, -0.25],
        [-0.4, 0.0, 0.3],
        [0.15, -0.1, 0.0],
    ]) * 0.008;
    let recovered = contract_third_fourth_indices(&sigma.dexpm()?, &sigma.dexpinv(&rate()));
    Assert {
        abs_tol: 1e-8,
        rel_tol: 1e-8,
        ..Default::default()
    }
    .eq_within_tols(&recovered, &(&rate() * &sigma.expm()?))
}

#[test]
fn dexpinv_correction_is_non_trivial() {
    use crate::math::Tensor;
    let sigma = TensorRank2::<3, Current, Current>::from([
        [0.0, 0.4, -0.2],
        [-0.3, 0.0, 0.5],
        [0.1, -0.15, 0.0],
    ]);
    assert!((sigma.dexpinv(&rate()) - &rate()).norm().value() > 1e-2)
}

fn sigma_direction() -> TensorRank2<3, Current, Current> {
    [[0.1, 0.6, -0.3], [-0.2, 0.4, 0.5], [0.7, -0.15, -0.5]].into()
}

fn rate_direction() -> TensorRank2<3, Current, Current> {
    [[-0.3, 0.2, 0.8], [0.5, -0.6, 0.1], [-0.4, 0.9, 0.9]].into()
}

#[test]
fn dexpinv_tangent_matches_finite_difference() -> Result<(), AssertionError> {
    let sigma = TensorRank2::<3, Current, Current>::from([
        [0.0, 0.4, -0.2],
        [-0.3, 0.0, 0.5],
        [0.1, -0.15, 0.0],
    ]);
    let (d_sigma, d_rate) = (sigma_direction(), rate_direction());
    let step = crate::EPSILON * 0.5;
    let plus = (&sigma + (&d_sigma * step)).dexpinv(&(&rate() + (&d_rate * step)));
    let minus = (&sigma - (&d_sigma * step)).dexpinv(&(&rate() - (&d_rate * step)));
    Assert::default().eq_within_fd_tol(
        sigma.dexpinv_tangent(&rate(), &d_sigma, &d_rate),
        &((plus - minus) / crate::EPSILON),
    )
}
