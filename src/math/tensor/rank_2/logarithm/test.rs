use crate::math::{
    Rank2, TensorArray, TensorRank2, TensorRank4,
    tensor::test::assert_eq_within,
    test::{TestError, assert_eq_within_tols},
};

fn get_rotation() -> TensorRank2<3, 1, 1> {
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

fn from_eigenvalues(eigenvalues: [f64; 3]) -> TensorRank2<3, 1, 1> {
    let rotation = get_rotation();
    let diagonal = TensorRank2::from([
        [eigenvalues[0], 0.0, 0.0],
        [0.0, eigenvalues[1], 0.0],
        [0.0, 0.0, eigenvalues[2]],
    ]);
    let tensor = &(&rotation * &diagonal) * &rotation.transpose();
    (tensor.clone() + tensor.transpose()) * 0.5
}

fn get_symmetric_tensor() -> TensorRank2<3, 1, 1> {
    from_eigenvalues([2.0, 0.5, 1.5])
}

fn get_symmetric_tensor_logm() -> TensorRank2<3, 1, 1> {
    from_eigenvalues([2.0_f64.ln(), 0.5_f64.ln(), 1.5_f64.ln()])
}

fn contract_third_fourth_indices(
    dlogm: &TensorRank4<3, 1, 1, 1, 1>,
    tensor: &TensorRank2<3, 1, 1>,
) -> TensorRank2<3, 1, 1> {
    let mut result = TensorRank2::zero();
    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            result[i][j] = (0..3)
                .map(|k| {
                    (0..3)
                        .map(|l| dlogm[i][j][k][l] * tensor[k][l])
                        .sum::<f64>()
                })
                .sum();
        })
    });
    result
}

#[test]
fn logm_identity() -> Result<(), TestError> {
    assert_eq_within_tols(
        &TensorRank2::<3, 1, 1>::identity().logm()?,
        &TensorRank2::zero(),
    )
}

#[test]
fn logm_diagonal() -> Result<(), TestError> {
    let tensor = TensorRank2::<3, 1, 1>::from([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.5]]);
    let expected = TensorRank2::from([
        [2.0_f64.ln(), 0.0, 0.0],
        [0.0, 0.5_f64.ln(), 0.0],
        [0.0, 0.0, 1.5_f64.ln()],
    ]);
    assert_eq_within_tols(&tensor.logm()?, &expected)
}

#[test]
fn logm_symmetric() -> Result<(), TestError> {
    assert_eq_within(
        &get_symmetric_tensor().logm()?,
        &get_symmetric_tensor_logm(),
        1e-10,
        1e-10,
    )
}

#[test]
fn logm_symmetric_trace_equals_ln_determinant() -> Result<(), TestError> {
    let tensor = get_symmetric_tensor();
    let determinant = tensor.determinant();
    assert_eq_within(&tensor.logm()?.trace(), &determinant.ln(), 1e-10, 1e-10)
}

#[test]
fn logm_near_identity_two_terms() -> Result<(), TestError> {
    let tensor = from_eigenvalues([1.00003, 0.999985, 1.00002]);
    let expected = from_eigenvalues([1.00003_f64.ln(), 0.999985_f64.ln(), 1.00002_f64.ln()]);
    assert_eq_within(&tensor.logm()?, &expected, 1e-9, 1e-9)
}

#[test]
fn logm_near_identity_three_terms() -> Result<(), TestError> {
    let tensor = from_eigenvalues([1.0005, 0.9996, 1.0002]);
    let expected = from_eigenvalues([1.0005_f64.ln(), 0.9996_f64.ln(), 1.0002_f64.ln()]);
    assert_eq_within(&tensor.logm()?, &expected, 1e-9, 1e-9)
}

#[test]
fn logm_near_identity_five_terms() -> Result<(), TestError> {
    let tensor = from_eigenvalues([1.003, 0.9985, 1.002]);
    let expected = from_eigenvalues([1.003_f64.ln(), 0.9985_f64.ln(), 1.002_f64.ln()]);
    assert_eq_within(&tensor.logm()?, &expected, 1e-9, 1e-9)
}

#[test]
fn dlogm_diagonal_matches_finite_difference_of_logm() -> Result<(), TestError> {
    let tensor = TensorRank2::<3, 1, 1>::from([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.5]]);
    let dlogm = tensor.dlogm()?;
    let epsilon = 1e-6;
    let directions = [
        TensorRank2::from([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        TensorRank2::from([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        TensorRank2::from([[0.3, 0.2, 0.1], [0.2, -0.4, 0.05], [0.1, 0.05, 0.1]]),
    ];
    for direction in directions.iter() {
        let perturbation = direction * epsilon;
        let tensor_plus = tensor.clone() + perturbation.clone();
        let tensor_minus = tensor.clone() - perturbation;
        let finite_difference = (tensor_plus.logm()? - tensor_minus.logm()?) / (2.0 * epsilon);
        let predicted = contract_third_fourth_indices(&dlogm, direction);
        assert_eq_within(&finite_difference, &predicted, 1e-6, 1e-6)?;
    }
    Ok(())
}

#[test]
fn dlogm_symmetric_matches_finite_difference_of_logm() -> Result<(), TestError> {
    let tensor = get_symmetric_tensor();
    let dlogm = tensor.dlogm()?;
    let epsilon = 1e-6;
    let directions = [
        TensorRank2::from([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        TensorRank2::from([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        TensorRank2::from([[0.3, 0.2, 0.1], [0.2, -0.4, 0.05], [0.1, 0.05, 0.1]]),
    ];
    for direction in directions.iter() {
        let perturbation = direction * epsilon;
        let tensor_plus = tensor.clone() + perturbation.clone();
        let tensor_minus = tensor.clone() - perturbation;
        let finite_difference = (tensor_plus.logm()? - tensor_minus.logm()?) / (2.0 * epsilon);
        let predicted = contract_third_fourth_indices(&dlogm, direction);
        assert_eq_within(&finite_difference, &predicted, 1e-6, 1e-6)?;
    }
    Ok(())
}
