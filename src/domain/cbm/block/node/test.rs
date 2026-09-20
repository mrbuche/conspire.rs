use super::{ElementNodalReferenceCoordinates, Weighting};

#[test]
fn solid_angle_weights_are_uniform_for_a_regular_tetrahedron() {
    let coordinates = ElementNodalReferenceCoordinates::from([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]);
    Weighting::SolidAngle
        .weights(&coordinates)
        .iter()
        .for_each(|&weight| assert!((weight - 0.25).abs() < 1e-6, "{weight}"));
}

#[test]
fn solid_angle_weights_sum_to_one_and_differ_for_an_irregular_tetrahedron() {
    let coordinates = ElementNodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let weights = Weighting::SolidAngle.weights(&coordinates);
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "{sum}");
    assert!(weights.iter().any(|&weight| (weight - 0.25).abs() > 0.01));
}
