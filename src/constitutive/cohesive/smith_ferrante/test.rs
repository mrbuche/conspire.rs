use super::{super::test::SMITHFERRANTEPARAMETERS, *};
use crate::{math::TensorArray, EPSILON};

#[test]
fn finite_difference_1() {
    let model = SmithFerrante::new(SMITHFERRANTEPARAMETERS);
    let mut displacement = Displacement::new([0.1, 0.2, 0.3]);
    let normal = Normal::new([-0.4, 0.2, -0.1]).normalized();
    let mut fd = 0.0;
    model
        .calculate_stiffnesses(&displacement, &normal)
        .0
        .iter()
        .enumerate()
        .for_each(|(i, stiffness_i)| {
            stiffness_i
                .iter()
                .enumerate()
                .for_each(|(j, stiffness_ij)| {
                    displacement = Displacement::new([0.1, 0.2, 0.3]);
                    displacement[j] += 0.5 * EPSILON;
                    fd = model.calculate_traction(&displacement, &normal)[i];
                    displacement[j] -= EPSILON;
                    fd -= model.calculate_traction(&displacement, &normal)[i];
                    assert!((stiffness_ij / fd * EPSILON - 1.0).abs() < EPSILON)
                })
        })
}

#[test]
fn finite_difference_2() {
    let model = SmithFerrante::new(SMITHFERRANTEPARAMETERS);
    let displacement = Displacement::new([0.1, 0.2, 0.3]);
    let mut normal = Normal::new([-0.4, 0.2, -0.1]).normalized();
    let mut fd = 0.0;
    model
        .calculate_stiffnesses(&displacement, &normal)
        .1
        .iter()
        .enumerate()
        .for_each(|(i, stiffness_i)| {
            stiffness_i
                .iter()
                .enumerate()
                .for_each(|(j, stiffness_ij)| {
                    normal = Normal::new([-0.4, 0.2, -0.1]).normalized();
                    normal[j] += 0.5 * EPSILON;
                    fd = model.calculate_traction(&displacement, &normal)[i];
                    normal[j] -= EPSILON;
                    fd -= model.calculate_traction(&displacement, &normal)[i];
                    assert!((stiffness_ij / fd * EPSILON - 1.0).abs() < EPSILON)
                })
        })
}
