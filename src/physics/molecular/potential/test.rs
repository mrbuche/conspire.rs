use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::molecular::potential::{Harmonic, Morse, Potential},
};

const NUM: usize = 333;

#[test]
fn finite_difference() -> Result<(), TestError> {
    let e = 1.2;
    let a = 1.1;
    let x0 = 1.5;
    let x_max = x0 + 0.98 * 2.0_f64.ln() / a;
    let potential = Harmonic {
        rest_length: x0,
        stiffness: e,
    };
    (0..NUM)
        .map(|k| x0 + (x_max - x0) * k as Scalar / NUM as Scalar)
        .into_iter()
        .try_for_each(|mut x| {
            let energy = potential.energy(x);
            let mut force = potential.force(x);
            let stiffness = potential.stiffness(x);
            x += 0.5 * EPSILON;
            let mut force_fd = potential.energy(x);
            let mut stiffness_fd = potential.force(x);
            x -= EPSILON;
            force_fd = (force_fd - potential.energy(x)) / EPSILON;
            stiffness_fd = (stiffness_fd - potential.force(x)) / EPSILON;
            assert_eq_from_fd(&force, &force_fd)?;
            assert_eq_from_fd(&stiffness, &stiffness_fd)?;
            let extension = potential.extension(force);
            let compliance = potential.compliance(force);
            force += 0.5 * EPSILON;
            let mut extension_fd = potential.legendre(force);
            let mut compliance_fd = potential.extension(force);
            force -= EPSILON;
            extension_fd = (potential.legendre(force) - extension_fd) / EPSILON;
            compliance_fd = (compliance_fd - potential.extension(force)) / EPSILON;
            assert_eq_from_fd(&extension, &extension_fd)?;
            assert_eq_from_fd(&compliance, &compliance_fd)
        })?;
    let potential = Morse {
        rest_length: x0,
        depth: e,
        parameter: a,
    };
    (0..NUM)
        .map(|k| x0 + (x_max - x0) * k as Scalar / NUM as Scalar)
        .into_iter()
        .try_for_each(|mut x| {
            let energy = potential.energy(x);
            let mut force = potential.force(x);
            let stiffness = potential.stiffness(x);
            x += 0.5 * EPSILON;
            let mut force_fd = potential.energy(x);
            let mut stiffness_fd = potential.force(x);
            x -= EPSILON;
            force_fd = (force_fd - potential.energy(x)) / EPSILON;
            stiffness_fd = (stiffness_fd - potential.force(x)) / EPSILON;
            assert_eq_from_fd(&force, &force_fd)?;
            assert_eq_from_fd(&stiffness, &stiffness_fd)?;
            let extension = potential.extension(force);
            let compliance = potential.compliance(force);
            force += 0.5 * EPSILON;
            let mut extension_fd = potential.legendre(force);
            let mut compliance_fd = potential.extension(force);
            force -= EPSILON;
            extension_fd = (potential.legendre(force) - extension_fd) / EPSILON;
            compliance_fd = (compliance_fd - potential.extension(force)) / EPSILON;
            assert_eq_from_fd(&extension, &extension_fd)?;
            assert_eq_from_fd(&compliance, &compliance_fd)
        })
}
