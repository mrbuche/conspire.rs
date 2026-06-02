use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd, assert_eq_within_tols},
    },
    physics::molecular::potential::{Cosine, Harmonic, Morse, Potential},
};
use std::f64::consts::PI;

const NUM: usize = 333;

#[test]
fn test_consistency() -> Result<(), TestError> {
    let model = Harmonic {
        rest_length: 1.5,
        stiffness: 1.2,
    };
    let energy = model.energy(1.7);
    let forces = model.forces_at_energy(energy);
    let extensions = model.extensions_at_energy(energy);
    assert_eq_within_tols(&energy, &model.energy_at_force(forces[0]))?;
    assert_eq_within_tols(&energy, &model.energy_at_force(forces[1]))?;
    assert_eq_within_tols(&energy, &model.energy(extensions[0] + model.rest_length))?;
    assert_eq_within_tols(&energy, &model.energy(extensions[1] + model.rest_length))?;
    let model = Morse {
        rest_length: 1.5,
        depth: 1.9,
        parameter: 1.1,
    };
    let energy = model.energy(1.51);
    let forces = model.forces_at_energy(energy);
    let extensions = model.extensions_at_energy(energy);
    assert_eq_within_tols(&energy, &model.energy_at_force(forces[0]))?;
    assert_eq_within_tols(&energy, &model.energy_at_force(forces[1]))?;
    assert_eq_within_tols(&energy, &model.energy(extensions[0] + model.rest_length))?;
    assert_eq_within_tols(&energy, &model.energy(extensions[1] + model.rest_length))?;
    let model = Cosine {
        rest_length: 1.5,
        stiffness: 1.2,
    };
    let energy = model.energy(1.7);
    let forces = model.forces_at_energy(energy);
    let extensions = model.extensions_at_energy(energy);
    assert_eq_within_tols(&energy, &model.energy_at_force(forces[0]))?;
    assert_eq_within_tols(&energy, &model.energy_at_force(forces[1]))?;
    assert_eq_within_tols(&energy, &model.energy(extensions[0] + model.rest_length))?;
    assert_eq_within_tols(&energy, &model.energy(extensions[1] + model.rest_length))
}

#[test]
fn finite_difference() -> Result<(), TestError> {
    let e = 1.2;
    let a = 1.1;
    let x0 = 1.5;
    let x_max = x0 + 0.98 * 2.0_f64.ln() / a;
    let t = 1e-1;
    let potential = Harmonic {
        rest_length: x0,
        stiffness: e,
    };
    (0..NUM)
        .map(|k| x0 + (x_max - x0) * k as Scalar / NUM as Scalar)
        .into_iter()
        .try_for_each(|mut x| {
            let mut force = potential.force(x);
            let stiffness = potential.stiffness(x);
            let anharmonicity = potential.anharmonicity(x);
            assert_eq_within_tols(&potential.energy(x), &potential.energy_at_force(force))?;
            x += 0.5 * EPSILON;
            let mut force_fd = potential.energy(x);
            let mut stiffness_fd = potential.force(x);
            let mut anharmonicity_fd = potential.stiffness(x);
            x -= EPSILON;
            force_fd = (force_fd - potential.energy(x)) / EPSILON;
            stiffness_fd = (stiffness_fd - potential.force(x)) / EPSILON;
            anharmonicity_fd = (anharmonicity_fd - potential.stiffness(x)) / EPSILON;
            assert_eq_from_fd(&force, &force_fd)?;
            assert_eq_from_fd(&stiffness, &stiffness_fd)?;
            assert_eq_from_fd(&anharmonicity, &anharmonicity_fd)?;
            let extension = potential.extension(force);
            let compliance = potential.compliance(force);
            let nondimensional_extension = potential.nondimensional_extension(force, t);
            let nondimensional_force = potential.nondimensional_force(nondimensional_extension, t);
            assert_eq_within_tols(
                &potential.nondimensional_energy(nondimensional_extension, t),
                &potential.nondimensional_energy_at_nondimensional_force(nondimensional_force, t),
            )?;
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
    (1..NUM)
        .map(|k| x0 + (x_max - x0) * k as Scalar / NUM as Scalar)
        .into_iter()
        .try_for_each(|mut x| {
            let mut force = potential.force(x);
            let stiffness = potential.stiffness(x);
            let anharmonicity = potential.anharmonicity(x);
            assert_eq_within_tols(&potential.energy(x), &potential.energy_at_force(force))?;
            x += 0.5 * EPSILON;
            let mut force_fd = potential.energy(x);
            let mut stiffness_fd = potential.force(x);
            let mut anharmonicity_fd = potential.stiffness(x);
            x -= EPSILON;
            force_fd = (force_fd - potential.energy(x)) / EPSILON;
            stiffness_fd = (stiffness_fd - potential.force(x)) / EPSILON;
            anharmonicity_fd = (anharmonicity_fd - potential.stiffness(x)) / EPSILON;
            assert_eq_from_fd(&force, &force_fd)?;
            assert_eq_from_fd(&stiffness, &stiffness_fd)?;
            assert_eq_from_fd(&anharmonicity, &anharmonicity_fd)?;
            let extension = potential.extension(force);
            let compliance = potential.compliance(force);
            let nondimensional_extension = potential.nondimensional_extension(force, t);
            // let nondimensional_force = potential.nondimensional_force(nondimensional_extension, t);
            // assert_eq_within_tols(
            //     &potential.nondimensional_energy(nondimensional_extension, t),
            //     &potential.nondimensional_energy_at_nondimensional_force(nondimensional_force, t),
            // )?;
            force += 0.5 * EPSILON;
            let mut extension_fd = potential.legendre(force);
            let mut compliance_fd = potential.extension(force);
            let mut nondimensional_extension_fd = potential.nondimensional_legendre(force, t);
            force -= EPSILON;
            extension_fd = (potential.legendre(force) - extension_fd) / EPSILON;
            compliance_fd = (compliance_fd - potential.extension(force)) / EPSILON;
            nondimensional_extension_fd = (potential.nondimensional_legendre(force, t)
                - nondimensional_extension_fd)
                / EPSILON;
            assert_eq_from_fd(&extension, &extension_fd)?;
            assert_eq_from_fd(&compliance, &compliance_fd)?;
            assert_eq_from_fd(&nondimensional_extension, &nondimensional_extension_fd)
        })?;
    let potential = Cosine {
        rest_length: x0,
        stiffness: e,
    };
    let x_max = x0 + 0.8 * 0.5 * PI;
    (0..NUM)
        .map(|k| x0 + (x_max - x0) * k as Scalar / NUM as Scalar)
        .into_iter()
        .try_for_each(|mut x| {
            let mut force = potential.force(x);
            let stiffness = potential.stiffness(x);
            let anharmonicity = potential.anharmonicity(x);
            assert_eq_within_tols(&potential.energy(x), &potential.energy_at_force(force))?;
            x += 0.5 * EPSILON;
            let mut force_fd = potential.energy(x);
            let mut stiffness_fd = potential.force(x);
            let mut anharmonicity_fd = potential.stiffness(x);
            x -= EPSILON;
            force_fd = (force_fd - potential.energy(x)) / EPSILON;
            stiffness_fd = (stiffness_fd - potential.force(x)) / EPSILON;
            anharmonicity_fd = (anharmonicity_fd - potential.stiffness(x)) / EPSILON;
            assert_eq_from_fd(&force, &force_fd)?;
            assert_eq_from_fd(&stiffness, &stiffness_fd)?;
            assert_eq_from_fd(&anharmonicity, &anharmonicity_fd)?;
            let extension = potential.extension(force);
            let compliance = potential.compliance(force);
            let nondimensional_extension = potential.nondimensional_extension(force, t);
            let nondimensional_force = potential.nondimensional_force(nondimensional_extension, t);
            assert_eq_within_tols(
                &potential.nondimensional_energy(nondimensional_extension, t),
                &potential.nondimensional_energy_at_nondimensional_force(nondimensional_force, t),
            )?;
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
