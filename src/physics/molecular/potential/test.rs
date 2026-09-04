use crate::math::assert::Assert;
use crate::{
    EPSILON,
    math::{
        Quantity, Scalar,
        assert::{AssertionError, perturbation},
    },
    physics::molecular::potential::{AngularPotential, Cosine, Harmonic, Morse, Potential},
    units::{Force, Length, Temperature},
};
use std::f64::consts::PI;

const NUM: usize = 333;

#[test]
fn test_consistency() -> Result<(), AssertionError> {
    let model = Harmonic {
        rest_length: 1.5,
        stiffness: 1.2,
    };
    let energy = model.energy(Quantity::new(1.7));
    let forces = model.forces_at_energy(energy);
    let extensions = model.extensions_at_energy(energy);
    Assert::default().eq_within_tols(energy, &model.energy_at_force(forces[0]))?;
    Assert::default().eq_within_tols(energy, &model.energy_at_force(forces[1]))?;
    Assert::default().eq_within_tols(energy, &model.energy(extensions[0] + model.rest_length()))?;
    Assert::default().eq_within_tols(energy, &model.energy(extensions[1] + model.rest_length()))?;
    let model = Morse {
        rest_length: 1.5,
        depth: 1.9,
        parameter: 1.1,
    };
    let energy = model.energy(Quantity::new(1.51));
    let forces = model.forces_at_energy(energy);
    let extensions = model.extensions_at_energy(energy);
    Assert::default().eq_within_tols(energy, &model.energy_at_force(forces[0]))?;
    Assert::default().eq_within_tols(energy, &model.energy_at_force(forces[1]))?;
    Assert::default().eq_within_tols(energy, &model.energy(extensions[0] + model.rest_length()))?;
    Assert::default().eq_within_tols(energy, &model.energy(extensions[1] + model.rest_length()))?;
    let model = Cosine {
        rest_angle: 1.5,
        stiffness: 1.2,
    };
    Assert::default().eq_within_tols(model.energy(model.rest_angle), &Quantity::new(0.0))?;
    Assert::default().eq_within_tols(
        model.energy(model.rest_angle + 0.3),
        &model.energy(model.rest_angle - 0.3),
    )
}

#[test]
fn finite_difference() -> Result<(), AssertionError> {
    let e = 1.2;
    let a = 1.1;
    let x0 = 1.5;
    let x_max = x0 + 0.98 * 2.0_f64.ln() / a;
    let t = Quantity::<Temperature>::new(1e-1);
    let potential = Harmonic {
        rest_length: x0,
        stiffness: e,
    };
    (0..NUM)
        .map(|k| Quantity::new(x0 + (x_max - x0) * k as Scalar / NUM as Scalar))
        .into_iter()
        .try_for_each(|mut x| {
            let mut force = potential.force(x);
            let stiffness = potential.stiffness(x);
            let anharmonicity = potential.anharmonicity(x);
            Assert::default()
                .eq_within_tols(potential.energy(x), &potential.energy_at_force(force))?;
            x += perturbation(0.5 * EPSILON);
            let force_fd = potential.energy(x);
            let stiffness_fd = potential.force(x);
            let anharmonicity_fd = potential.stiffness(x);
            x -= perturbation(EPSILON);
            let force_fd = (force_fd - potential.energy(x)) / perturbation::<Length>(EPSILON);
            let stiffness_fd =
                (stiffness_fd - potential.force(x)) / perturbation::<Length>(EPSILON);
            let anharmonicity_fd =
                (anharmonicity_fd - potential.stiffness(x)) / perturbation::<Length>(EPSILON);
            Assert::default().eq_within_fd_tol(force, &force_fd)?;
            Assert::default().eq_within_fd_tol(stiffness, &stiffness_fd)?;
            Assert::default().eq_within_fd_tol(anharmonicity, &anharmonicity_fd)?;
            let extension = potential.extension(force);
            let compliance = potential.compliance(force);
            let nondimensional_extension = potential.nondimensional_extension(force.value(), t);
            let nondimensional_force = potential.nondimensional_force(nondimensional_extension, t);
            Assert::default().eq_within_tols(
                potential.nondimensional_energy(nondimensional_extension, t),
                &potential.nondimensional_energy_at_nondimensional_force(nondimensional_force, t),
            )?;
            force += perturbation(0.5 * EPSILON);
            let extension_fd = potential.legendre(force);
            let compliance_fd = potential.extension(force);
            force -= perturbation(EPSILON);
            let extension_fd =
                (potential.legendre(force) - extension_fd) / perturbation::<Force>(EPSILON);
            let compliance_fd =
                (compliance_fd - potential.extension(force)) / perturbation::<Force>(EPSILON);
            Assert::default().eq_within_fd_tol(extension, &extension_fd)?;
            Assert::default().eq_within_fd_tol(compliance, &compliance_fd)
        })?;
    let potential = Morse {
        rest_length: x0,
        depth: e,
        parameter: a,
    };
    (1..NUM)
        .map(|k| Quantity::new(x0 + (x_max - x0) * k as Scalar / NUM as Scalar))
        .into_iter()
        .try_for_each(|mut x| {
            let mut force = potential.force(x);
            let stiffness = potential.stiffness(x);
            let anharmonicity = potential.anharmonicity(x);
            Assert::default()
                .eq_within_tols(potential.energy(x), &potential.energy_at_force(force))?;
            x += perturbation(0.5 * EPSILON);
            let force_fd = potential.energy(x);
            let stiffness_fd = potential.force(x);
            let anharmonicity_fd = potential.stiffness(x);
            x -= perturbation(EPSILON);
            let force_fd = (force_fd - potential.energy(x)) / perturbation::<Length>(EPSILON);
            let stiffness_fd =
                (stiffness_fd - potential.force(x)) / perturbation::<Length>(EPSILON);
            let anharmonicity_fd =
                (anharmonicity_fd - potential.stiffness(x)) / perturbation::<Length>(EPSILON);
            Assert::default().eq_within_fd_tol(force, &force_fd)?;
            Assert::default().eq_within_fd_tol(stiffness, &stiffness_fd)?;
            Assert::default().eq_within_fd_tol(anharmonicity, &anharmonicity_fd)?;
            let extension = potential.extension(force);
            let compliance = potential.compliance(force);
            let nondimensional_extension = potential.nondimensional_extension(force.value(), t);
            // let nondimensional_force = potential.nondimensional_force(nondimensional_extension, t);
            // Assert::default().eq_within_tols(
            //     &potential.nondimensional_energy(nondimensional_extension, t),
            //     &potential.nondimensional_energy_at_nondimensional_force(nondimensional_force, t),
            // )?;
            force += perturbation(0.5 * EPSILON);
            let extension_fd = potential.legendre(force);
            let compliance_fd = potential.extension(force);
            let mut nondimensional_extension_fd =
                potential.nondimensional_legendre(force.value(), t);
            force -= perturbation(EPSILON);
            let extension_fd =
                (potential.legendre(force) - extension_fd) / perturbation::<Force>(EPSILON);
            let compliance_fd =
                (compliance_fd - potential.extension(force)) / perturbation::<Force>(EPSILON);
            nondimensional_extension_fd = (potential.nondimensional_legendre(force.value(), t)
                - nondimensional_extension_fd)
                / EPSILON;
            Assert::default().eq_within_fd_tol(extension, &extension_fd)?;
            Assert::default().eq_within_fd_tol(compliance, &compliance_fd)?;
            Assert::default()
                .eq_within_fd_tol(nondimensional_extension, &nondimensional_extension_fd)
        })?;
    let potential = Cosine {
        rest_angle: x0,
        stiffness: e,
    };
    let x_max = x0 + 0.8 * 0.5 * PI;
    (0..NUM)
        .map(|k| x0 + (x_max - x0) * k as Scalar / NUM as Scalar)
        .into_iter()
        .try_for_each(|x| {
            Assert::default().eq_within_tols(
                potential.energy(x),
                &Quantity::new(e * (1.0 - (x - x0).cos())),
            )
        })
}
