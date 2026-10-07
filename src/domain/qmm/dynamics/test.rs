use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::{
        NodalCoordinates, NodalVelocities,
        mass::{InverseMass, MassMatrix},
        qmm::{Discretization, Support, block::Block},
        solid::NodalForcesSolid,
    },
    geometry::mesh::test::tetrahedra,
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
        integrate::VelocityVerlet,
        optimize::EqualityConstraint,
    },
    units::{Density, Stress, Time},
};
use std::sync::LazyLock;

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

const STEPS: usize = 200;

const DT: f64 = 1e-5;

static DISCRETIZATION: LazyLock<Discretization> = LazyLock::new(|| {
    let support = |spacing, reach| Support {
        spacing: Quantity::new(spacing),
        reach,
    };
    Discretization::new(&tetrahedra(8), support(0.4, 2.6), support(0.2, 3.6), 3, 1).unwrap()
});

type B = Block<NeoHookean, Quantity<Density>>;

fn model() -> crate::domain::Model<B, 3> {
    let discretization = DISCRETIZATION.clone();
    let coordinates = discretization.coordinates().clone();
    (
        Block::from((
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            discretization,
        ))
        .with_density(DENSITY),
        coordinates,
    )
        .into()
}

fn reference() -> Vec<[f64; 3]> {
    DISCRETIZATION
        .coordinates()
        .iter()
        .map(|coordinate| [0, 1, 2].map(|c| coordinate[c].value()))
        .collect()
}

fn stretched() -> NodalCoordinates<3> {
    let mut coordinates = reference();
    coordinates[1][0] += 0.01;
    coordinates[2][1] -= 0.01;
    coordinates.into()
}

fn uniform_velocities(velocity: [f64; 3]) -> NodalVelocities<3> {
    reference()
        .iter()
        .map(|_| velocity)
        .collect::<Vec<_>>()
        .into()
}

fn integrator() -> VelocityVerlet {
    VelocityVerlet::new(Time::seconds(DT))
}

fn time() -> [Quantity<Time>; 2] {
    [Time::seconds(0.0), Time::seconds(STEPS as f64 * DT)]
}

fn noise_tolerant() -> Assert {
    Assert {
        abs_tol: 1e-9,
        rel_tol: 1e-9,
        ..Assert::default()
    }
}

#[test]
fn a_free_body_in_uniform_motion_translates_uniformly() -> Result<(), AssertionError> {
    let model = model();
    let masses = model.nodal_lumped_masses();
    let (times, coordinates, ..) = model.integrate(
        &integrator(),
        &time(),
        (reference().into(), uniform_velocities([3.0, -1.0, 2.0])),
        &NodalForcesSolid::zero(reference().len()),
        &masses,
        EqualityConstraint::None,
    )?;
    let elapsed = times[times.len() - 1].in_seconds();
    let expected: Vec<[f64; 3]> = reference()
        .iter()
        .map(|&[x, y, z]| [x + 3.0 * elapsed, y - elapsed, z + 2.0 * elapsed])
        .collect();
    noise_tolerant().eq_within_tols(
        &coordinates[coordinates.len() - 1],
        &NodalCoordinates::from(expected),
    )
}

#[test]
fn lumped_momentum_is_conserved_by_a_released_stretch() {
    let model = model();
    let masses = model.nodal_lumped_masses();
    let (_, _, velocities, _) = model
        .integrate(
            &integrator(),
            &time(),
            (stretched(), uniform_velocities([0.0; 3])),
            &NodalForcesSolid::zero(reference().len()),
            &masses,
            EqualityConstraint::None,
        )
        .unwrap();
    let last = &velocities[STEPS];
    let speed = last.iter().map(|v| v.norm().value()).fold(0.0, f64::max);
    assert!(speed > 0.0, "the stretch did not move the body");
    let momentum: [f64; 3] = [0, 1, 2].map(|c| {
        masses
            .iter()
            .zip(last.iter())
            .map(|(mass, velocity)| mass.value() * velocity[c].value())
            .sum()
    });
    let scale = masses.iter().map(|mass| mass.value()).sum::<f64>() * speed;
    momentum
        .iter()
        .for_each(|&component| assert!(component.abs() < 1e-9 * scale, "{momentum:?}"));
}

#[test]
fn fixed_degrees_of_freedom_do_not_move() {
    let model = model();
    let masses = model.nodal_lumped_masses();
    let fixed = vec![0, 1, 2, 5];
    let (_, coordinates, velocities, accelerations) = model
        .integrate(
            &integrator(),
            &time(),
            (stretched(), uniform_velocities([3.0, 0.0, 4.0])),
            &NodalForcesSolid::zero(reference().len()),
            &masses,
            EqualityConstraint::Fixed(fixed.clone()),
        )
        .unwrap();
    fixed.iter().for_each(|&dof| {
        let (node, c) = (dof / 3, dof % 3);
        assert_eq!(coordinates[STEPS][node][c], stretched()[node][c]);
        assert_eq!(velocities[STEPS][node][c].value(), 0.0);
        assert_eq!(accelerations[STEPS][node][c].value(), 0.0);
    });
    assert_ne!(coordinates[STEPS][3], stretched()[3]);
}

#[test]
fn consistent_masses_conserve_momentum_and_agree_on_a_uniform_acceleration() {
    let model = model();
    let masses = model.nodal_masses();
    let (_, _, velocities, _) = model
        .integrate(
            &integrator(),
            &time(),
            (stretched(), uniform_velocities([0.0; 3])),
            &NodalForcesSolid::zero(reference().len()),
            &masses,
            EqualityConstraint::None,
        )
        .unwrap();
    let last = &velocities[STEPS];
    let speed = last.iter().map(|v| v.norm().value()).fold(0.0, f64::max);
    assert!(speed > 0.0);
    let total: [f64; 3] = [0, 1, 2].map(|c| {
        masses
            .iter()
            .flat_map(|row| row.entries())
            .map(|(_, mass)| mass.value())
            .zip(
                masses
                    .iter()
                    .flat_map(|row| row.entries().map(|(b, _)| b))
                    .map(|b| last[b][c].value()),
            )
            .map(|(mass, velocity)| mass * velocity)
            .sum()
    });
    let scale = model
        .nodal_lumped_masses()
        .iter()
        .map(|mass| mass.value())
        .sum::<f64>()
        * speed;
    total
        .iter()
        .for_each(|&component| assert!(component.abs() < 1e-9 * scale, "{total:?}"));
    let inverse = MassMatrix::<3>::inverse(&masses, &[]).unwrap();
    let uniform = [0.0, 0.0, -9.81];
    let accelerations = inverse.nodal_accelerations(
        &masses.inertial_forces(
            &reference()
                .iter()
                .map(|_| uniform)
                .collect::<Vec<_>>()
                .into(),
        ),
        &NodalForcesSolid::zero(reference().len()),
    );
    accelerations.iter().for_each(|a| {
        assert!((a[2].value() + 9.81).abs() < 1e-6, "{}", a[2].value());
    });
}
