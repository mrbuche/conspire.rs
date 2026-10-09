use crate::{
    cbm::{
        ElasticDynamics, ElasticElements, HyperelasticElements, Model, NodalAccelerations,
        NodalCoordinates, NodalForcesSolid, NodalReferenceCoordinates, NodalVelocities,
        block::Block,
    },
    constitutive::solid::hyperelastic::NeoHookean,
    geometry::mesh::PrimitiveConnectivity,
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
        integrate::VelocityVerlet,
        optimize::EqualityConstraint,
    },
    units::{Density, Energy, STANDARD_GRAVITY, Stress, Time},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

const COORDINATES: [[f64; 3]; 5] = [
    [0.1, 0.2, 0.0],
    [1.3, 0.1, 0.2],
    [0.2, 0.9, 0.1],
    [0.3, 0.4, 1.2],
    [1.5, 1.4, 1.3],
];

const STEPS: usize = 100;

type B = Block<NeoHookean, Quantity<Density>>;

fn model() -> Model<B, 3> {
    let reference = NodalReferenceCoordinates::from(COORDINATES);
    (
        B::from((
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            DENSITY,
            PrimitiveConnectivity::from(vec![[0, 1, 2, 3], [1, 2, 3, 4]]),
            &reference,
        )),
        reference,
    )
        .into()
}

fn uniform_velocities(velocity: [f64; 3]) -> NodalVelocities<3> {
    COORDINATES.map(|_| velocity).into()
}

fn gravity() -> NodalAccelerations<3> {
    COORDINATES
        .map(|_| [0.0, 0.0, -STANDARD_GRAVITY.in_meters_per_second_squared()])
        .into()
}

fn noise_tolerant() -> Assert {
    Assert {
        abs_tol: 1e-6,
        rel_tol: 1e-6,
        ..Assert::default()
    }
}

fn integrator(dt: f64) -> VelocityVerlet {
    VelocityVerlet::new(Time::seconds(dt))
}

fn time(dt: f64) -> [Quantity<Time>; 2] {
    [Time::seconds(0.0), Time::seconds(STEPS as f64 * dt)]
}

#[test]
fn a_body_in_its_reference_configuration_falls_with_gravity() -> Result<(), AssertionError> {
    let model = model();
    let masses = model.nodal_lumped_masses();
    let weights = masses.inertial_forces(&gravity());
    let accelerations = model
        .nodal_accelerations(&NodalCoordinates::from(COORDINATES), &weights, &masses)
        .unwrap();
    noise_tolerant().eq_within_tols(&accelerations, &gravity())
}

#[test]
fn internal_forces_decelerate_what_they_resist() -> Result<(), AssertionError> {
    let model = model();
    let masses = model.nodal_lumped_masses();
    let mut stretched = COORDINATES;
    stretched[4][0] += 0.1;
    let stretched = NodalCoordinates::from(stretched);
    let zero = NodalForcesSolid::zero(COORDINATES.len());
    let internal = model.nodal_forces(&stretched).unwrap();
    let accelerations = model
        .nodal_accelerations(&stretched, &zero, &masses)
        .unwrap();
    let balance = masses.inertial_forces(&accelerations) + &internal;
    noise_tolerant().zero_within_tols(&balance)
}

#[test]
fn a_free_body_in_uniform_motion_translates_uniformly() -> Result<(), AssertionError> {
    let (model, dt) = (model(), 1e-4);
    let masses = model.nodal_lumped_masses();
    let (times, coordinates, ..) = model.integrate(
        &integrator(dt),
        &time(dt),
        (
            NodalCoordinates::from(COORDINATES),
            uniform_velocities([3.0, -1.0, 2.0]),
        ),
        &NodalForcesSolid::zero(COORDINATES.len()),
        &masses,
        EqualityConstraint::None,
    )?;
    let elapsed = times[times.len() - 1].in_seconds();
    let expected =
        COORDINATES.map(|[x, y, z]| [x + 3.0 * elapsed, y - 1.0 * elapsed, z + 2.0 * elapsed]);
    noise_tolerant().eq_within_tols(
        &coordinates[coordinates.len() - 1],
        &NodalCoordinates::from(expected),
    )
}

#[test]
fn a_free_body_falls_with_gravity() -> Result<(), AssertionError> {
    let (model, dt) = (model(), 1e-4);
    let masses = model.nodal_lumped_masses();
    let weights = masses.inertial_forces(&gravity());
    let (times, coordinates, velocities, accelerations) = model.integrate(
        &integrator(dt),
        &time(dt),
        (
            NodalCoordinates::from(COORDINATES),
            uniform_velocities([0.0; 3]),
        ),
        &weights,
        &masses,
        EqualityConstraint::None,
    )?;
    let (elapsed, g) = (
        times[times.len() - 1].in_seconds(),
        STANDARD_GRAVITY.in_meters_per_second_squared(),
    );
    noise_tolerant().eq_within_tols(
        &coordinates[STEPS],
        &NodalCoordinates::from(
            COORDINATES.map(|[x, y, z]| [x, y, z - 0.5 * g * elapsed * elapsed]),
        ),
    )?;
    noise_tolerant().eq_within_tols(
        &velocities[STEPS],
        &uniform_velocities([0.0, 0.0, -g * elapsed]),
    )?;
    noise_tolerant().eq_within_tols(&accelerations[STEPS], &gravity())
}

#[test]
fn fixed_degrees_of_freedom_do_not_move() -> Result<(), AssertionError> {
    let (model, dt) = (model(), 1e-4);
    let masses = model.nodal_lumped_masses();
    let weights = masses.inertial_forces(&gravity());
    let (_, coordinates, velocities, accelerations) = model.integrate(
        &integrator(dt),
        &time(dt),
        (
            NodalCoordinates::from(COORDINATES),
            uniform_velocities([3.0, 0.0, 4.0]),
        ),
        &weights,
        &masses,
        EqualityConstraint::Fixed(vec![0, 1, 2, 5]),
    )?;
    let last = &coordinates[STEPS];
    assert_eq!(last[0], NodalCoordinates::from(COORDINATES)[0]);
    assert_eq!(last[1][2], NodalCoordinates::from(COORDINATES)[1][2]);
    assert_eq!(velocities[STEPS][0], [0.0; 3].into());
    assert_eq!(accelerations[STEPS][0], [0.0; 3].into());
    assert_ne!(last[3], NodalCoordinates::from(COORDINATES)[3]);
    Ok(())
}

#[test]
fn the_total_energy_of_a_released_stretch_is_conserved() -> Result<(), AssertionError> {
    let (model, dt) = (model(), 1e-6);
    let masses = model.nodal_lumped_masses();
    let mut stretched = COORDINATES;
    stretched[4][0] += 0.01;
    let (_, coordinates, velocities, _) = model.integrate(
        &integrator(dt),
        &[Time::seconds(0.0), Time::seconds(2000.0 * dt)],
        (
            NodalCoordinates::from(stretched),
            uniform_velocities([0.0; 3]),
        ),
        &NodalForcesSolid::zero(COORDINATES.len()),
        &masses,
        EqualityConstraint::None,
    )?;
    let energy = |step: usize| -> Quantity<Energy> {
        masses.kinetic_energy(&velocities[step])
            + model.helmholtz_free_energy(&coordinates[step]).unwrap()
    };
    let initial = energy(0);
    assert!(initial.in_joules() > 0.0);
    assert!(velocities[2000].iter().any(|v| v.norm().value() > 0.0));
    (0..=2000).try_for_each(|step| {
        Assert {
            abs_tol: 0.0,
            rel_tol: 1e-3,
            ..Assert::default()
        }
        .eq_within_tols(energy(step), &initial)
    })
}
