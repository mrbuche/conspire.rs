use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::solid::dynamics::test::fixed_dofs,
    domain::solid::hyperelastic::HyperelasticElements,
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
        integrate::{Newmark, VelocityVerlet},
        optimize::{NewtonRaphson, Tolerances},
    },
    units::{Density, Energy, STANDARD_GRAVITY, Stress, Time},
    vem::{
        ElasticExplicitDynamics, ElasticImplicitDynamics, Model, NodalAccelerations,
        NodalCoordinates, NodalForcesSolid, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block,
            element::{DEFAULT_STABILIZATION, Element},
        },
    },
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

const COORDINATES: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.3, 0.3, 0.3],
    [0.3, 0.3, -1.0],
];

const STEPS: usize = 10;

type B = Block<NeoHookean, Element, Quantity<Density>>;

fn model() -> Model<B, 3> {
    let reference = NodalReferenceCoordinates::from(COORDINATES);
    let faces = vec![
        vec![0, 1, 3],
        vec![1, 2, 3],
        vec![2, 0, 3],
        vec![1, 0, 4],
        vec![2, 1, 4],
        vec![0, 2, 4],
    ];
    (
        Block::<_, Element>::from((
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            vec![(0..faces.len()).collect()],
            faces,
            &reference,
            DEFAULT_STABILIZATION,
        ))
        .with_density(DENSITY),
        reference,
    )
        .into()
}

fn uniform_velocities(velocity: [f64; 3]) -> NodalVelocities {
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

fn time(dt: f64, steps: usize) -> [Quantity<Time>; 2] {
    [Time::seconds(0.0), Time::seconds(steps as f64 * dt)]
}

fn solver(residual: f64) -> NewtonRaphson {
    NewtonRaphson {
        abs_tol: Tolerances {
            constraint: 1e-12,
            residual,
        },
        rel_tol: Some(1e-6),
        ..Default::default()
    }
}

fn stretched() -> NodalCoordinates {
    let mut stretched = COORDINATES;
    stretched[4][2] -= 0.01;
    NodalCoordinates::from(stretched)
}

mod explicit {
    use super::*;
    #[test]
    fn a_free_body_falls_with_gravity() -> Result<(), AssertionError> {
        let (model, dt) = (model(), 1e-4);
        let masses = model.nodal_lumped_masses();
        let weights = masses.inertial_forces(&gravity());
        let (times, coordinates, velocities, accelerations) = ElasticExplicitDynamics::integrate(
            &model,
            &VelocityVerlet::new(Time::seconds(dt)),
            &time(dt, 100),
            (
                NodalCoordinates::from(COORDINATES),
                uniform_velocities([0.0; 3]),
            ),
            &weights,
            &masses,
            &fixed_dofs(&[]),
        )?;
        let (elapsed, g) = (
            times[times.len() - 1].in_seconds(),
            STANDARD_GRAVITY.in_meters_per_second_squared(),
        );
        noise_tolerant().eq_within_tols(
            &coordinates[100],
            &NodalCoordinates::from(
                COORDINATES.map(|[x, y, z]| [x, y, z - 0.5 * g * elapsed * elapsed]),
            ),
        )?;
        noise_tolerant().eq_within_tols(
            &velocities[100],
            &uniform_velocities([0.0, 0.0, -g * elapsed]),
        )?;
        noise_tolerant().eq_within_tols(&accelerations[100], &gravity())
    }
}

mod implicit {
    use super::*;
    fn integrator(dt: f64) -> Newmark {
        Newmark::new(Time::seconds(dt))
    }
    #[test]
    fn a_free_body_in_uniform_motion_translates_uniformly() -> Result<(), AssertionError> {
        let (model, dt) = (model(), 1e-3);
        let masses = model.nodal_lumped_masses();
        let (times, coordinates, ..) = ElasticImplicitDynamics::integrate(
            &model,
            &integrator(dt),
            solver(1e-3),
            &time(dt, STEPS),
            (
                NodalCoordinates::from(COORDINATES),
                uniform_velocities([3.0, -1.0, 2.0]),
            ),
            &NodalForcesSolid::zero(COORDINATES.len()),
            &masses,
            &fixed_dofs(&[]),
        )?;
        let elapsed = times[times.len() - 1].in_seconds();
        let expected =
            COORDINATES.map(|[x, y, z]| [x + 3.0 * elapsed, y - 1.0 * elapsed, z + 2.0 * elapsed]);
        noise_tolerant().eq_within_tols(&coordinates[STEPS], &NodalCoordinates::from(expected))
    }
    #[test]
    fn a_free_body_falls_with_gravity() -> Result<(), AssertionError> {
        let (model, dt) = (model(), 1e-3);
        let masses = model.nodal_lumped_masses();
        let weights = masses.inertial_forces(&gravity());
        let (times, coordinates, velocities, accelerations) = ElasticImplicitDynamics::integrate(
            &model,
            &integrator(dt),
            solver(1e-3),
            &time(dt, STEPS),
            (
                NodalCoordinates::from(COORDINATES),
                uniform_velocities([0.0; 3]),
            ),
            &weights,
            &masses,
            &fixed_dofs(&[]),
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
        let (model, dt) = (model(), 1e-3);
        let masses = model.nodal_lumped_masses();
        let weights = masses.inertial_forces(&gravity());
        let (_, coordinates, velocities, accelerations) = ElasticImplicitDynamics::integrate(
            &model,
            &integrator(dt),
            solver(1e-3),
            &time(dt, STEPS),
            (
                NodalCoordinates::from(COORDINATES),
                uniform_velocities([3.0, 0.0, 4.0]),
            ),
            &weights,
            &masses,
            &fixed_dofs(&[0, 1, 2, 5]),
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
    fn a_released_stretch_follows_the_explicit_motion() -> Result<(), AssertionError> {
        let (model, dt) = (model(), 2e-5);
        let masses = model.nodal_lumped_masses();
        let zero = NodalForcesSolid::zero(COORDINATES.len());
        let interval = time(dt, 100);
        let state = || (stretched(), uniform_velocities([0.0; 3]));
        let (_, implicit, ..) = ElasticImplicitDynamics::integrate(
            &model,
            &integrator(dt),
            solver(1.0),
            &interval,
            state(),
            &zero,
            &masses,
            &fixed_dofs(&[]),
        )?;
        let (_, explicit, ..) = ElasticExplicitDynamics::integrate(
            &model,
            &VelocityVerlet::new(Time::seconds(dt)),
            &interval,
            state(),
            &zero,
            &masses,
            &fixed_dofs(&[]),
        )?;
        let moved = (implicit[100][4][2] - implicit[0][4][2]).norm().value();
        assert!(moved > 0.0);
        Assert {
            abs_tol: 1e-2 * moved,
            rel_tol: 1e-2,
            ..Assert::default()
        }
        .eq_within_tols(&implicit[100], &explicit[100])
    }
    #[test]
    fn the_energy_stays_bounded_far_beyond_the_explicit_limit() -> Result<(), AssertionError> {
        let (model, dt) = (model(), 1e-4);
        let masses = model.nodal_lumped_masses();
        let steps = 200;
        let (_, coordinates, velocities, _) = ElasticImplicitDynamics::integrate(
            &model,
            &integrator(dt),
            solver(1.0),
            &time(dt, steps),
            (stretched(), uniform_velocities([0.0; 3])),
            &NodalForcesSolid::zero(COORDINATES.len()),
            &masses,
            &fixed_dofs(&[]),
        )?;
        let energy = |step: usize| -> Quantity<Energy> {
            masses.kinetic_energy(&velocities[step])
                + model.helmholtz_free_energy(&coordinates[step]).unwrap()
        };
        let initial = energy(0);
        assert!(initial.in_joules() > 0.0);
        (0..=steps).try_for_each(|step| {
            Assert {
                abs_tol: 0.0,
                rel_tol: 1e-2,
                ..Assert::default()
            }
            .eq_within_tols(energy(step), &initial)
        })
    }
}
