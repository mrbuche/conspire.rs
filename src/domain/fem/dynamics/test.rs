use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Model, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::Block,
        block::element::linear::Tetrahedron,
        solid::{NodalForcesSolid, elastic::ElasticElements},
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::{Density, Energy, Mass, STANDARD_GRAVITY, Stress},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

const COORDINATES: [[f64; 3]; 5] = [
    [0.1, 0.2, 0.0],
    [1.3, 0.1, 0.2],
    [0.2, 0.9, 0.1],
    [0.3, 0.4, 1.2],
    [1.5, 1.4, 1.3],
];

type B = Block<NeoHookean, Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;

fn model() -> Model<B, 3> {
    let reference = NodalReferenceCoordinates::from(COORDINATES);
    (
        B::from((
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            DENSITY,
            vec![[0, 1, 2, 3], [1, 2, 3, 4]],
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

mod kinetic_energy {
    use super::*;
    fn expected(speed: f64, mass: Quantity<Mass>) -> Quantity<Energy> {
        Energy::joules(0.5 * mass.in_kilograms() * speed * speed)
    }
    #[test]
    fn of_a_uniform_motion_is_half_the_mass_times_the_speed_squared() -> Result<(), AssertionError>
    {
        let model = model();
        let velocities = uniform_velocities([3.0, 0.0, 4.0]);
        let mass = model.nodal_lumped_masses().iter().copied().sum();
        Assert::default().eq_within_tols(
            model.nodal_lumped_masses().kinetic_energy(&velocities),
            &expected(5.0, mass),
        )?;
        Assert::default().eq_within_tols(
            model.nodal_masses().kinetic_energy(&velocities),
            &expected(5.0, mass),
        )
    }
}

mod inertial_forces {
    use super::*;
    #[test]
    fn consistent_and_lumped_agree_on_a_uniform_acceleration() -> Result<(), AssertionError> {
        let model = model();
        Assert::default().eq_within_tols(
            model.nodal_lumped_masses().inertial_forces(&gravity()),
            &model.nodal_masses().inertial_forces(&gravity()),
        )
    }
}

mod accelerations {
    use super::*;
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
}

mod consistent_accelerations {
    use super::*;
    use crate::fem::mass::InverseMass;
    #[test]
    fn solving_with_the_mass_undoes_applying_it() -> Result<(), AssertionError> {
        let masses = model().nodal_masses();
        let accelerations = NodalAccelerations::from(vec![
            [1.0, -2.0, 3.0],
            [0.5, 0.1, -STANDARD_GRAVITY.in_meters_per_second_squared()],
            [4.0, 4.0, 4.0],
            [-1.0, 0.0, 2.0],
            [7.0, -3.0, 0.25],
        ]);
        let forces = masses.inertial_forces(&accelerations);
        let recovered = masses
            .factor()
            .unwrap()
            .nodal_accelerations(&forces, &NodalForcesSolid::zero(COORDINATES.len()));
        Assert {
            abs_tol: 1e-9,
            rel_tol: 1e-9,
            ..Assert::default()
        }
        .eq_within_tols(&recovered, &accelerations)
    }
    #[test]
    fn a_body_in_its_reference_configuration_falls_with_gravity() -> Result<(), AssertionError> {
        let model = model();
        let masses = model.nodal_masses();
        let weights = masses.inertial_forces(&gravity());
        let accelerations = model
            .nodal_accelerations(
                &NodalCoordinates::from(COORDINATES),
                &weights,
                &masses.factor().unwrap(),
            )
            .unwrap();
        noise_tolerant().eq_within_tols(&accelerations, &gravity())
    }
}
