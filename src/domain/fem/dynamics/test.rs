use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Model, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::Block,
        block::element::linear::Tetrahedron,
        solid::{NodalForcesSolid, elastic::ElasticElements},
    },
    math::{Quantity, Tensor},
    units::{Density, Energy, Stress},
};

const EPSILON: f64 = 1e-12;

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
    Model::from((
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
    ))
}

fn uniform_velocities(velocity: [f64; 3]) -> NodalVelocities<3> {
    NodalVelocities::from(COORDINATES.map(|_| velocity))
}

mod kinetic_energy {
    use super::*;

    fn expected(speed: f64, mass: Quantity<crate::units::Mass>) -> Quantity<Energy> {
        Quantity::new(0.5 * mass.value() * speed * speed)
    }

    #[test]
    fn of_a_uniform_motion_is_half_the_mass_times_the_speed_squared() {
        let model = model();
        let velocities = uniform_velocities([3.0, 0.0, 4.0]);
        let mass = model.nodal_lumped_masses().iter().copied().sum();
        assert!(
            !model
                .nodal_lumped_masses()
                .kinetic_energy(&velocities)
                .differs(expected(5.0, mass), EPSILON)
        );
        assert!(
            !model
                .nodal_masses()
                .kinetic_energy(&velocities)
                .differs(expected(5.0, mass), EPSILON)
        );
    }
}

mod inertial_forces {
    use super::*;

    fn accelerations() -> NodalAccelerations<3> {
        NodalAccelerations::from(COORDINATES.map(|_| [0.0, 0.0, -9.81]))
    }

    #[test]
    fn consistent_and_lumped_agree_on_a_uniform_acceleration() {
        let model = model();
        let lumped = model
            .nodal_lumped_masses()
            .inertial_forces(&accelerations());
        let consistent = model.nodal_masses().inertial_forces(&accelerations());
        lumped.iter().zip(consistent.iter()).for_each(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .for_each(|(a_i, b_i)| assert!(!a_i.differs(*b_i, EPSILON)))
        });
    }
}

mod accelerations {
    use super::*;

    #[test]
    fn a_body_in_its_reference_configuration_falls_with_gravity() {
        let model = model();
        let masses = model.nodal_lumped_masses();
        let gravity = NodalAccelerations::from(COORDINATES.map(|_| [0.0, 0.0, -9.81]));
        let weights: NodalForcesSolid<3> = masses.inertial_forces(&gravity);
        let accelerations = model
            .nodal_accelerations(&NodalCoordinates::from(COORDINATES), &weights, &masses)
            .unwrap();
        accelerations.iter().zip(gravity.iter()).for_each(|(a, g)| {
            a.iter()
                .zip(g.iter())
                .for_each(|(a_i, g_i)| assert!(!a_i.differs_severely(*g_i, 1e-6)))
        });
    }

    #[test]
    fn internal_forces_decelerate_what_they_resist() {
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
        let resisted = masses.inertial_forces(&accelerations);
        resisted.iter().zip(internal.iter()).for_each(|(r, f)| {
            r.iter().zip(f.iter()).for_each(|(r_i, f_i)| {
                assert!((r_i + f_i).value().abs() <= 1e-9 * f_i.value().abs().max(1.0))
            })
        });
    }
}

mod consistent_accelerations {
    use super::*;
    use crate::fem::mass::InverseMass;

    fn assert_close(a: &NodalAccelerations<3>, b: &NodalAccelerations<3>, tolerance: f64) {
        a.iter().zip(b.iter()).for_each(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .for_each(|(a_i, b_i)| assert!(!a_i.differs_severely(*b_i, tolerance)))
        });
    }

    #[test]
    fn solving_with_the_mass_undoes_applying_it() {
        let masses = model().nodal_masses();
        let accelerations = NodalAccelerations::from(vec![
            [1.0, -2.0, 3.0],
            [0.5, 0.1, -9.81],
            [4.0, 4.0, 4.0],
            [-1.0, 0.0, 2.0],
            [7.0, -3.0, 0.25],
        ]);
        let forces = masses.inertial_forces(&accelerations);
        let recovered = masses
            .factor()
            .unwrap()
            .nodal_accelerations(&forces, &NodalForcesSolid::zero(COORDINATES.len()));
        assert_close(&recovered, &accelerations, 1e-9);
    }

    #[test]
    fn a_body_in_its_reference_configuration_falls_with_gravity() {
        let model = model();
        let masses = model.nodal_masses();
        let gravity = NodalAccelerations::from(COORDINATES.map(|_| [0.0, 0.0, -9.81]));
        let weights = masses.inertial_forces(&gravity);
        let accelerations = model
            .nodal_accelerations(
                &NodalCoordinates::from(COORDINATES),
                &weights,
                &masses.factor().unwrap(),
            )
            .unwrap();
        assert_close(&accelerations, &gravity, 1e-6);
    }
}
