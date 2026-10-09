use crate::{
    domain::{
        NodalAccelerations, NodalVelocities,
        solid::{
            NodalForcesSolid,
            mass::{InverseMass, NodalLumpedMasses, NodalMasses},
        },
    },
    math::assert::{Assert, AssertionError},
    units::{Energy, Mass},
};

fn lumped() -> NodalLumpedMasses {
    [Mass::kilograms(3.0), Mass::kilograms(2.0)].into()
}

fn consistent() -> NodalMasses {
    let mut masses = NodalMasses::zero(2);
    [(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)]
        .iter()
        .for_each(|&(a, b, mass)| masses[a][b] += Mass::kilograms(mass));
    masses
}

mod lumped_masses {
    use super::*;
    #[test]
    fn kinetic_energy_is_half_the_mass_times_the_speed_squared() -> Result<(), AssertionError> {
        let velocities = NodalVelocities::from([[1.0, 2.0, 2.0], [0.0, 0.0, 4.0]]);
        Assert::default().eq_within_tols(
            lumped().kinetic_energy(&velocities),
            &Energy::joules(0.5 * (3.0 * 9.0 + 2.0 * 16.0)),
        )
    }
    #[test]
    fn inertial_forces_are_the_mass_times_the_acceleration() -> Result<(), AssertionError> {
        let accelerations = NodalAccelerations::from([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        Assert::default().eq_within_tols(
            lumped().inertial_forces(&accelerations),
            NodalForcesSolid::from([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
        )
    }
    #[test]
    fn accelerations_are_the_net_force_over_the_mass() -> Result<(), AssertionError> {
        let external = NodalForcesSolid::from([[7.0, 0.0, 0.0], [0.0, 0.0, 6.0]]);
        let internal = NodalForcesSolid::from([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        Assert::default().eq_within_tols(
            lumped().nodal_accelerations(&external, &internal),
            NodalAccelerations::from([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),
        )
    }
}

mod consistent_masses {
    use super::*;
    #[test]
    fn kinetic_energy_of_a_uniform_motion_matches_the_lumped_masses() -> Result<(), AssertionError>
    {
        let velocities = NodalVelocities::from([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]);
        Assert::default().eq_within_tols(
            consistent().kinetic_energy(&velocities),
            &Energy::joules(12.0),
        )?;
        let row_sums: NodalLumpedMasses = [Mass::kilograms(3.0), Mass::kilograms(3.0)].into();
        Assert::default().eq_within_tols(
            consistent().kinetic_energy(&velocities),
            &row_sums.kinetic_energy(&velocities),
        )
    }
    #[test]
    fn inertial_forces_of_a_uniform_acceleration_are_the_row_sums() -> Result<(), AssertionError> {
        let accelerations = NodalAccelerations::from([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]);
        Assert::default().eq_within_tols(
            consistent().inertial_forces(&accelerations),
            NodalForcesSolid::from([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]]),
        )
    }
    #[test]
    fn solving_with_the_factorization_undoes_applying_the_mass() -> Result<(), AssertionError> {
        let masses = consistent();
        let accelerations = NodalAccelerations::from([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]);
        let forces = masses.inertial_forces(&accelerations);
        Assert::default().eq_within_tols(
            masses
                .factor()
                .unwrap()
                .nodal_accelerations(&forces, &NodalForcesSolid::zero(2)),
            accelerations,
        )
    }
}
