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

mod integrate {
    use super::*;
    use crate::{
        fem::solid::hyperelastic::HyperelasticElements,
        math::{integrate::VelocityVerlet, optimize::EqualityConstraint},
        units::Time,
    };
    const STEPS: usize = 100;
    fn integrator(dt: f64) -> VelocityVerlet {
        VelocityVerlet::new(Time::seconds(dt))
    }
    fn time(dt: f64) -> [Quantity<Time>; 2] {
        [Time::seconds(0.0), Time::seconds(STEPS as f64 * dt)]
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
        assert_eq!(times.len(), STEPS + 1);
        assert_eq!(coordinates.len(), STEPS + 1);
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
        let stretched = NodalCoordinates::from(stretched);
        let (_, coordinates, velocities, _) = model.integrate(
            &integrator(dt),
            &[Time::seconds(0.0), Time::seconds(2000.0 * dt)],
            (stretched, uniform_velocities([0.0; 3])),
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
    #[test]
    fn linear_constraints_are_not_supported() {
        let (model, dt) = (model(), 1e-4);
        let masses = model.nodal_lumped_masses();
        let result = model.integrate(
            &integrator(dt),
            &time(dt),
            (
                NodalCoordinates::from(COORDINATES),
                uniform_velocities([0.0; 3]),
            ),
            &NodalForcesSolid::zero(COORDINATES.len()),
            &masses,
            EqualityConstraint::Linear(Default::default(), Default::default()),
        );
        assert!(result.is_err());
    }
}

mod consistent_masses {
    use super::*;
    use crate::{
        fem::mass::{InverseMass, MassMatrix},
        math::{integrate::VelocityVerlet, optimize::EqualityConstraint},
        units::Time,
    };
    const FIXED: [usize; 4] = [0, 1, 2, 5];
    fn integrator() -> VelocityVerlet {
        VelocityVerlet::new(Time::seconds(1e-4))
    }
    fn forces() -> NodalForcesSolid<3> {
        NodalForcesSolid::from(vec![
            [1.0, -2.0, 3.0],
            [0.5, 0.1, -4.0],
            [4.0, 4.0, 4.0],
            [-1.0, 0.0, 2.0],
            [7.0, -3.0, 0.25],
        ])
    }
    #[test]
    fn fixed_degrees_of_freedom_are_eliminated_from_the_solve() -> Result<(), AssertionError> {
        let masses = model().nodal_masses();
        let inverse = MassMatrix::<3>::inverse(&masses, &FIXED).unwrap();
        let accelerations = inverse.nodal_accelerations(&forces(), &NodalForcesSolid::zero(5));
        FIXED
            .iter()
            .for_each(|&index| assert_eq!(accelerations[index / 3][index % 3].value(), 0.0));
        let mut residual = masses.inertial_forces(&accelerations) - &forces();
        FIXED
            .iter()
            .for_each(|&index| residual[index / 3][index % 3] = Default::default());
        Assert {
            abs_tol: 1e-9,
            rel_tol: 1e-9,
            ..Assert::default()
        }
        .zero_within_tols(&residual)
    }
    #[test]
    fn without_fixed_degrees_of_freedom_the_full_mass_is_solved() -> Result<(), AssertionError> {
        let masses = model().nodal_masses();
        let free = MassMatrix::<3>::inverse(&masses, &[])
            .unwrap()
            .nodal_accelerations(&forces(), &NodalForcesSolid::zero(5));
        let full = masses
            .factor::<3>()
            .unwrap()
            .nodal_accelerations(&forces(), &NodalForcesSolid::zero(5));
        Assert {
            abs_tol: 1e-9,
            rel_tol: 1e-9,
            ..Assert::default()
        }
        .eq_within_tols(&free, &full)
    }
    #[test]
    fn a_free_body_falls_with_gravity() -> Result<(), AssertionError> {
        let model = model();
        let masses = model.nodal_masses();
        let weights = masses.inertial_forces(&gravity());
        let (times, coordinates, ..) = model.integrate(
            &integrator(),
            &[Time::seconds(0.0), Time::seconds(0.01)],
            (
                NodalCoordinates::from(COORDINATES),
                uniform_velocities([0.0; 3]),
            ),
            &weights,
            &masses,
            EqualityConstraint::None,
        )?;
        let elapsed = times[times.len() - 1].in_seconds();
        let g = STANDARD_GRAVITY.in_meters_per_second_squared();
        noise_tolerant().eq_within_tols(
            &coordinates[coordinates.len() - 1],
            &NodalCoordinates::from(
                COORDINATES.map(|[x, y, z]| [x, y, z - 0.5 * g * elapsed * elapsed]),
            ),
        )
    }
    #[test]
    fn fixed_degrees_of_freedom_do_not_move() -> Result<(), AssertionError> {
        let model = model();
        let masses = model.nodal_masses();
        let weights = masses.inertial_forces(&gravity());
        let (_, coordinates, velocities, accelerations) = model.integrate(
            &integrator(),
            &[Time::seconds(0.0), Time::seconds(0.01)],
            (
                NodalCoordinates::from(COORDINATES),
                uniform_velocities([3.0, 0.0, 4.0]),
            ),
            &weights,
            &masses,
            EqualityConstraint::Fixed(FIXED.to_vec()),
        )?;
        let (last, initial) = (
            &coordinates[coordinates.len() - 1],
            NodalCoordinates::from(COORDINATES),
        );
        let end = velocities.len() - 1;
        FIXED.iter().for_each(|&index| {
            assert_eq!(last[index / 3][index % 3], initial[index / 3][index % 3]);
            assert_eq!(velocities[end][index / 3][index % 3].value(), 0.0);
            assert_eq!(accelerations[end][index / 3][index % 3].value(), 0.0);
        });
        assert_ne!(last[3], initial[3]);
        Ok(())
    }
}

mod bar_wave {
    use super::*;
    use crate::{
        fem::{
            NodalAccelerationsHistory, NodalCoordinatesHistory, NodalVelocitiesHistory,
            block::element::linear::Hexahedron,
        },
        math::{
            integrate::{DormandPrince, DormandPrinceFixedStep, ExplicitDynamics, VelocityVerlet},
            optimize::EqualityConstraint,
        },
        units::Time,
    };
    type Bar = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
    const ELEMENTS: usize = 40;
    const LENGTH: f64 = 1.0;
    const WIDTH: f64 = 0.1;
    const SHEAR_MODULUS: f64 = 3.0e9;
    const BULK_MODULUS: f64 = 13.0e9;
    const STRAIN: f64 = 1.0e-4;
    fn constrained_modulus() -> f64 {
        BULK_MODULUS + 4.0 * SHEAR_MODULUS / 3.0
    }
    fn wave_speed() -> f64 {
        (constrained_modulus() / DENSITY.in_kilograms_per_cubic_meter()).sqrt()
    }
    fn layer(k: usize) -> [[f64; 3]; 4] {
        let z = LENGTH * k as f64 / ELEMENTS as f64;
        [
            [0.0, 0.0, z],
            [WIDTH, 0.0, z],
            [WIDTH, WIDTH, z],
            [0.0, WIDTH, z],
        ]
    }
    fn coordinates() -> Vec<[f64; 3]> {
        (0..=ELEMENTS).flat_map(layer).collect()
    }
    fn model() -> Model<Bar, 3> {
        let reference = NodalReferenceCoordinates::from(coordinates());
        (
            Bar::from((
                NeoHookean {
                    shear_modulus: Stress::pascals(SHEAR_MODULUS),
                    bulk_modulus: Stress::pascals(BULK_MODULUS),
                },
                DENSITY,
                (0..ELEMENTS)
                    .map(|k| std::array::from_fn(|i| 4 * k + i))
                    .collect::<Vec<[usize; 8]>>(),
                &reference,
            )),
            reference,
        )
            .into()
    }
    fn fixed() -> Vec<usize> {
        let nodes = 4 * (ELEMENTS + 1);
        (0..nodes)
            .flat_map(|node| [3 * node, 3 * node + 1])
            .chain((0..4).map(|node| 3 * node + 2))
            .collect()
    }
    fn tip_load() -> (NodalForcesSolid<3>, f64) {
        let force = STRAIN * constrained_modulus() * WIDTH * WIDTH;
        let mut forces = vec![[0.0; 3]; 4 * (ELEMENTS + 1)];
        (4 * ELEMENTS..4 * (ELEMENTS + 1)).for_each(|node| forces[node][2] = force / 4.0);
        (NodalForcesSolid::from(forces), STRAIN * LENGTH)
    }
    type History = (crate::math::integrate::Times, Vec<f64>);
    type Failure = crate::math::integrate::IntegrationError;
    fn run_with(
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        time: &[Quantity<Time>],
    ) -> Result<History, Failure> {
        let model = model();
        let masses = model.nodal_lumped_masses();
        let (forces, _) = tip_load();
        let (times, coordinates, ..) = model.integrate(
            integrator,
            time,
            (
                NodalCoordinates::from(self::coordinates()),
                NodalVelocities::from(vec![[0.0; 3]; 4 * (ELEMENTS + 1)]),
            ),
            &forces,
            &masses,
            EqualityConstraint::Fixed(fixed()),
        )?;
        let tip = 4 * ELEMENTS;
        let displacements = coordinates
            .iter()
            .map(|coordinates| coordinates[tip][2].value() - LENGTH)
            .collect();
        Ok((times, displacements))
    }
    fn run(dt: f64, end: f64) -> Result<History, Failure> {
        run_with(
            &VelocityVerlet::new(Time::seconds(dt)),
            &[Time::seconds(0.0), Time::seconds(end)],
        )
    }
    fn grid(dt: f64, end: f64) -> Vec<Quantity<Time>> {
        let steps = (end / dt).ceil() as usize;
        (0..=steps)
            .map(|step| Time::seconds(end * step as f64 / steps as f64))
            .collect()
    }
    fn peak((times, displacements): &History) -> (f64, f64) {
        let (step, displacement) =
            displacements
                .iter()
                .copied()
                .enumerate()
                .fold(
                    (0, f64::MIN),
                    |best, (step, u)| if u > best.1 { (step, u) } else { best },
                );
        (times[step].value(), displacement)
    }
    fn assert_wave_peak(history: &History) {
        let transit = 2.0 * LENGTH / wave_speed();
        let static_deflection = tip_load().1;
        let (time, displacement) = peak(history);
        assert!(
            (time / transit - 1.0).abs() < 0.03,
            "peak at {time} of {transit}"
        );
        assert!(
            (displacement / (2.0 * static_deflection) - 1.0).abs() < 0.05,
            "peak {displacement} of {}",
            2.0 * static_deflection
        );
    }
    fn element_transit_time() -> f64 {
        LENGTH / ELEMENTS as f64 / wave_speed()
    }
    #[test]
    fn the_tip_of_a_loaded_bar_peaks_after_two_wave_transits_at_twice_the_static_deflection() {
        let transit = 2.0 * LENGTH / wave_speed();
        let (times, displacements) = run(0.5 * element_transit_time(), 1.5 * transit).unwrap();
        let (peak, displacement) =
            displacements
                .iter()
                .copied()
                .enumerate()
                .fold(
                    (0, f64::MIN),
                    |best, (step, u)| if u > best.1 { (step, u) } else { best },
                );
        let static_deflection = tip_load().1;
        assert!(
            (times[peak].value() / transit - 1.0).abs() < 0.03,
            "peak at {} of {transit}",
            times[peak].value()
        );
        assert!(
            (displacement / (2.0 * static_deflection) - 1.0).abs() < 0.05,
            "peak {displacement} of {}",
            2.0 * static_deflection
        );
    }
    #[test]
    fn the_tip_of_a_loaded_bar_moves_at_the_strain_times_the_wave_speed_until_the_wave_returns() {
        let transit = 2.0 * LENGTH / wave_speed();
        let (times, displacements) = run(0.5 * element_transit_time(), 0.5 * transit).unwrap();
        let end = displacements.len() - 1;
        let expected = STRAIN * wave_speed() * times[end].value();
        assert!(
            (displacements[end] / expected - 1.0).abs() < 0.02,
            "{} of {expected}",
            displacements[end]
        );
    }
    #[test]
    fn a_stacked_fixed_step_integrator_reproduces_the_wave_peak() {
        let transit = 2.0 * LENGTH / wave_speed();
        let history = run_with(
            &DormandPrinceFixedStep::default(),
            &grid(0.5 * element_transit_time(), 1.5 * transit),
        )
        .unwrap();
        assert_wave_peak(&history);
    }
    #[test]
    fn a_stacked_adaptive_integrator_reproduces_the_wave_peak() {
        let transit = 2.0 * LENGTH / wave_speed();
        let history = run_with(
            &DormandPrince::default(),
            &[Time::seconds(0.0), Time::seconds(1.5 * transit)],
        )
        .unwrap();
        assert_wave_peak(&history);
    }
    #[test]
    fn stacked_and_velocity_verlet_agree_on_the_same_grid() {
        let transit = 2.0 * LENGTH / wave_speed();
        let time = grid(0.5 * element_transit_time(), 1.5 * transit);
        let stacked = run_with(&DormandPrinceFixedStep::default(), &time).unwrap();
        let verlet = run_with(&VelocityVerlet::new(Time::seconds(0.0)), &time).unwrap();
        assert_eq!(stacked.1.len(), verlet.1.len());
        let scale = 2.0 * tip_load().1;
        stacked
            .1
            .iter()
            .zip(verlet.1.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 0.05 * scale, "{a} against {b}"));
    }
    #[test]
    fn a_time_step_above_the_transit_time_of_an_element_diverges() {
        let transit = 2.0 * LENGTH / wave_speed();
        let stable = run(0.9 * element_transit_time(), 1.5 * transit).unwrap().1;
        let unstable = run(1.5 * element_transit_time(), 1.5 * transit);
        let static_deflection = tip_load().1;
        assert!(stable.iter().all(|u| u.abs() < 3.0 * static_deflection));
        assert!(match unstable {
            Err(_) => true,
            Ok((_, displacements)) => displacements
                .iter()
                .any(|u| !u.is_finite() || u.abs() > 1.0e3 * static_deflection),
        });
    }
}
