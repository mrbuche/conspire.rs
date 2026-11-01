use crate::{
    cbm::{
        Checked, ElasticElements, ElasticExplicitDynamics, Execution, Lumped, Model,
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities, Unchecked, block::Block,
    },
    constitutive::solid::hyperelastic::NeoHookean,
    domain::solid::dynamics::test::fixed_dofs,
    domain::{
        solid::time_scale::TimeScaleElements,
        time_scale::{largest_eigenvalue, time_scale_from_eigenvalue},
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{
        Quantity, Tensor,
        integrate::{IntegrationError, VelocityVerlet},
    },
    units::{Density, Stress, Time},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

const COORDINATES: [[f64; 3]; 5] = [
    [0.1, 0.2, 0.0],
    [1.3, 0.1, 0.2],
    [0.2, 0.9, 0.1],
    [0.3, 0.4, 1.2],
    [1.5, 1.4, 1.3],
];

type B = Block<NeoHookean, Quantity<Density>>;

fn model(coordinates: [[f64; 3]; 5]) -> Model<B, 3> {
    let reference = NodalReferenceCoordinates::from(coordinates);
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

fn global_time_scale(model: &Model<B, 3>, coordinates: &NodalCoordinates<3>) -> f64 {
    let stiffnesses = model.nodal_stiffnesses(coordinates).unwrap();
    let masses: Vec<f64> = model
        .nodal_lumped_masses()
        .iter()
        .flat_map(|mass| [mass.value(); 3])
        .collect();
    let mut dense = vec![vec![0.0; masses.len()]; masses.len()];
    stiffnesses.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(b, block)| {
            (0..3).for_each(|i| {
                (0..3).for_each(|j| dense[3 * a + i][3 * b + j] = block[i][j].value())
            })
        })
    });
    time_scale_from_eigenvalue(largest_eigenvalue(
        masses.len(),
        |row, column| dense[row][column],
        &masses,
    ))
    .in_seconds()
}

#[test]
fn the_patch_bound_is_not_longer_than_the_time_scale_of_the_assembled_system() {
    let model = model(COORDINATES);
    let coordinates = NodalCoordinates::from(COORDINATES);
    let bound = model
        .fastest_time_scale(&NodalReferenceCoordinates::from(COORDINATES), &coordinates)
        .unwrap()
        .in_seconds();
    let exact = global_time_scale(&model, &coordinates);
    assert!(bound.is_finite() && bound > 0.0);
    assert!(bound <= exact * (1.0 + 1e-8), "{bound} > {exact}");
    assert!(bound > 0.1 * exact, "{bound} is far below {exact}");
}

#[test]
fn the_bound_scales_with_the_square_root_of_the_density()
-> Result<(), crate::math::assert::AssertionError> {
    let reference = NodalReferenceCoordinates::from(COORDINATES);
    let coordinates = NodalCoordinates::from(COORDINATES);
    let scaled = |density: Quantity<Density>| {
        B::from((
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            density,
            PrimitiveConnectivity::from(vec![[0, 1, 2, 3], [1, 2, 3, 4]]),
            &reference,
        ))
        .fastest_time_scale(&reference, &coordinates)
        .unwrap()
        .in_seconds()
    };
    let ratio = scaled(DENSITY * 4.0) / scaled(DENSITY);
    crate::math::assert::Assert {
        abs_tol: 1e-6,
        rel_tol: 1e-6,
        ..Default::default()
    }
    .eq_within_tols(ratio, &2.0)
}

#[test]
fn a_stiffer_body_is_faster() {
    let reference = NodalReferenceCoordinates::from(COORDINATES);
    let coordinates = NodalCoordinates::from(COORDINATES);
    let stiff = |factor: f64| {
        B::from((
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9 * factor),
                bulk_modulus: Stress::pascals(13.0e9 * factor),
            },
            DENSITY,
            PrimitiveConnectivity::from(vec![[0, 1, 2, 3], [1, 2, 3, 4]]),
            &reference,
        ))
        .fastest_time_scale(&reference, &coordinates)
        .unwrap()
        .in_seconds()
    };
    assert!(stiff(4.0) < stiff(1.0));
}

#[test]
fn a_bounded_integration_runs_below_the_limit_and_errors_above_it() {
    let model = model(COORDINATES);
    let limit = model
        .fastest_time_scale(
            &NodalReferenceCoordinates::from(COORDINATES),
            &NodalCoordinates::from(COORDINATES),
        )
        .unwrap()
        .in_seconds();
    let run = |dt: f64| {
        model.integrate(
            Execution::Serial,
            Checked {
                safety: 1.0,
                interval: 1,
            },
            Lumped,
            &VelocityVerlet::new(Time::seconds(dt)),
            &[Time::seconds(0.0), Time::seconds(10.0 * dt)],
            (
                NodalCoordinates::from(COORDINATES),
                NodalVelocities::from(COORDINATES.map(|_| [0.0; 3])),
            ),
            &(),
            &fixed_dofs(&[]),
        )
    };
    assert!(run(0.5 * limit).is_ok());
    assert!(run(4.0 * limit).is_err());
}

/// A short run of the model at rest, at a time step, with a stability policy.
fn rest(
    model: &Model<B, 3>,
    stability: impl crate::cbm::Stability<Model<B, 3>, Lumped>,
    dt: f64,
) -> Result<(), IntegrationError> {
    model
        .integrate(
            Execution::Serial,
            stability,
            Lumped,
            &VelocityVerlet::new(Time::seconds(dt)),
            &[Time::seconds(0.0), Time::seconds(10.0 * dt)],
            (
                NodalCoordinates::from(COORDINATES),
                NodalVelocities::from(COORDINATES.map(|_| [0.0; 3])),
            ),
            &(),
            &fixed_dofs(&[]),
        )
        .map(|_| ())
}

fn limit() -> f64 {
    model(COORDINATES)
        .fastest_time_scale(
            &NodalReferenceCoordinates::from(COORDINATES),
            &NodalCoordinates::from(COORDINATES),
        )
        .unwrap()
        .in_seconds()
}

#[test]
fn an_unchecked_integration_accepts_a_step_that_a_checked_one_refuses() {
    let (model, step) = (model(COORDINATES), 4.0 * limit());
    let checked = Checked {
        safety: 1.0,
        interval: 1,
    };
    assert!(matches!(
        rest(&model, checked, step),
        Err(IntegrationError::UnstableTimeStep(..))
    ));
    assert!(rest(&model, Unchecked, step).is_ok());
}

#[test]
fn a_smaller_fraction_of_the_limit_refuses_a_step_that_a_larger_one_accepts() {
    let (model, step) = (model(COORDINATES), 1.5 * limit());
    let checked = |safety| Checked {
        safety,
        interval: 1,
    };
    assert!(rest(&model, checked(1.0), step).is_ok());
    assert!(rest(&model, checked(0.5), step).is_err());
}

#[test]
fn a_checked_integration_needs_an_interval() {
    let model = model(COORDINATES);
    let checked = Checked {
        safety: 1.0,
        interval: 0,
    };
    assert!(matches!(
        rest(&model, checked, 0.5 * limit()),
        Err(IntegrationError::Intermediate(message)) if message.contains("at least one")
    ));
}

#[test]
fn a_checked_integration_with_a_longer_interval_still_checks_the_first_step() {
    let (model, step) = (model(COORDINATES), 4.0 * limit());
    let checked = Checked {
        safety: 1.0,
        interval: 1000,
    };
    assert!(rest(&model, checked, step).is_err());
}
