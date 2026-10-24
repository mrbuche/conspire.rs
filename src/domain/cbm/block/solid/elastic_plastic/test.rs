use super::{PlasticStateVariablesField, RateIndependentRoot};
use crate::{
    cbm::{Model, NodalCoordinatesHistory, NodalReferenceCoordinates, block::Block},
    constitutive::{
        canonical::Canonical,
        fluid::plastic::{Linear, PlasticFlow, VonMises},
        solid::hyperelastic::NeoHookean,
    },
    domain::boundary::{BoundaryConditions, ConstrainAt, Displacement, Solid},
    geometry::mesh::PrimitiveConnectivity,
    math::{
        Quantity, Tensor,
        optimize::{NewtonRaphson, SolveStrategy},
    },
    units::{Dimensionless, Length, Stress, Time},
};

type Plastic = Block<Canonical<NeoHookean, PlasticFlow<VonMises, Linear>>>;

fn model() -> Model<Plastic, 3> {
    let reference = NodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]);
    (
        Plastic::from((
            Canonical::from((
                NeoHookean {
                    bulk_modulus: Stress::pascals(13.0),
                    shear_modulus: Stress::pascals(3.0),
                },
                PlasticFlow {
                    surface: VonMises,
                    hardening: Linear {
                        yield_stress: Stress::pascals(0.5),
                        hardening_slope: Stress::pascals(1.0),
                    },
                },
            )),
            PrimitiveConnectivity::from(vec![[0, 1, 2, 3], [1, 4, 2, 3]]),
            &reference,
        )),
        reference,
    )
        .into()
}

fn conditions() -> impl ConstrainAt<Solid<3>> {
    BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<1>, 0..5)
        .fix_nodes(Displacement::<2>, 0..5)
        .fix_nodes(Displacement::<0>, [0, 2, 3])
        .prescribe_nodes_with(Displacement::<0>, [1], |time| {
            Length::meters(0.6 * time.value())
        })
}

fn times(values: &[f64]) -> Vec<Quantity<Time>> {
    values.iter().copied().map(Time::seconds).collect()
}

fn run(
    values: &[f64],
    tolerance: f64,
) -> (
    NodalCoordinatesHistory<3>,
    Vec<PlasticStateVariablesField<1>>,
) {
    model()
        .root(
            NewtonRaphson::default(),
            &conditions(),
            &(),
            &times(values),
            SolveStrategy::Condensed(NewtonRaphson::default()),
            Dimensionless::of(tolerance),
        )
        .unwrap()
}

fn largest_plastic_strain(state: &PlasticStateVariablesField<1>) -> f64 {
    state
        .iter()
        .flat_map(|element| element.iter())
        .map(|variables| variables.1.value())
        .fold(0.0, f64::max)
}

#[test]
fn the_requested_times_are_the_only_ones_reported() {
    let (history, states) = run(&[0.0, 0.25, 1.0], 1e-1);
    assert_eq!(history.len(), 3);
    assert_eq!(states.len(), 3);
    assert_eq!(largest_plastic_strain(&states[0]), 0.0);
    let held = history[2][1][0].value();
    assert!((held - 1.6).abs() < 1e-9, "{held}");
}

#[test]
fn a_span_reports_every_accepted_step_and_a_tighter_tolerance_takes_more() {
    let loose = run(&[0.0, 1.0], 1e-1).0.len();
    let tight = run(&[0.0, 1.0], 1e-2).0.len();
    assert!(loose > 2, "{loose}");
    assert!(tight > loose, "{loose} {tight}");
}

#[test]
fn the_solution_yields_and_changes_little_with_the_tolerance() {
    let strain = |tolerance| largest_plastic_strain(run(&[0.0, 1.0], tolerance).1.last().unwrap());
    let (loose, tight) = (strain(1e-1), strain(1e-2));
    assert!(loose > 0.0);
    assert!((loose - tight).abs() < 2e-3, "{loose} {tight}");
}
