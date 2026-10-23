use super::{DamageCohesiveElement, DamageState, initial_state};
use crate::{
    EPSILON,
    constitutive::cohesive::damage::{Damage, TvergaardHutchinson},
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, ReferenceFiniteElement,
        cohesive::{CohesiveFiniteElement, linear::Hexahedron},
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError, perturbation},
    },
    mechanics::Scalar,
    units::{Area, Force, Length, Stress},
};

const G: usize = 4;
const N: usize = 8;
const P: usize = 4;
const REFERENCE: [[Scalar; 3]; N] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
];
const MODEL: TvergaardHutchinson = TvergaardHutchinson {
    strength: Stress::pascals(2.0),
    normal_length: Length::meters(0.1),
    tangential_length: Length::meters(0.2),
    onset: 0.2,
    softening: 0.6,
};

fn element() -> Hexahedron {
    Hexahedron::from(ElementNodalReferenceCoordinates::from(REFERENCE))
}

fn moved(displacement: [Scalar; 3]) -> ElementNodalCoordinates<N> {
    let mut coordinates = ElementNodalCoordinates::from(REFERENCE);
    coordinates.iter_mut().skip(P).for_each(|coordinate| {
        (0..3).for_each(|i| coordinate[i] += Quantity::new(displacement[i]))
    });
    coordinates
}

fn uniform(opening: Scalar) -> DamageState<G> {
    (0..G).map(|_| Quantity::new(opening)).collect()
}

fn area() -> Quantity<Area> {
    element().integration_weights().iter().copied().sum()
}

fn top_force(forces: &ElementNodalForcesSolid<N>, i: usize) -> Quantity<Force> {
    forces.iter().skip(P).map(|force| force[i]).sum()
}

#[test]
fn an_undamaged_element_at_its_reference_has_no_force_or_state() -> Result<(), AssertionError> {
    let state = initial_state::<G>();
    Assert::default().eq_within_tols(
        &element().nodal_forces(&MODEL, &moved([0.0; 3]), &state)?,
        &[[0.0; 3]; N].into(),
    )?;
    Assert::default().eq_within_tols(
        &element().updated_state(&MODEL, &moved([0.0; 3]), &state)?,
        &state,
    )
}

#[test]
fn loading_follows_the_curve() -> Result<(), AssertionError> {
    let forces = element().nodal_forces(&MODEL, &moved([0.0, 0.0, 0.05]), &initial_state::<G>())?;
    Assert::default().eq_within_tols(top_force(&forces, 2), &(Stress::pascals(2.0) * area()))?;
    Assert::default().eq_within_tols(
        forces
            .iter()
            .take(P)
            .map(|force| -force[2])
            .sum::<Quantity<Force>>(),
        &(Stress::pascals(2.0) * area()),
    )
}

#[test]
fn unloading_returns_on_the_secant() -> Result<(), AssertionError> {
    let state = element().updated_state(&MODEL, &moved([0.0, 0.0, 0.05]), &initial_state::<G>())?;
    Assert::default().eq_within_tols(&state, &uniform(0.5))?;
    let unloaded = moved([0.0, 0.0, 0.025]);
    let forces = element().nodal_forces(&MODEL, &unloaded, &state)?;
    Assert::default().eq_within_tols(top_force(&forces, 2), &(Stress::pascals(1.0) * area()))?;
    let virgin = element().nodal_forces(&MODEL, &unloaded, &initial_state::<G>())?;
    Assert::default().eq_within_tols(top_force(&virgin, 2), &(Stress::pascals(2.0) * area()))?;
    Assert::default().eq_within_tols(&element().updated_state(&MODEL, &unloaded, &state)?, &state)
}

#[test]
fn the_state_follows_the_greatest_opening_at_each_point() -> Result<(), AssertionError> {
    let coordinates = moved([0.0, 0.0, 0.09]);
    let state = element().updated_state(&MODEL, &coordinates, &uniform(0.2))?;
    Assert::default().eq_within_tols(&state, &uniform(0.9))?;
    Assert::default().eq_within_tols(
        &element().updated_state(&MODEL, &coordinates, &uniform(0.95))?,
        &uniform(0.95),
    )
}

fn check_stiffnesses(
    coordinates: &ElementNodalCoordinates<N>,
    state: &DamageState<G>,
) -> Result<(), AssertionError> {
    let mut finite_difference = Quantity::<Force>::new(0.0);
    let nodal_stiffnesses_fd = (0..N)
        .map(|a| {
            (0..N)
                .map(|b| {
                    (0..3)
                        .map(|i| {
                            (0..3)
                                .map(|j| {
                                    let mut nodal_coordinates = coordinates.clone();
                                    nodal_coordinates[b][j] += perturbation(0.5 * EPSILON);
                                    finite_difference = element().nodal_forces(
                                        &MODEL,
                                        &nodal_coordinates,
                                        state,
                                    )?[a][i];
                                    nodal_coordinates[b][j] -= perturbation(EPSILON);
                                    finite_difference -= element().nodal_forces(
                                        &MODEL,
                                        &nodal_coordinates,
                                        state,
                                    )?[a][i];
                                    Ok(finite_difference / perturbation::<Length>(EPSILON))
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect::<Result<ElementNodalStiffnessesSolid<N>, AssertionError>>()?;
    Assert::default().eq_within_fd_tol(
        &element().nodal_stiffnesses(&MODEL, coordinates, state)?,
        &nodal_stiffnesses_fd,
    )
}

#[test]
fn stiffness_is_the_derivative_of_the_forces() -> Result<(), AssertionError> {
    let coordinates = moved([0.03, 0.01, 0.07]);
    check_stiffnesses(&coordinates, &initial_state::<G>())?;
    check_stiffnesses(&coordinates, &uniform(0.95))?;
    check_stiffnesses(&moved([0.02, -0.03, -0.04]), &initial_state::<G>())
}

fn expected_error(
    old: &ElementNodalCoordinates<N>,
    new: &ElementNodalCoordinates<N>,
    state_old: &DamageState<G>,
    state: &DamageState<G>,
) -> (Scalar, Scalar) {
    let components = |coordinates: &ElementNodalCoordinates<N>| {
        Hexahedron::separations(coordinates)
            .into_iter()
            .map(|separation| {
                (
                    separation[2],
                    Length::meters(separation[0].value().hypot(separation[1].value())),
                )
            })
            .collect::<Vec<_>>()
    };
    components(old)
        .into_iter()
        .zip(components(new))
        .zip(state_old.iter().zip(state.iter()))
        .map(|((before, after), (&state_old, &state))| {
            let error = MODEL
                .step_error(before, after, state_old, state)
                .unwrap()
                .relative_to(MODEL.elastic_opening());
            (error.direction.value(), error.overshoot.value())
        })
        .fold((0.0, 0.0), |worst, (direction, overshoot)| {
            (worst.0.max(direction), worst.1.max(overshoot))
        })
}

#[test]
fn the_step_error_of_a_uniform_opening_is_that_of_a_point() -> Result<(), AssertionError> {
    let (old, new) = (moved([0.0; 3]), moved([0.0, 0.0, 0.05]));
    let state = element().updated_state(&MODEL, &new, &initial_state::<G>())?;
    let error = element().step_error(&MODEL, &old, &new, &initial_state::<G>(), &state)?;
    Assert::default().eq_within_tols(error.direction.value(), 0.0)?;
    Assert::default().eq_within_tols(error.overshoot.value(), 0.3 / 0.2)
}

#[test]
fn the_step_error_is_the_largest_over_the_points() -> Result<(), AssertionError> {
    let old = moved([0.0, 0.0, 0.01]);
    let mut new = moved([0.0, 0.0, 0.01]);
    new[P][2] += Quantity::new(0.12);
    new[P][0] += Quantity::new(0.04);
    new[P + 2][2] += Quantity::new(0.02);
    let state_old = element().updated_state(&MODEL, &old, &initial_state::<G>())?;
    let state = element().updated_state(&MODEL, &new, &state_old)?;
    let error = element().step_error(&MODEL, &old, &new, &state_old, &state)?;
    let (direction, overshoot) = expected_error(&old, &new, &state_old, &state);
    assert!(direction > 0.0 && overshoot > 0.0);
    Assert::default().eq_within_tols(error.direction.value(), direction)?;
    Assert::default().eq_within_tols(error.overshoot.value(), overshoot)
}

#[test]
fn a_step_that_does_not_load_has_no_step_error() -> Result<(), AssertionError> {
    let loaded = moved([0.0, 0.0, 0.05]);
    let state = element().updated_state(&MODEL, &loaded, &initial_state::<G>())?;
    let unloaded = moved([0.0, 0.0, 0.02]);
    let error = element().step_error(&MODEL, &loaded, &unloaded, &state, &state)?;
    Assert::default().eq_within_tols(error.direction.value(), 0.0)?;
    Assert::default().eq_within_tols(error.overshoot.value(), 0.0)
}
