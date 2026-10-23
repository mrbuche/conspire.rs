use super::TvergaardHutchinson;
use crate::{
    EPSILON,
    constitutive::cohesive::{
        damage::{Damage, HyperDamage},
        test::{check_stiffness, perpendiculars},
    },
    math::{
        Quantity, Rank2,
        assert::{Assert, AssertionError, perturbation},
    },
    mechanics::{Normal, Scalar, Separation},
    units::{Length, Stress},
};

const NORMAL_LENGTH: Scalar = 0.1;
const TANGENTIAL_LENGTH: Scalar = 0.2;
const STATES: [Scalar; 5] = [0.0, 0.1, 0.5, 0.9, 1.2];
const OPENINGS: [(Scalar, Scalar); 5] = [
    (0.05, 0.03),
    (0.3, 0.2),
    (0.3, 0.9),
    (0.6, 0.5),
    (-0.3, 0.4),
];

fn model() -> TvergaardHutchinson {
    TvergaardHutchinson {
        strength: Stress::pascals(2.0),
        normal_length: Length::meters(NORMAL_LENGTH),
        tangential_length: Length::meters(TANGENTIAL_LENGTH),
        onset: 0.2,
        softening: 0.6,
    }
}

fn normals() -> [Normal; 2] {
    [
        Normal::from([0.0, 0.0, 1.0]),
        Normal::from([0.3, -0.5, 0.8]).normalized(),
    ]
}

fn lengths(normal: Scalar, tangential: Scalar) -> (Quantity<Length>, Quantity<Length>) {
    (
        Length::meters(normal * NORMAL_LENGTH),
        Length::meters(tangential * TANGENTIAL_LENGTH),
    )
}

fn tractions(normal: Scalar, tangential: Scalar, state: Scalar) -> (Scalar, Scalar) {
    let (n, t) = lengths(normal, tangential);
    let (n, t) = model().tractions(n, t, Quantity::new(state)).unwrap();
    (n.value(), t.value())
}

fn potential(normal: Scalar, tangential: Scalar, state: Scalar) -> Scalar {
    let (n, t) = lengths(normal, tangential);
    model()
        .potential(n, t, Quantity::new(state))
        .unwrap()
        .value()
}

fn separated(normal: &Normal, a: Scalar, b: Scalar) -> Separation {
    let (tangent, _) = perpendiculars(normal);
    normal * Length::meters(a * NORMAL_LENGTH) + tangent * Length::meters(b * TANGENTIAL_LENGTH)
}

fn close(a: Scalar, b: Scalar) -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(a, b)
}

fn limit(a: Scalar, b: Scalar) -> Result<(), AssertionError> {
    Assert {
        abs_tol: 1e-6,
        rel_tol: 1e-6,
        ..Assert::default()
    }
    .eq_within_tols(a, b)
}

#[test]
fn normal_curve() -> Result<(), AssertionError> {
    for (opening, expected) in [
        (0.1, 1.0),
        (0.2, 2.0),
        (0.4, 2.0),
        (0.6, 2.0),
        (0.8, 1.0),
        (1.0, 0.0),
        (1.5, 0.0),
    ] {
        let (n, t) = tractions(opening, 0.0, 0.0);
        close(n, expected)?;
        close(t, 0.0)?;
    }
    Ok(())
}

#[test]
fn tangential_strength_follows_the_lengths() -> Result<(), AssertionError> {
    for (opening, expected) in [(0.1, 0.5), (0.4, 1.0), (0.8, 0.5), (1.0, 0.0)] {
        let (n, t) = tractions(0.0, opening, 0.0);
        close(n, 0.0)?;
        close(t, expected)?;
    }
    Ok(())
}

#[test]
fn unloading_returns_on_the_secant() -> Result<(), AssertionError> {
    close(tractions(0.4, 0.0, 0.8).0, 2.0 * 0.625 * 0.4)?;
    close(tractions(0.0, 0.0, 0.8).0, 0.0)?;
    close(tractions(0.9, 0.0, 0.8).0, tractions(0.9, 0.0, 0.0).0)?;
    let (n, t) = tractions(0.5, 0.5, 1.2);
    close(n, 0.0)?;
    close(t, 0.0)
}

#[test]
fn compression_does_not_open_the_interface() -> Result<(), AssertionError> {
    let (n, t) = lengths(-0.3, 0.0);
    close(model().opening(n, t).unwrap().value(), 0.0)?;
    let (n, t) = lengths(-0.3, 0.4);
    close(model().opening(n, t).unwrap().value(), 0.4)?;
    let state = |z: Scalar, x: Scalar, old: Scalar| {
        model()
            .state(
                Separation::from([x * TANGENTIAL_LENGTH, 0.0, z * NORMAL_LENGTH]),
                Normal::from([0.0, 0.0, 1.0]),
                Quantity::new(old),
            )
            .unwrap()
            .value()
    };
    close(state(-0.9, 0.0, 0.3), 0.3)?;
    close(state(0.3, 0.4, 0.0), 0.5)?;
    close(state(0.3, 0.4, 0.7), 0.7)
}

#[test]
fn compression_has_the_secant_stiffness_of_tension() -> Result<(), AssertionError> {
    close(tractions(-0.3, 0.0, 0.0).0, -3.0)?;
    close(tractions(-0.3, 0.0, 0.9).0, -2.0 * 0.3 * 0.1 / (0.4 * 0.9))?;
    close(tractions(-0.3, 0.0, 1.2).0, 0.0)
}

#[test]
fn shear_softens_under_compression() -> Result<(), AssertionError> {
    let (n, t) = tractions(-0.3, 0.4, 0.0);
    close(n, -1.5)?;
    close(t, 0.71875)?;
    close(tractions(-0.3, 0.4, 0.9).1, tractions(0.0, 0.4, 0.9).1)?;
    close(tractions(-0.3, 0.1, 0.0).1, tractions(0.0, 0.1, 0.0).1)
}

#[test]
fn stiffness_is_continuous_through_zero_normal_separation() -> Result<(), AssertionError> {
    let k = |a: Scalar, state: Scalar| {
        let (n, t) = lengths(a, 0.3);
        model().stiffnesses(n, t, Quantity::new(state)).unwrap()
    };
    for state in STATES {
        let (below, above) = (k(-1e-9, state), k(1e-9, state));
        limit(below.0.value(), above.0.value())?;
        limit(below.1.value(), above.1.value())?;
        limit(below.2.value(), above.2.value())?;
        limit(below.3.value(), above.3.value())?;
    }
    Ok(())
}

#[test]
fn energy_to_fail_the_interface() -> Result<(), AssertionError> {
    let expected = 0.5 * 2.0 * NORMAL_LENGTH * (1.0 - 0.2 + 0.6);
    for (a, b) in [(1.0, 0.0), (0.0, 1.0), (2.0, 3.0)] {
        close(potential(a, b, 0.0), expected)?;
    }
    Ok(())
}

#[test]
fn potential_along_the_normal_curve() -> Result<(), AssertionError> {
    for (opening, expected) in [(0.1, 0.005), (0.4, 0.06), (0.8, 0.13), (1.0, 0.14)] {
        close(potential(opening, 0.0, 0.0), expected)?;
    }
    Ok(())
}

#[test]
fn potential_is_continuous_between_the_branches() -> Result<(), AssertionError> {
    for opening in [0.2, 0.6, 1.0] {
        limit(
            potential(opening - 1e-9, 0.0, 0.0),
            potential(opening + 1e-9, 0.0, 0.0),
        )?;
    }
    Ok(())
}

#[test]
fn potential_is_continuous_where_unloading_starts() -> Result<(), AssertionError> {
    for opening in [0.1, 0.4, 0.8] {
        close(
            potential(opening, 0.0, 0.0),
            potential(opening, 0.0, opening),
        )?;
    }
    close(potential(0.0, 0.0, 0.1), 0.0)?;
    assert!(potential(0.0, 0.0, 0.4) > 0.0);
    assert!(potential(0.0, 0.0, 0.8) > potential(0.0, 0.0, 0.4));
    Ok(())
}

#[test]
fn traction_is_the_derivative_of_the_potential() -> Result<(), AssertionError> {
    let step = perturbation::<Length>(0.5 * EPSILON);
    for state in STATES {
        let state = Quantity::new(state);
        let energy = |n, t| model().potential(n, t, state).unwrap();
        for (a, b) in OPENINGS {
            let (n, t) = lengths(a, b);
            let (normal, tangential) = model().tractions(n, t, state).unwrap();
            let by_normal =
                (energy(n + step, t) - energy(n - step, t)) / perturbation::<Length>(EPSILON);
            let by_tangential =
                (energy(n, t + step) - energy(n, t - step)) / perturbation::<Length>(EPSILON);
            Assert::default().eq_within_fd_tol(normal, &by_normal)?;
            Assert::default().eq_within_fd_tol(tangential, &by_tangential)?;
        }
    }
    Ok(())
}

#[test]
fn coupling_is_the_derivative_of_both_tractions() -> Result<(), AssertionError> {
    let step = perturbation::<Length>(0.5 * EPSILON);
    for state in STATES {
        let state = Quantity::new(state);
        let traction = |n, t| model().tractions(n, t, state).unwrap();
        for (a, b) in OPENINGS {
            let (n, t) = lengths(a, b);
            let (_, normal_tangential, tangential_normal, _) =
                model().stiffnesses(n, t, state).unwrap();
            let normal_by_tangential = (traction(n, t + step).0 - traction(n, t - step).0)
                / perturbation::<Length>(EPSILON);
            let tangential_by_normal = (traction(n + step, t).1 - traction(n - step, t).1)
                / perturbation::<Length>(EPSILON);
            Assert::default().eq_within_fd_tol(normal_tangential, &normal_by_tangential)?;
            Assert::default().eq_within_fd_tol(tangential_normal, &tangential_by_normal)?;
        }
    }
    Ok(())
}

#[test]
fn stiffness_is_symmetric() -> Result<(), AssertionError> {
    for state in STATES {
        for (a, b) in OPENINGS {
            let (n, t) = lengths(a, b);
            let state = Quantity::new(state);
            let (_, normal_tangential, tangential_normal, _) =
                model().stiffnesses(n, t, state).unwrap();
            Assert::default().eq_within_tols(normal_tangential, &tangential_normal)?;
        }
    }
    for normal in normals() {
        for (a, b) in OPENINGS {
            let stiffness = model()
                .stiffness(separated(&normal, a, b), normal.clone(), Quantity::new(0.0))
                .unwrap();
            Assert::default().eq_within_tols(&stiffness.0, &stiffness.0.transpose())?;
        }
    }
    Ok(())
}

#[test]
fn invalid_openings_are_errors() {
    for softening in [1.0, 0.1] {
        let mut model = model();
        model.softening = softening;
        let (n, t) = lengths(0.0, 0.0);
        assert!(model.tractions(n, t, Quantity::new(0.0)).is_err());
        assert!(model.potential(n, t, Quantity::new(0.0)).is_err());
    }
}

fn check(separation: Separation, normal: Normal, state: Scalar) -> Result<(), AssertionError> {
    let stiffness = model()
        .stiffness(separation.clone(), normal.clone(), Quantity::new(state))
        .unwrap();
    check_stiffness(
        |separation, normal| {
            model()
                .traction(separation, normal, Quantity::new(state))
                .unwrap()
        },
        &stiffness,
        &separation,
        &normal,
    )
}

#[test]
fn stiffness_on_every_branch() -> Result<(), AssertionError> {
    for normal in normals() {
        for (a, b) in OPENINGS {
            check(separated(&normal, a, b), normal.clone(), 0.0)?;
            check(separated(&normal, a, b), normal.clone(), 0.9)?;
        }
        check(separated(&normal, 0.4, 0.0), normal.clone(), 0.0)?;
        check(separated(&normal, -0.2, 0.0), normal.clone(), 0.0)?;
    }
    Ok(())
}

fn step_error(
    old: (Scalar, Scalar),
    new: (Scalar, Scalar),
    state_old: Scalar,
    state: Scalar,
) -> (Scalar, Scalar) {
    let error = model()
        .step_error(
            lengths(old.0, old.1),
            lengths(new.0, new.1),
            Quantity::new(state_old),
            Quantity::new(state),
        )
        .unwrap();
    (error.direction.value(), error.overshoot.value())
}

fn opening(point: (Scalar, Scalar)) -> Scalar {
    point.0.max(0.0).hypot(point.1)
}

fn loaded(old: (Scalar, Scalar), new: (Scalar, Scalar)) -> (Scalar, Scalar) {
    step_error(old, new, opening(old), opening(new))
}

#[test]
fn a_step_without_growth_of_the_greatest_opening_has_no_error() -> Result<(), AssertionError> {
    let (direction, overshoot) = step_error((0.4, 0.3), (0.1, 0.9), 0.9, 0.9);
    close(direction, 0.0)?;
    close(overshoot, 0.0)?;
    let (direction, overshoot) = step_error((0.0, 0.0), (0.0, 0.0), 0.0, 0.0);
    close(direction, 0.0)?;
    close(overshoot, 0.0)
}

#[test]
fn the_direction_vanishes_under_proportional_opening() -> Result<(), AssertionError> {
    for (old, new) in [
        ((0.1, 0.05), (0.8, 0.4)),
        ((0.0, 0.0), (0.5, 0.2)),
        ((0.3, 0.0), (2.0, 0.0)),
        ((0.0, 0.2), (0.0, 1.5)),
    ] {
        close(loaded(old, new).0, 0.0)?;
    }
    Ok(())
}

#[test]
fn the_direction_is_half_the_growth_times_the_turning() -> Result<(), AssertionError> {
    close(loaded((0.3, 0.0), (0.0, 0.6)).0, 0.5 * 0.3 * 2.0_f64.sqrt())
}

#[test]
fn the_direction_falls_as_the_square_of_the_step() -> Result<(), AssertionError> {
    let spiral = |t: Scalar| {
        let (radius, angle) = (0.3 + 0.3 * t, 0.5 * std::f64::consts::PI * t);
        (radius * angle.cos(), radius * angle.sin())
    };
    let error = |h: Scalar| loaded(spiral(0.0), spiral(h)).0;
    for h in [0.04, 0.02, 0.01] {
        assert!((error(h) / error(0.5 * h) - 4.0).abs() < 0.1);
    }
    Ok(())
}

#[test]
fn the_overshoot_is_how_far_the_step_runs_past_a_kink() -> Result<(), AssertionError> {
    for (state_old, state, expected) in [
        (0.1, 0.5, 0.3),
        (0.1, 0.7, 0.5),
        (0.5, 1.2, 0.6),
        (0.3, 0.7, 0.1),
        (0.7, 1.2, 0.2),
        (0.25, 0.5, 0.0),
        (0.7, 0.9, 0.0),
        (1.1, 1.4, 0.0),
    ] {
        let (_, overshoot) = step_error((state_old, 0.0), (state, 0.0), state_old, state);
        close(overshoot, expected)?;
    }
    Ok(())
}

#[test]
fn the_overshoot_falls_as_the_step() -> Result<(), AssertionError> {
    for h in [0.04, 0.02, 0.01] {
        let (_, overshoot) = loaded((0.6 - 0.5 * h, 0.0), (0.6 + 0.5 * h, 0.0));
        close(overshoot, 0.5 * h)?;
    }
    Ok(())
}

#[test]
fn reloading_past_the_greatest_opening_is_a_kink() -> Result<(), AssertionError> {
    let (_, overshoot) = step_error((0.4, 0.0), (0.9, 0.0), 0.8, 0.9);
    close(overshoot, 0.1)?;
    let (_, overshoot) = step_error((0.4, 0.0), (0.7, 0.0), 0.8, 0.8);
    close(overshoot, 0.0)?;
    let (_, overshoot) = step_error((0.8 * (1.0 - 1e-9), 0.0), (0.9, 0.0), 0.8, 0.9);
    close(overshoot, 0.0)
}

#[test]
fn compression_does_not_enter_the_error() -> Result<(), AssertionError> {
    let (direction, overshoot) = step_error((-0.5, 0.3), (-0.9, 0.5), 0.3, 0.5);
    close(direction, 0.0)?;
    close(overshoot, 0.0)
}

#[test]
fn an_invalid_opening_is_an_error_of_the_step() {
    let mut model = model();
    model.softening = 1.0;
    assert!(
        model
            .step_error(
                lengths(0.1, 0.0),
                lengths(0.2, 0.0),
                Quantity::new(0.1),
                Quantity::new(0.2)
            )
            .is_err()
    );
}

#[test]
fn the_elastic_opening_is_where_the_strength_is_reached() -> Result<(), AssertionError> {
    close(model().elastic_opening().value(), 0.2)
}
