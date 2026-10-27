use super::{Outcome, StepError, StepIndicators, Stepper, adapt};
use crate::{
    math::Quantity,
    units::{Dimensionless, Length, Time},
};

const TOLERANCE: f64 = 1e-3;

fn error(direction: f64, overshoot: f64) -> StepError {
    StepError {
        direction: Dimensionless::of(direction),
        overshoot: Dimensionless::of(overshoot),
    }
}

fn judge(stepper: &mut Stepper, direction: f64, overshoot: f64) -> Outcome {
    let step = stepper.propose();
    stepper
        .judge(
            &step,
            error(direction, overshoot),
            Dimensionless::of(TOLERANCE),
        )
        .unwrap()
}

#[test]
fn fewer_than_two_times_or_times_not_increasing_are_refused() {
    for times in [vec![], vec![0.0], vec![1.0, 0.0], vec![0.5, 0.5]] {
        let times = times.into_iter().map(Time::seconds).collect::<Vec<_>>();
        assert!(Stepper::new(&times).is_err());
    }
}

#[test]
fn the_first_step_is_the_whole_span() {
    let times = [0.0, 2.0].map(Time::seconds);
    let step = Stepper::new(&times).unwrap().propose();
    assert_eq!(step.time, Time::seconds(2.0));
}

#[test]
fn a_step_longer_than_the_distance_to_a_breakpoint_ends_on_it_exactly() {
    let times = [0.0, 0.3, 1.0].map(Time::seconds);
    let step = Stepper::new(&times).unwrap().propose();
    assert_eq!(step.time, Time::seconds(0.3));
}

#[test]
fn a_small_error_takes_the_step_and_grows_the_next() {
    let times = [0.0, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    assert_eq!(
        judge(&mut stepper, 0.0, 0.0),
        Outcome::Accepted { report: true }
    );
    assert!(stepper.is_done());
}

#[test]
fn a_large_error_rejects_the_step_and_shrinks_the_next() {
    let times = [0.0, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    let first = stepper.propose();
    assert_eq!(judge(&mut stepper, 1.0, 0.0), Outcome::Rejected);
    assert!(stepper.propose().time < first.time);
    assert!(!stepper.is_done());
}

#[test]
fn either_error_alone_can_reject() {
    let times = [0.0, 1.0].map(Time::seconds);
    for (direction, overshoot) in [(1.0, 0.0), (0.0, 1.0)] {
        let mut stepper = Stepper::new(&times).unwrap();
        assert_eq!(judge(&mut stepper, direction, overshoot), Outcome::Rejected);
    }
}

#[test]
fn a_rejected_step_is_cut_by_no_more_than_the_floor_factor() {
    let times = [0.0, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    judge(&mut stepper, 1e12, 0.0);
    assert!(stepper.propose().time >= Time::seconds(0.2 - 1e-12));
}

#[test]
fn only_the_breakpoints_are_reported_when_there_are_more_than_two() {
    let times = [0.0, 0.25, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    let mut reported = Vec::new();
    while !stepper.is_done() {
        let step = stepper.propose();
        if let Outcome::Accepted { report: true } = judge(&mut stepper, 0.0, 0.0) {
            reported.push(step.time);
        }
    }
    assert_eq!(reported, [0.25, 1.0].map(Time::seconds));
}

#[test]
fn every_accepted_step_is_reported_for_a_span() {
    let times = [0.0, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    judge(&mut stepper, 1.0, 0.0);
    assert_eq!(
        judge(&mut stepper, 0.0, 0.0),
        Outcome::Accepted { report: true }
    );
}

#[test]
fn a_step_that_could_not_be_solved_is_cut_back_until_the_floor() {
    let times = [0.0, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    let mut cuts = 0;
    loop {
        let step = stepper.propose();
        if !stepper.fail(&step) {
            break;
        }
        cuts += 1;
        assert!(cuts < 100);
    }
    assert!(cuts > 10);
}

#[test]
fn a_rejection_at_the_floor_is_an_error() {
    let times = [0.0, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    let mut result = Ok(Outcome::Rejected);
    for _ in 0..200 {
        let step = stepper.propose();
        result = stepper.judge(&step, error(1e12, 0.0), Dimensionless::of(TOLERANCE));
        if result.is_err() {
            break;
        }
    }
    assert!(result.is_err());
}

#[test]
fn a_short_step_onto_a_breakpoint_does_not_shorten_the_next() {
    let times = [0.0, 0.01, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    judge(&mut stepper, 0.0, 0.0);
    assert_eq!(stepper.propose().time, Time::seconds(1.0));
}

#[test]
fn the_maximum_takes_each_indicator_from_whichever_is_larger() {
    let a = StepError {
        direction: Dimensionless::of(3.0),
        overshoot: Dimensionless::of(1.0),
    };
    let b = StepError {
        direction: Dimensionless::of(2.0),
        overshoot: Dimensionless::of(5.0),
    };
    let expected = StepError {
        direction: Dimensionless::of(3.0),
        overshoot: Dimensionless::of(5.0),
    };
    assert_eq!(a.max(b), expected);
    assert_eq!(b.max(a), expected);
    assert_eq!(StepError::default().max(a), a);
}

fn quadratic(rate: f64) -> impl Fn(&f64, &f64, &(), &()) -> Result<StepError, String> {
    move |start, end, _, _| {
        Ok(StepError {
            direction: Dimensionless::of(rate * (end - start).powi(2)),
            overshoot: Dimensionless::of(0.0),
        })
    }
}

fn advance(time: Quantity<Time>, _: &f64, _: &()) -> Result<(f64, ()), String> {
    Ok((time.value(), ()))
}

fn reported_times(reported: &[(Quantity<Time>, f64, ())]) -> Vec<f64> {
    reported.iter().map(|(time, _, _)| time.value()).collect()
}

#[test]
fn a_step_within_the_tolerance_reports_the_span() {
    let times = [0.0, 2.0].map(Time::seconds);
    let reported = adapt(
        &times,
        Dimensionless::of(TOLERANCE),
        (0.0, ()),
        advance,
        quadratic(0.0),
        |message| message,
    )
    .unwrap();
    assert_eq!(reported_times(&reported), [0.0, 2.0]);
    assert_eq!(reported[1].1, 2.0);
}

#[test]
fn breakpoints_are_landed_on_and_are_all_that_is_reported() {
    let times = [0.0, 0.3, 1.0, 1.5].map(Time::seconds);
    let reported = adapt(
        &times,
        Dimensionless::of(TOLERANCE),
        (0.0, ()),
        advance,
        quadratic(0.0),
        |message| message,
    )
    .unwrap();
    assert_eq!(reported_times(&reported), [0.0, 0.3, 1.0, 1.5]);
}

#[test]
fn a_rejected_step_is_refined_until_every_accepted_step_is_within_the_tolerance() {
    let times = [0.0, 1.0].map(Time::seconds);
    let reported = adapt(
        &times,
        Dimensionless::of(TOLERANCE),
        (0.0, ()),
        advance,
        quadratic(1.0),
        |message| message,
    )
    .unwrap();
    let steps = reported_times(&reported)
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    assert!(steps.len() > 2);
    assert!(steps.iter().all(|step| step * step <= TOLERANCE));
    assert_eq!(*reported_times(&reported).last().unwrap(), 1.0);
}

#[test]
fn a_step_that_cannot_be_advanced_is_cut_back() {
    let times = [0.0, 1.0].map(Time::seconds);
    let reported = adapt(
        &times,
        Dimensionless::of(TOLERANCE),
        (0.0, ()),
        |time, start: &f64, state| {
            if time.value() - start > 0.3 {
                Err("too far".to_string())
            } else {
                advance(time, start, state)
            }
        },
        quadratic(0.0),
        |message| message,
    )
    .unwrap();
    assert!(reported.len() > 2);
    assert_eq!(*reported_times(&reported).last().unwrap(), 1.0);
}

#[test]
fn a_step_that_can_never_be_advanced_returns_its_error() {
    let times = [0.0, 1.0].map(Time::seconds);
    let result = adapt(
        &times,
        Dimensionless::of(TOLERANCE),
        (0.0, ()),
        |_, _: &f64, _: &()| Err::<(f64, ()), _>("never".to_string()),
        quadratic(0.0),
        |message| message,
    );
    assert_eq!(result.err().unwrap(), "never");
}

#[test]
fn times_that_cannot_be_stepped_are_a_message_in_the_error_of_the_caller() {
    let times = [1.0, 0.0].map(Time::seconds);
    let result = adapt(
        &times,
        Dimensionless::of(TOLERANCE),
        (0.0, ()),
        advance,
        quadratic(0.0),
        |message| format!("failed: {message}"),
    );
    assert!(result.err().unwrap().starts_with("failed: "));
}

fn metric(direction: f64, overshoot: f64) -> StepError<Length> {
    StepError {
        direction: Length::meters(direction),
        overshoot: Length::meters(overshoot),
    }
}

fn next_step<I: StepIndicators>(error: I, tolerance: I::Tolerance) -> (Outcome, Option<f64>) {
    let times = [0.0, 1.0].map(Time::seconds);
    let mut stepper = Stepper::new(&times).unwrap();
    let step = stepper.propose();
    let outcome = stepper.judge(&step, error, tolerance).unwrap();
    (
        outcome,
        (!stepper.is_done()).then(|| stepper.propose().time.value()),
    )
}

#[test]
fn an_error_in_another_unit_is_held_to_a_tolerance_in_it() {
    let tolerance = Length::meters(TOLERANCE);
    assert_eq!(
        next_step(metric(0.1 * TOLERANCE, 0.0), tolerance).0,
        Outcome::Accepted { report: true }
    );
    assert_eq!(
        next_step(metric(10.0 * TOLERANCE, 0.0), tolerance).0,
        Outcome::Rejected
    );
    assert_eq!(
        next_step(metric(0.0, 10.0 * TOLERANCE), tolerance).0,
        Outcome::Rejected
    );
}

#[test]
fn every_source_must_be_within_its_own_tolerance() {
    let tolerance = (Dimensionless::of(TOLERANCE), Length::meters(TOLERANCE));
    let (small, large) = (0.1 * TOLERANCE, 10.0 * TOLERANCE);
    assert_eq!(
        next_step((error(small, 0.0), metric(small, 0.0)), tolerance).0,
        Outcome::Accepted { report: true }
    );
    assert_eq!(
        next_step((error(large, 0.0), metric(small, 0.0)), tolerance).0,
        Outcome::Rejected
    );
    assert_eq!(
        next_step((error(small, 0.0), metric(0.0, large)), tolerance).0,
        Outcome::Rejected
    );
}

#[test]
fn the_next_step_follows_the_most_restrictive_source() {
    let tolerance = (Dimensionless::of(TOLERANCE), Length::meters(TOLERANCE));
    let (mild, severe) = (2.0 * TOLERANCE, 50.0 * TOLERANCE);
    let by_mild = next_step(error(mild, 0.0), Dimensionless::of(TOLERANCE))
        .1
        .unwrap();
    let by_severe = next_step(metric(severe, 0.0), Length::meters(TOLERANCE))
        .1
        .unwrap();
    assert!(by_severe < by_mild);
    let (_, together) = next_step((error(mild, 0.0), metric(severe, 0.0)), tolerance);
    let together = together.unwrap();
    assert_eq!(together, by_severe);
    let (_, swapped) = next_step(
        (metric(severe, 0.0), error(mild, 0.0)),
        (tolerance.1, tolerance.0),
    );
    assert_eq!(swapped.unwrap(), by_severe);
}

#[test]
fn a_source_without_indicators_does_not_hold_the_step() {
    assert_eq!(next_step((), ()).0, Outcome::Accepted { report: true });
    let (outcome, size) = next_step(
        ((), error(2.0 * TOLERANCE, 0.0)),
        ((), Dimensionless::of(TOLERANCE)),
    );
    let (alone, alone_size) = next_step(error(2.0 * TOLERANCE, 0.0), Dimensionless::of(TOLERANCE));
    assert_eq!(outcome, alone);
    assert_eq!(size, alone_size);
}

#[test]
fn each_source_of_an_adaptive_loop_refines_by_its_own_tolerance() {
    let times = [0.0, 1.0].map(Time::seconds);
    let count = |rates: (f64, f64)| {
        adapt(
            &times,
            (Dimensionless::of(TOLERANCE), Length::meters(TOLERANCE)),
            (0.0, ()),
            advance,
            |start: &f64, end: &f64, _: &(), _: &()| {
                let step = (end - start).powi(2);
                Ok::<_, String>((error(rates.0 * step, 0.0), metric(rates.1 * step, 0.0)))
            },
            |message| message,
        )
        .unwrap()
        .len()
    };
    let (none, first, second) = (count((0.0, 0.0)), count((1.0, 0.0)), count((0.0, 1.0)));
    assert_eq!(none, 2);
    assert!(first > none && second > none);
    assert_eq!(first, second);
    assert!(count((1.0, 1.0)) >= first);
}

#[test]
fn the_indicators_in_units_of_a_scale_are_dimensionless() {
    let relative = metric(3.0, 5.0).relative_to(Length::meters(2.0));
    assert_eq!(relative, error(1.5, 2.5));
    assert_eq!(
        StepError::<Length>::default().relative_to(Length::meters(2.0)),
        error(0.0, 0.0)
    );
}
