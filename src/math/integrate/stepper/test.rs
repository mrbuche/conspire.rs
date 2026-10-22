use super::{Outcome, StepError, Stepper, adapt};
use crate::{
    math::Quantity,
    units::{Dimensionless, Time},
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
