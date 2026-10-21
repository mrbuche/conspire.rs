use super::{Outcome, Stepper};
use crate::{
    constitutive::solid::elastic_plastic::StepError,
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
