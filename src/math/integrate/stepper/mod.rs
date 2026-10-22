#[cfg(test)]
mod test;

use super::StepController;
use crate::{
    math::{Quantity, Scalar},
    units::Time,
};

const DT_CUT: Scalar = 0.2;

/// The error indicators of one step of a load parameter.
///
/// A step is resolved by how much the direction of the response changes over it, and by
/// whether it runs past a kink in the response. Both vanish when the response is exact for
/// the step, as for a constant direction, and both are dimensionless.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepError {
    /// The estimated error of the step from the change of the direction of the response
    /// over it, which falls as the square of the step.
    pub direction: Quantity,
    /// How far the step runs past a kink in the response that it begins before, and zero
    /// otherwise, which falls as the step.
    pub overshoot: Quantity,
}

impl StepError {
    /// The larger of each of the two, as for the points of a mesh.
    pub fn max(self, other: Self) -> Self {
        Self {
            direction: self.direction.max(other.direction),
            overshoot: self.overshoot.max(other.overshoot),
        }
    }
}

/// The steps of a load parameter, chosen by the [`StepError`] of each.
///
/// Two times give only the span, and every accepted step is reported. More than two are
/// breakpoints: the steps are cut to end on each, and only they are reported, since the
/// response has kinks at yielding and at unloading that an interpolant between steps would
/// smooth over.
pub struct Stepper<'a> {
    times: &'a [Quantity<Time>],
    time: Quantity<Time>,
    size: Quantity<Time>,
    floor: Quantity<Time>,
    breakpoint: usize,
    direction: StepController,
    overshoot: StepController,
}

/// A step proposed by a [`Stepper`].
#[derive(Clone, Copy, Debug)]
pub struct Step {
    /// The time the step ends at.
    pub time: Quantity<Time>,
    size: Quantity<Time>,
    landing: bool,
}

/// What a [`Stepper`] made of a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The step was taken, and is to be reported if `report`.
    Accepted { report: bool },
    /// The step was too large, and the next proposed is smaller.
    Rejected,
}

impl<'a> Stepper<'a> {
    pub fn new(times: &'a [Quantity<Time>]) -> Result<Self, String> {
        match (times.first(), times.last()) {
            (Some(&start), Some(&end)) if times.len() >= 2 && start < end => Ok(Self {
                times,
                time: start,
                size: end - start,
                floor: (end - start) * 1e-10,
                breakpoint: 1,
                direction: StepController::new(0.9, 2.0, DT_CUT, 5.0),
                overshoot: StepController::new(0.9, 1.0, DT_CUT, 5.0),
            }),
            _ => Err("The times must be at least two and increasing.".to_string()),
        }
    }
    /// Whether the last breakpoint has been reached.
    pub fn is_done(&self) -> bool {
        self.breakpoint >= self.times.len()
    }
    /// The step to try next.
    pub fn propose(&self) -> Step {
        let remaining = self.times[self.breakpoint] - self.time;
        let landing = self.size >= remaining;
        let size = if landing { remaining } else { self.size };
        Step {
            time: if landing {
                self.times[self.breakpoint]
            } else {
                self.time + size
            },
            size,
            landing,
        }
    }
    /// Takes the step if its error is within the tolerance, and sizes the next one by it.
    pub fn judge(
        &mut self,
        step: &Step,
        error: StepError,
        abs_tol: Quantity,
    ) -> Result<Outcome, String> {
        let scale = self
            .direction
            .scale(error.direction, abs_tol)
            .min(self.overshoot.scale(error.overshoot, abs_tol));
        if error.direction <= abs_tol && error.overshoot <= abs_tol {
            self.time = step.time;
            if step.landing {
                self.breakpoint += 1;
            }
            self.size = if step.landing {
                self.size.max(step.size * scale)
            } else {
                step.size * scale
            };
            Ok(Outcome::Accepted {
                report: step.landing || self.times.len() == 2,
            })
        } else if step.size <= self.floor {
            Err("The adaptive step fell below the floor.".to_string())
        } else {
            self.size = step.size * scale;
            Ok(Outcome::Rejected)
        }
    }
    /// Cuts back after a step that could not be solved, returning whether it could be.
    pub fn fail(&mut self, step: &Step) -> bool {
        if step.size <= self.floor {
            return false;
        }
        self.size = step.size * DT_CUT;
        true
    }
}

/// Steps a load parameter adaptively through `times`, and returns the accepted steps that
/// are reported, beginning with `initial` at the first time.
///
/// `advance` takes the state at a step's start to its end at the given time, and `error`
/// gives the [`StepError`] of the step from the states at its two ends. A step that
/// `advance` fails on is cut back like a rejected one, and `message` makes the error
/// of a step controller that cannot continue.
pub fn adapt<X, S, E>(
    times: &[Quantity<Time>],
    abs_tol: Quantity,
    initial: (X, S),
    mut advance: impl FnMut(Quantity<Time>, &X, &S) -> Result<(X, S), E>,
    error: impl Fn(&X, &X, &S, &S) -> Result<StepError, E>,
    message: impl Fn(String) -> E,
) -> Result<Vec<(Quantity<Time>, X, S)>, E>
where
    X: Clone,
    S: Clone,
{
    let mut stepper = Stepper::new(times).map_err(&message)?;
    let (mut current, mut state) = initial;
    let mut reported = vec![(times[0], current.clone(), state.clone())];
    while !stepper.is_done() {
        let step = stepper.propose();
        let (next, state_next) = match advance(step.time, &current, &state) {
            Ok(next) => next,
            Err(_) if stepper.fail(&step) => continue,
            Err(error) => return Err(error),
        };
        let step_error = error(&current, &next, &state, &state_next)?;
        match stepper
            .judge(&step, step_error, abs_tol)
            .map_err(&message)?
        {
            Outcome::Accepted { report } => {
                current = next;
                state = state_next;
                if report {
                    reported.push((step.time, current.clone(), state.clone()));
                }
            }
            Outcome::Rejected => {}
        }
    }
    Ok(reported)
}
