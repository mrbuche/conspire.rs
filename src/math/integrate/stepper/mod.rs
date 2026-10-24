#[cfg(test)]
mod test;

use super::StepController;
use crate::{
    math::{Quantity, Scalar},
    units::{Dimensionless, Time},
};
use std::fmt::{self, Debug, Formatter};

const DT_CUT: Scalar = 0.2;

/// The error indicators of one step of a load parameter.
///
/// A step is resolved by how much the direction of the response changes over it, and by
/// whether it runs past a kink in the response. Both vanish when the response is exact for
/// the step, as for a constant direction. A solver measures them against the reversible
/// range of the response, with [`relative_to`](Self::relative_to), before comparing them
/// to a tolerance.
pub struct StepError<U = Dimensionless> {
    /// The estimated error of the step from the change of the direction of the response
    /// over it, which falls as the square of the step.
    pub direction: Quantity<U>,
    /// How far the step runs past a kink in the response that it begins before, and zero
    /// otherwise, which falls as the step.
    pub overshoot: Quantity<U>,
}

impl<U> Clone for StepError<U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<U> Copy for StepError<U> {}

impl<U> Debug for StepError<U> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter
            .debug_struct("StepError")
            .field("direction", &self.direction)
            .field("overshoot", &self.overshoot)
            .finish()
    }
}

impl<U> Default for StepError<U> {
    fn default() -> Self {
        Self {
            direction: Quantity::default(),
            overshoot: Quantity::default(),
        }
    }
}

impl<U> PartialEq for StepError<U> {
    fn eq(&self, other: &Self) -> bool {
        self.direction == other.direction && self.overshoot == other.overshoot
    }
}

impl<U> StepError<U> {
    /// The indicators in units of a scale of the same unit, such as the reversible range of
    /// the response, which makes them dimensionless.
    pub fn relative_to(self, scale: Quantity<U>) -> StepError {
        StepError {
            direction: self.direction.ratio(scale),
            overshoot: self.overshoot.ratio(scale),
        }
    }
    /// The larger of each of the two, as for the points of a mesh.
    pub fn max(self, other: Self) -> Self {
        Self {
            direction: self.direction.max(other.direction),
            overshoot: self.overshoot.max(other.overshoot),
        }
    }
}

/// The error indicators of the sources of a step, each held to a tolerance in its own unit.
///
/// A tuple of indicators holds a step to the tolerances of all of them, and sizes the next
/// step by the most restrictive. Indicators of different units are never combined, only
/// compared to their own tolerance.
pub trait StepIndicators {
    /// The tolerances the indicators are held to, of the same structure.
    type Tolerance: Copy;
    /// Whether every indicator is within its tolerance.
    fn within(&self, tolerance: &Self::Tolerance) -> bool;
    /// The factor to multiply the step by, the smallest over the indicators.
    fn scale(
        &self,
        tolerance: &Self::Tolerance,
        direction: &StepController,
        overshoot: &StepController,
    ) -> Quantity;
}

impl<U> StepIndicators for StepError<U> {
    type Tolerance = Quantity<U>;
    fn within(&self, tolerance: &Quantity<U>) -> bool {
        self.direction <= *tolerance && self.overshoot <= *tolerance
    }
    fn scale(
        &self,
        tolerance: &Quantity<U>,
        direction: &StepController,
        overshoot: &StepController,
    ) -> Quantity {
        direction
            .scale(self.direction, *tolerance)
            .min(overshoot.scale(self.overshoot, *tolerance))
    }
}

impl StepIndicators for () {
    type Tolerance = ();
    fn within(&self, _: &()) -> bool {
        true
    }
    fn scale(&self, _: &(), direction: &StepController, _: &StepController) -> Quantity {
        direction.grow
    }
}

impl<A, B> StepIndicators for (A, B)
where
    A: StepIndicators,
    B: StepIndicators,
{
    type Tolerance = (A::Tolerance, B::Tolerance);
    fn within(&self, tolerance: &Self::Tolerance) -> bool {
        self.0.within(&tolerance.0) && self.1.within(&tolerance.1)
    }
    fn scale(
        &self,
        tolerance: &Self::Tolerance,
        direction: &StepController,
        overshoot: &StepController,
    ) -> Quantity {
        self.0
            .scale(&tolerance.0, direction, overshoot)
            .min(self.1.scale(&tolerance.1, direction, overshoot))
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
    pub fn judge<I>(
        &mut self,
        step: &Step,
        error: I,
        tolerance: I::Tolerance,
    ) -> Result<Outcome, String>
    where
        I: StepIndicators,
    {
        let scale = error.scale(&tolerance, &self.direction, &self.overshoot);
        if error.within(&tolerance) {
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
/// gives the [`StepIndicators`] of the step from the states at its two ends. A step that
/// `advance` fails on is cut back like a rejected one, and `message` makes the error
/// of a step controller that cannot continue.
pub fn adapt<X, S, E, I>(
    times: &[Quantity<Time>],
    tolerance: I::Tolerance,
    initial: (X, S),
    mut advance: impl FnMut(Quantity<Time>, &X, &S) -> Result<(X, S), E>,
    error: impl Fn(&X, &X, &S, &S) -> Result<I, E>,
    message: impl Fn(String) -> E,
) -> Result<Vec<(Quantity<Time>, X, S)>, E>
where
    X: Clone,
    S: Clone,
    I: StepIndicators,
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
            .judge(&step, step_error, tolerance)
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
