#[cfg(test)]
mod test;

use crate::math::{
    Erase, Jacobian, Quantity, Scalar, Solution, Style, StyledError, Tensor, styled_error,
};
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::Mul,
};

/// Available line search algorithms.
///
/// The conditions come in two kinds. A sufficient decrease is met by every step
/// short enough, so `cut_back` shortens the step offered until one is, and the
/// step returned is never longer than the one offered. Pairing that decrease
/// with a condition only a long enough step can meet accepts an interval
/// instead, which the step offered may fall below; there `cut_back` sets how
/// fast the step is grown to bracket that interval, and the step returned may
/// be longer than the one offered.
#[derive(Clone, Debug)]
pub enum LineSearch {
    /// The Armijo condition.
    Armijo {
        control: Scalar,
        cut_back: Scalar,
        max_steps: usize,
    },
    /// Backtrack for errors.
    Error { cut_back: Scalar, max_steps: usize },
    /// The Goldstein conditions.
    ///
    /// Accepts an interval, so the step offered may be grown.
    Goldstein {
        control: Scalar,
        cut_back: Scalar,
        max_steps: usize,
    },
    /// The Wolfe conditions.
    ///
    /// Accepts an interval, so the step offered may be grown.
    Wolfe {
        control_1: Scalar,
        control_2: Scalar,
        cut_back: Scalar,
        max_steps: usize,
        strong: bool,
    },
    /// No line search.
    None,
}

impl Default for LineSearch {
    fn default() -> Self {
        Self::Armijo {
            control: 1e-3,
            cut_back: 9e-1,
            max_steps: 100,
        }
    }
}

impl Display for LineSearch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Armijo { .. } => write!(f, "Armijo"),
            Self::Error { .. } => write!(f, "Error"),
            Self::Goldstein { .. } => write!(f, "Goldstein"),
            Self::Wolfe { .. } => write!(f, "Wolfe"),
            Self::None { .. } => write!(f, "None"),
        }
    }
}

impl LineSearch {
    /// Search a merit function of the step size alone.
    ///
    /// The exact penalty function is not differentiable, its norm having a kink
    /// wherever a constraint is satisfied, so its slope along the step is
    /// supplied rather than recovered from a gradient.
    pub fn search_merit(
        &self,
        mut merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        slope: Scalar,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        if step_size <= 0.0 {
            return Err(LineSearchError::NegativeStepSize(
                format!("{self:?}"),
                step_size,
            ));
        } else if slope <= 0.0 {
            return Err(LineSearchError::NotDescentDirection(format!("{self:?}")));
        }
        let mut n = step_size;
        match self {
            Self::Armijo {
                control,
                cut_back,
                max_steps,
            } => {
                let t = control * slope;
                for _ in 0..*max_steps {
                    if let Ok(trial) = merit(n)
                        && value - trial >= n * t
                    {
                        return Ok(n);
                    } else {
                        n *= cut_back
                    }
                }
                Err(LineSearchError::MaximumStepsReached(
                    format!("{self:?}"),
                    *max_steps,
                ))
            }
            Self::Error {
                cut_back,
                max_steps,
            } => {
                for _ in 0..*max_steps {
                    if merit(n).is_ok() {
                        return Ok(n);
                    } else {
                        n *= cut_back
                    }
                }
                Err(LineSearchError::MaximumStepsReached(
                    format!("{self:?}"),
                    *max_steps,
                ))
            }
            Self::Goldstein {
                control,
                cut_back,
                max_steps,
            } => {
                let t = control * slope;
                let u = (1.0 - control) * slope;
                bracket_and_zoom(
                    self,
                    |n| match merit(n) {
                        Ok(trial) => judge_goldstein(value - trial, n, t, u),
                        Err(_) => Trial::Long,
                    },
                    n,
                    cut_back.recip(),
                    *max_steps,
                )
            }
            Self::Wolfe { .. } => panic!(
                "The Wolfe conditions need the gradient of the merit function, which the exact penalty function does not have."
            ),
            Self::None => {
                panic!("Cannot call the line search when there is no algorithm.")
            }
        }
    }
    pub fn search<X, J, D, W, E>(
        &self,
        mut function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        mut jacobian: impl FnMut(&X) -> Result<J, String>,
        argument: &X,
        jacobian0: &J,
        decrement: &D,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E>,
        E: Tensor,
        X: Solution,
        for<'a> &'a D: Mul<Quantity<W>, Output = X>,
    {
        if step_size <= 0.0 {
            return Err(LineSearchError::NegativeStepSize(
                format!("{self:?}"),
                step_size,
            ));
        }
        let mut n = step_size;
        let f = if let Ok(value) = function(argument, 0.0) {
            value
        } else {
            return Err(LineSearchError::InvalidStartingPoint(format!("{self:?}")));
        };
        let m = jacobian0.erase().full_contraction(decrement.erase());
        if m <= 0.0 {
            return Err(LineSearchError::NotDescentDirection(format!("{self:?}")));
        }
        let trial = |n: Scalar| decrement * Quantity::new(-n) + argument;
        match self {
            Self::Armijo {
                control,
                cut_back,
                max_steps,
            } => {
                let mut f_n;
                let t = control * m;
                for _ in 0..*max_steps {
                    f_n = function(&trial(n), n);
                    if let Ok(value) = f_n
                        && f - value >= n * t
                    {
                        return Ok(n);
                    } else {
                        n *= cut_back
                    }
                }
                Err(LineSearchError::MaximumStepsReached(
                    format!("{self:?}"),
                    *max_steps,
                ))
            }
            Self::Error {
                cut_back,
                max_steps,
            } => {
                for _ in 0..*max_steps {
                    if function(&trial(n), n).is_ok() {
                        return Ok(n);
                    } else {
                        n *= cut_back
                    }
                }
                Err(LineSearchError::MaximumStepsReached(
                    format!("{self:?}"),
                    *max_steps,
                ))
            }
            Self::Goldstein {
                control,
                cut_back,
                max_steps,
            } => {
                let t = control * m;
                let u = (1.0 - control) * m;
                bracket_and_zoom(
                    self,
                    |n| match function(&trial(n), n) {
                        Ok(value) => judge_goldstein(f - value, n, t, u),
                        Err(_) => Trial::Long,
                    },
                    n,
                    cut_back.recip(),
                    *max_steps,
                )
            }
            Self::Wolfe {
                control_1,
                control_2,
                cut_back,
                max_steps,
                strong,
            } => {
                let t_1 = control_1 * m;
                let t_2 = control_2 * m;
                bracket_and_zoom(
                    self,
                    |n| {
                        let trial_argument = trial(n);
                        match (function(&trial_argument, n), jacobian(&trial_argument)) {
                            (Ok(value), Ok(j_n)) => judge_wolfe(
                                f - value,
                                j_n.erase().full_contraction(decrement.erase()),
                                n,
                                t_1,
                                t_2,
                                *strong,
                            ),
                            _ => Trial::Long,
                        }
                    },
                    n,
                    cut_back.recip(),
                    *max_steps,
                )
            }
            Self::None => {
                panic!("Cannot call the line search when there is no algorithm.")
            }
        }
    }
}

/// How a trial step sits against the steps a condition would accept.
///
/// A sufficient decrease alone is met by every step short enough, so missing it
/// can only mean the step was too long. Pairing it with a condition that only a
/// long enough step can meet leaves an interval instead, which a trial can miss
/// from either side, and the side it missed from is what says where to look next.
enum Trial {
    Accept,
    Long,
    Short,
}

/// Whether the decrease is enough to keep, and enough of the step was taken.
///
/// Too little decrease for the step means the step outran the descent it was
/// promised; too much of it means the step stopped short of where that descent
/// was still being collected.
fn judge_goldstein(decrease: Scalar, n: Scalar, t: Scalar, u: Scalar) -> Trial {
    if decrease < n * t {
        Trial::Long
    } else if decrease > n * u {
        Trial::Short
    } else {
        Trial::Accept
    }
}

/// Whether the decrease is enough to keep, and the slope has flattened enough.
///
/// A slope still steeply downhill says the step stopped short of the descent on
/// offer. The strong form also refuses a slope that has turned as steeply
/// uphill, which is the step having climbed back out the far side of a minimum
/// it passed, and so a step too long rather than one too short.
fn judge_wolfe(
    decrease: Scalar,
    slope: Scalar,
    n: Scalar,
    t_1: Scalar,
    t_2: Scalar,
    strong: bool,
) -> Trial {
    if decrease < n * t_1 || (strong && slope < -t_2) {
        Trial::Long
    } else if slope > t_2 {
        Trial::Short
    } else {
        Trial::Accept
    }
}

/// Brackets a step the condition accepts, then bisects the bracket onto one.
///
/// The step offered is grown until one too long turns up to close the bracket
/// against the longest one still too short, since a condition met only above
/// some length cannot be reached by shortening. Between the two ends lies a step
/// that is neither, and bisection is what finds it.
///
/// A trial that cannot be evaluated counts as too long. There is nothing to
/// compare it against, and less of the step is the safer of the two guesses.
fn bracket_and_zoom(
    line_search: &LineSearch,
    mut judge: impl FnMut(Scalar) -> Trial,
    step_size: Scalar,
    growth: Scalar,
    max_steps: usize,
) -> Result<Scalar, LineSearchError> {
    let exhausted = || LineSearchError::MaximumStepsReached(format!("{line_search:?}"), max_steps);
    let mut long = None;
    let mut n = step_size;
    let mut short = 0.0;
    let mut steps = 0;
    while steps < max_steps {
        steps += 1;
        match judge(n) {
            Trial::Accept => return Ok(n),
            Trial::Long => {
                long = Some(n);
                break;
            }
            Trial::Short => {
                short = n;
                n *= growth
            }
        }
    }
    let mut long = long.ok_or_else(exhausted)?;
    while steps < max_steps {
        steps += 1;
        n = (short + long) / 2.0;
        match judge(n) {
            Trial::Accept => return Ok(n),
            Trial::Long => long = n,
            Trial::Short => short = n,
        }
    }
    Err(exhausted())
}

/// Possible errors encountered during line search.
pub enum LineSearchError {
    InvalidStartingPoint(String),
    MaximumStepsReached(String, usize),
    NegativeStepSize(String, Scalar),
    NotDescentDirection(String),
}

impl StyledError for LineSearchError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::InvalidStartingPoint(line_search) => format!(
                "{h}Starting point is invalid.{c}\n\
                In line search: {line_search}."
            ),
            Self::MaximumStepsReached(line_search, steps) => format!(
                "{h}Maximum number of steps ({steps}) reached.{c}\n\
                In line search: {line_search}."
            ),
            Self::NegativeStepSize(line_search, step_size) => format!(
                "{h}Negative step size ({step_size}) encountered.{c}\n\
                In line search: {line_search}."
            ),
            Self::NotDescentDirection(line_search) => format!(
                "{h}Direction is not a descent direction.{c}\n\
                In line search: {line_search}."
            ),
        }
    }
}

styled_error!(LineSearchError);
