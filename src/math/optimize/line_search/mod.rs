use super::{super::Scalar, OptimizationError};
use crate::{
    defeat_message,
    math::{Jacobian, Solution},
};
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::Mul,
};

/// Available line search algorithms.
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
    Goldstein {
        control: Scalar,
        cut_back: Scalar,
        max_steps: usize,
    },
    /// The Wolfe conditions.
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
            Self::Armijo { .. } => write!(f, "Armijo {{..}}"),
            Self::Error { .. } => write!(f, "Error {{..}}"),
            Self::Goldstein { .. } => write!(f, "Goldstein {{..}}"),
            Self::Wolfe { .. } => write!(f, "Wolfe {{..}}"),
            Self::None { .. } => write!(f, "None"),
        }
    }
}

impl LineSearch {
    pub fn backtrack<X, J>(
        &self,
        function: impl Fn(&X) -> Result<Scalar, OptimizationError>,
        jacobian: impl Fn(&X) -> Result<J, OptimizationError>,
        argument: &X,
        jacobian0: &J,
        decrement: &X,
        step_size: Scalar,
    ) -> Result<Scalar, LineSearchError>
    where
        J: Jacobian,
        for<'a> &'a J: From<&'a X>,
        X: Solution,
        for<'a> &'a X: Mul<Scalar, Output = X>,
    {
        if step_size <= 0.0 {
            return Err(LineSearchError::NegativeStepSize(self.clone(), step_size));
        }
        let mut n = step_size;
        let f = if let Ok(value) = function(argument) {
            value
        } else {
            return Err(LineSearchError::InvalidStartingPoint(self.clone()));
        };
        let m = jacobian0.full_contraction(decrement.into());
        if m <= 0.0 {
            return Err(LineSearchError::NotDescentDirection(self.clone()));
        }
        match self {
            Self::Armijo {
                control,
                cut_back,
                max_steps,
            } => {
                let mut f_n;
                let t = control * m;
                for _ in 0..*max_steps {
                    f_n = function(&(decrement * -n + argument));
                    if let Ok(value) = f_n
                        && f - value >= n * t
                    {
                        return Ok(n);
                    } else {
                        n *= cut_back
                    }
                }
                Err(LineSearchError::MaximumStepsReached(
                    self.clone(),
                    *max_steps,
                ))
            }
            Self::Error {
                cut_back,
                max_steps,
            } => {
                for _ in 0..*max_steps {
                    if function(&(decrement * -n + argument)).is_ok() {
                        return Ok(n);
                    } else {
                        n *= cut_back
                    }
                }
                Err(LineSearchError::MaximumStepsReached(
                    self.clone(),
                    *max_steps,
                ))
            }
            Self::Goldstein {
                control,
                cut_back,
                max_steps,
            } => {
                let mut f_n;
                let t = control * m;
                let u = (1.0 - control) * m;
                let mut v;
                for _ in 0..*max_steps {
                    f_n = function(&(decrement * -n + argument));
                    if let Ok(value) = f_n {
                        v = f - value;
                        if n * u < v || v < n * t {
                            n *= cut_back
                        } else {
                            return Ok(n);
                        }
                    } else {
                        n *= cut_back
                    }
                }
                Err(LineSearchError::MaximumStepsReached(
                    self.clone(),
                    *max_steps,
                ))
            }
            Self::Wolfe {
                control_1,
                control_2,
                cut_back,
                max_steps,
                strong,
            } => {
                let mut f_n;
                let mut j_n;
                let t_1 = control_1 * m;
                let t_2 = control_2 * m;
                let mut trial_argument = decrement * -n + argument;
                for _ in 0..*max_steps {
                    f_n = function(&trial_argument);
                    j_n = jacobian(&trial_argument);
                    if let Ok(f_val) = f_n
                        && let Ok(j_val) = j_n
                        && f - f_val >= n * t_1
                        && if *strong {
                            j_val.full_contraction(decrement.into()) < t_2
                        } else {
                            j_val.full_contraction(decrement.into()).abs() < t_2.abs() // less than?
                        }
                    {
                        return Ok(n);
                    } else {
                        n *= cut_back;
                        trial_argument = decrement * -n + argument
                    }
                }
                Err(LineSearchError::MaximumStepsReached(
                    self.clone(),
                    *max_steps,
                ))
            }
            Self::None => {
                panic!("Cannot call backtracking line search when there is no algorithm.")
            }
        }
    }
}

/// Possible errors encountered during line search.
pub enum LineSearchError {
    InvalidStartingPoint(LineSearch),
    MaximumStepsReached(LineSearch, usize),
    NegativeStepSize(LineSearch, Scalar),
    NotDescentDirection(LineSearch),
}

impl Debug for LineSearchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InvalidStartingPoint(line_search) => {
                format!(
                    "\x1b[1;91mStaring point is invalid.\x1b[0;91m\n\
                     In line search: {line_search:?}."
                )
            }
            Self::MaximumStepsReached(line_search, steps) => {
                format!(
                    "\x1b[1;91mMaximum number of steps ({steps}) reached.\x1b[0;91m\n\
                     In line search: {line_search:?}."
                )
            }
            Self::NegativeStepSize(line_search, step_size) => {
                format!(
                    "\x1b[1;91mNegative step size ({step_size}) encountered.\x1b[0;91m\n\
                     In line search: {line_search:?}."
                )
            }
            Self::NotDescentDirection(line_search) => {
                format!(
                    "\x1b[1;91mDirection is not a descent direction.\x1b[0;91m\n\
                     In line search: {line_search:?}."
                )
            }
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for LineSearchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InvalidStartingPoint(line_search) => {
                format!(
                    "\x1b[1;91mStaring point is invalid.\x1b[0;91m\n\
                     In line search: {line_search:?}."
                )
            }
            Self::MaximumStepsReached(line_search, steps) => {
                format!(
                    "\x1b[1;91mMaximum number of steps ({steps}) reached.\x1b[0;91m\n\
                     In line search: {line_search:?}."
                )
            }
            Self::NegativeStepSize(line_search, step_size) => {
                format!(
                    "\x1b[1;91mNegative step size ({step_size}) encountered.\x1b[0;91m\n\
                     In line search: {line_search:?}."
                )
            }
            Self::NotDescentDirection(line_search) => {
                format!(
                    "\x1b[1;91mDirection is not a descent direction.\x1b[0;91m\n\
                     In line search: {line_search:?}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}
