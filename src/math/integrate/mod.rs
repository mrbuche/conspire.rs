#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
mod test;

mod backward_euler;
mod bogacki_shampine;
mod dormand_prince;
mod verner_8;
mod verner_9;

pub use backward_euler::BackwardEuler;
pub use bogacki_shampine::BogackiShampine;
pub use dormand_prince::DormandPrince;
pub use verner_8::Verner8;
pub use verner_9::Verner9;

/// Alias for [`BackwardEuler`].
pub type Ode1be = BackwardEuler;

/// Alias for [`BogackiShampine`].
pub type Ode23 = BogackiShampine;

/// Alias for [`DormandPrince`].
pub type Ode45 = DormandPrince;

/// Alias for [`Verner8`].
pub type Ode78 = Verner8;

/// Alias for [`Verner9`].
pub type Ode89 = Verner9;

use super::{
    Scalar, Solution, Tensor, TensorArray, TensorVec, TestError, Vector, assert_eq_within_tols,
    interpolate::{InterpolateSolution, InterpolateSolutionIV},
    optimize::{FirstOrderRootFinding, ZerothOrderRootFinding},
};
use crate::defeat_message;
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Div, Mul, Sub},
};

/// Required methods for ordinary differential equation solvers.
pub trait OdeSolver<Y, U>
where
    Self: Debug,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    /// Returns the absolute error tolerance.
    fn abs_tol(&self) -> Scalar;
}

/// Required methods for variable-step ordinary differential equation solvers.
pub trait VariableStep
// where
//     Self: OdeSolver<Y, U>,
//     Y: Tensor,
//     U: TensorVec<Item = Y>,
{
    /// Returns the cut back factor for function errors.
    fn dt_cut(&self) -> Scalar;
    /// Returns the minimum value for the time step.
    fn dt_min(&self) -> Scalar;
}

/// Required methods for explicit ordinary differential equation solvers.
pub trait Explicit<Y, U>
where
    Self: InterpolateSolution<Y, U> + OdeSolver<Y, U> + VariableStep,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize;
    /// Returns the multiplier for adaptive time steps.
    fn dt_beta(&self) -> Scalar;
    /// Returns the exponent for adaptive time steps.
    fn dt_expn(&self) -> Scalar;
    #[doc = include_str!("explicit.md")]
    fn integrate(
        &self,
        mut function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        time: &[Scalar],
        initial_condition: Y,
    ) -> Result<(Vector, U, U), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = t_0;
        let mut dt = t_f - t_0;
        let mut k = vec![Y::default(); Self::SLOPES];
        k[0] = function(t, &initial_condition)?;
        let mut t_sol = Vector::new();
        t_sol.push(t_0);
        let mut y = initial_condition.clone();
        let mut y_sol = U::new();
        y_sol.push(initial_condition.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(k[0].clone());
        let mut y_trial = Y::default();
        while t < t_f {
            match self.slopes(&mut function, &y, t, dt, &mut k, &mut y_trial) {
                Ok(e) => {
                    if let Some(error) = self
                        .step(
                            &mut function,
                            &mut y,
                            &mut t,
                            &mut y_sol,
                            &mut t_sol,
                            &mut dydt_sol,
                            &mut dt,
                            &mut k,
                            &y_trial,
                            e,
                        )
                        .err()
                    {
                        dt *= self.dt_cut();
                        if dt < self.dt_min() {
                            return Err(IntegrationError::MinimumStepSizeUpstream(
                                self.dt_min(),
                                error,
                                format!("{self:?}"),
                            ));
                        }
                    } else {
                        dt = dt.min(t_f - t);
                        if dt < self.dt_min() && t < t_f {
                            return Err(IntegrationError::MinimumStepSizeReached(
                                self.dt_min(),
                                format!("{self:?}"),
                            ));
                        }
                    }
                }
                Err(error) => {
                    dt *= self.dt_cut();
                    if dt < self.dt_min() {
                        return Err(IntegrationError::MinimumStepSizeUpstream(
                            self.dt_min(),
                            error,
                            format!("{self:?}"),
                        ));
                    }
                }
            }
        }
        if time.len() > 2 {
            let t_int = Vector::from(time);
            let (y_int, dydt_int) = self.interpolate(&t_int, &t_sol, &y_sol, function)?;
            Ok((t_int, y_int, dydt_int))
        } else {
            Ok((t_sol, y_sol, dydt_sol))
        }
    }
    #[doc(hidden)]
    fn slopes(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &Y,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
    ) -> Result<Scalar, String>;
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn step(
        &self,
        function: impl FnMut(Scalar, &Y) -> Result<Y, String>,
        y: &mut Y,
        t: &mut Scalar,
        y_sol: &mut U,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        dt: &mut Scalar,
        k: &mut [Y],
        y_trial: &Y,
        e: Scalar,
    ) -> Result<(), String>;
    /// Provides the adaptive time step as a function of the error.
    ///
    /// ```math
    /// h_{n+1} = \beta h \left(\frac{e_\mathrm{tol}}{e_{n+1}}\right)^{1/p}
    /// ```
    fn time_step(&self, error: Scalar, dt: &mut Scalar) {
        if error > 0.0 {
            *dt *= (self.dt_beta() * (self.abs_tol() / error).powf(1.0 / self.dt_expn()))
                .max(self.dt_cut())
        }
    }
}

/// Required methods for explicit ordinary differential equation solvers with internal variables.
pub trait ExplicitIV<Y, Z, U, V>
where
    Self: InterpolateSolutionIV<Y, Z, U, V> + OdeSolver<Y, U> + VariableStep,
    Y: Tensor,
    Z: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    const SLOPES: usize;
    #[doc = include_str!("explicit_iv.md")]
    fn integrate(
        &self,
        mut function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        time: &[Scalar],
        initial_condition: Y,
        initial_evaluation: Z,
    ) -> Result<(Vector, U, U, V), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = t_0;
        let mut dt = t_f - t_0;
        let mut t_sol = Vector::new();
        t_sol.push(t_0);
        let mut y = initial_condition;
        let mut z = initial_evaluation;
        if assert_eq_within_tols(&evaluate(t, &y, &z)?, &z).is_err() {
            return Err(IntegrationError::InconsistentInitialConditions);
        }
        let mut k = vec![Y::default(); Self::SLOPES];
        k[0] = function(t, &y, &z)?;
        let mut y_sol = U::new();
        y_sol.push(y.clone());
        let mut z_sol = V::new();
        z_sol.push(z.clone());
        let mut dydt_sol = U::new();
        dydt_sol.push(k[0].clone());
        let mut y_trial = Y::default();
        let mut z_trial = Z::default();
        while t < t_f {
            match self.slopes(
                &mut function,
                &mut evaluate,
                &y,
                &z,
                t,
                dt,
                &mut k,
                &mut y_trial,
                &mut z_trial,
            ) {
                Ok(e) => {
                    if let Some(error) = self
                        .step(
                            &mut function,
                            &mut y,
                            &mut z,
                            &mut t,
                            &mut y_sol,
                            &mut z_sol,
                            &mut t_sol,
                            &mut dydt_sol,
                            &mut dt,
                            &mut k,
                            &y_trial,
                            &z_trial,
                            e,
                        )
                        .err()
                    {
                        dt *= self.dt_cut();
                        if dt < self.dt_min() {
                            return Err(IntegrationError::MinimumStepSizeUpstream(
                                self.dt_min(),
                                error,
                                format!("{self:?}"),
                            ));
                        }
                    } else {
                        dt = dt.min(t_f - t);
                        if dt < self.dt_min() && t < t_f {
                            return Err(IntegrationError::MinimumStepSizeReached(
                                self.dt_min(),
                                format!("{self:?}"),
                            ));
                        }
                    }
                }
                Err(error) => {
                    dt *= self.dt_cut();
                    if dt < self.dt_min() {
                        return Err(IntegrationError::MinimumStepSizeUpstream(
                            self.dt_min(),
                            error,
                            format!("{self:?}"),
                        ));
                    }
                }
            }
        }
        if time.len() > 2 {
            let t_int = Vector::from(time);
            let (y_int, dydt_int, z_int) =
                self.interpolate(&t_int, &t_sol, &y_sol, &z_sol, function, evaluate)?;
            Ok((t_int, y_int, dydt_int, z_int))
        } else {
            Ok((t_sol, y_sol, dydt_sol, z_sol))
        }
    }
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn slopes(
        &self,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<Scalar, String>;
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn step(
        &self,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        y: &mut Y,
        z: &mut Z,
        t: &mut Scalar,
        y_sol: &mut U,
        z_sol: &mut V,
        t_sol: &mut Vector,
        dydt_sol: &mut U,
        dt: &mut Scalar,
        k: &mut [Y],
        y_trial: &Y,
        z_trial: &Z,
        e: Scalar,
    ) -> Result<(), String>;
}

/// Required methods for zeroth-order implicit ordinary differential equation solvers.
pub trait ImplicitZerothOrder<Y, U>
where
    Self: InterpolateSolution<Y, U> + OdeSolver<Y, U>,
    Y: Solution,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    #[doc = include_str!("implicit.md")]
    fn integrate(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        time: &[Scalar],
        initial_condition: Y,
        solver: impl ZerothOrderRootFinding<Y>,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

/// Required methods for first-order implicit ordinary differential equation solvers.
pub trait ImplicitFirstOrder<Y, J, U>
where
    Self: InterpolateSolution<Y, U> + OdeSolver<Y, U>,
    Y: Solution + Div<J, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Tensor + TensorArray,
    U: TensorVec<Item = Y>,
{
    #[doc = include_str!("implicit.md")]
    fn integrate(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        jacobian: impl Fn(Scalar, &Y) -> Result<J, IntegrationError>,
        time: &[Scalar],
        initial_condition: Y,
        solver: impl FirstOrderRootFinding<Y, J, Y>,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

/// Possible errors encountered when integrating.
pub enum IntegrationError {
    InconsistentInitialConditions,
    InitialTimeNotLessThanFinalTime,
    Intermediate(String),
    LengthTimeLessThanTwo,
    MinimumStepSizeReached(Scalar, String),
    MinimumStepSizeUpstream(Scalar, String, String),
    Upstream(String, String),
}

impl From<String> for IntegrationError {
    fn from(error: String) -> Self {
        Self::Intermediate(error)
    }
}

impl Debug for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InconsistentInitialConditions => {
                "\x1b[1;91mThe initial condition z_0 is not consistent with g(t_0, y_0)."
                    .to_string()
            }
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::Intermediate(message) => message.to_string(),
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
            }
            Self::MinimumStepSizeReached(dt_min, integrator) => {
                format!(
                    "\x1b[1;91mMinimum time step ({dt_min:?}) reached.\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::MinimumStepSizeUpstream(dt_min, error, integrator) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    Causing error: \x1b[1;91mMinimum time step ({dt_min:?}) reached.\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::Upstream(error, integrator) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
        };
        write!(f, "\n{}\n\x1b[0;2;31m{}\x1b[0m\n", error, defeat_message())
    }
}

impl Display for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InconsistentInitialConditions => {
                "\x1b[1;91mThe initial condition z_0 is not consistent with g(t_0, y_0)."
                    .to_string()
            }
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::Intermediate(message) => message.to_string(),
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
            }
            Self::MinimumStepSizeReached(dt_min, integrator) => {
                format!(
                    "\x1b[1;91mMinimum time step ({dt_min:?}) reached.\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::MinimumStepSizeUpstream(dt_min, error, integrator) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    Causing error: \x1b[1;91mMinimum time step ({dt_min:?}) reached.\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::Upstream(error, integrator) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

impl From<IntegrationError> for String {
    fn from(error: IntegrationError) -> Self {
        format!("{}", error)
    }
}

impl From<IntegrationError> for TestError {
    fn from(error: IntegrationError) -> Self {
        TestError {
            message: error.to_string(),
        }
    }
}
