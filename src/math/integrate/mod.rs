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

pub type Ode1be = BackwardEuler;
pub type Ode23 = BogackiShampine;
pub type Ode45 = DormandPrince;
pub type Ode78 = Verner8;
pub type Ode89 = Verner9;

use super::{
    Scalar, Solution, Tensor, TensorArray, TensorVec, TestError, Vector,
    interpolate::InterpolateSolution,
    optimize::{FirstOrderRootFinding, ZerothOrderRootFinding},
};
use crate::defeat_message;
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Div, Mul, Sub},
};

/// Base trait for ordinary differential equation solvers.
pub trait OdeSolver<Y, U>
where
    Self: Debug,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<A, Y, U> OdeSolver<Y, U> for A
where
    A: Debug,
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

/// Base trait for explicit ordinary differential equation solvers.
pub trait Explicit<Y, U>: OdeSolver<Y, U>
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    const SLOPES: usize;
    /// Solves an initial value problem by explicitly integrating a system of ordinary differential equations.
    ///
    /// ```math
    /// \frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0
    /// ```
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
        let mut dt = t_f;
        let mut e;
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
            e = match self.slopes(&mut function, &y, &t, &dt, &mut k, &mut y_trial) {
                Ok(e) => e,
                Err(error) => {
                    return Err(IntegrationError::Upstream(error, format!("{self:?}")));
                }
            };
            match self.step(
                &mut function,
                &mut y,
                &mut t,
                &mut y_sol,
                &mut t_sol,
                &mut dydt_sol,
                &mut dt,
                &mut k,
                &y_trial,
                &e,
            ) {
                Ok(e) => e,
                Err(error) => {
                    return Err(IntegrationError::Upstream(error, format!("{self:?}")));
                }
            };
            dt = dt.min(t_f - t);
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
        t: &Scalar,
        dt: &Scalar,
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
        e: &Scalar,
    ) -> Result<(), String>;
}

/// Base trait for zeroth-order implicit ordinary differential equation solvers.
pub trait ImplicitZerothOrder<Y, U>: OdeSolver<Y, U>
where
    Self: InterpolateSolution<Y, U>,
    Y: Solution,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    /// Solves an initial value problem by implicitly integrating a system of ordinary differential equations.
    ///
    /// ```math
    /// \frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0,\quad \frac{\partial f}{\partial y} = J(t, y)
    /// ```
    fn integrate(
        &self,
        function: impl Fn(Scalar, &Y) -> Result<Y, IntegrationError>,
        time: &[Scalar],
        initial_condition: Y,
        solver: impl ZerothOrderRootFinding<Y>,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

/// Base trait for first-order implicit ordinary differential equation solvers.
pub trait ImplicitFirstOrder<Y, J, U>: OdeSolver<Y, U>
where
    Self: InterpolateSolution<Y, U>,
    Y: Solution + Div<J, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    J: Tensor + TensorArray,
    U: TensorVec<Item = Y>,
{
    /// Solves an initial value problem by implicitly integrating a system of ordinary differential equations.
    ///
    /// ```math
    /// \frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0,\quad \frac{\partial f}{\partial y} = J(t, y)
    /// ```
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
    InitialTimeNotLessThanFinalTime,
    Intermediate(String),
    LengthTimeLessThanTwo,
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
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::Intermediate(message) => message.to_string(),
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
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
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::Intermediate(message) => message.to_string(),
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
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
