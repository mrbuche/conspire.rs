use crate::{
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
        integrate::{
            ImplicitDaeFirstOrderMinimize, ImplicitDaeFirstOrderRoot,
            ImplicitDaeSecondOrderMinimize, ImplicitDaeZerothOrderRoot, IntegrationError, Times,
            VariableStepExplicit,
        },
        optimize::{
            EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, LinearSolver,
            SecondOrderOptimization, ZerothOrderRootFinding,
        },
    },
    units::{Time, UnitInv},
};
use std::ops::{Mul, Sub};

/// Variable-step explicit integrators for implicit differential-algebraic equations.
pub trait ImplicitDaeVariableStepExplicit<Y, U, V, T = Time>
where
    Self: VariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    fn integrate_implicit_dae_variable_step(
        &self,
        mut evolution: impl FnMut(
            Quantity<T>,
            &Y,
            &Derivative<Y, T>,
        ) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        let t_0 = time[0];
        let t_f = time[time.len() - 1];
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if t_0 >= t_f {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = t_0;
        let mut dt = t_f - t_0;
        let mut t_sol = Times::new();
        t_sol.push(t_0);
        let mut dydt = &initial_condition * Quantity::<<T as UnitInv>::Output>::default();
        let mut y = initial_condition;
        let mut k = vec![Derivative::<Y, T>::default(); Self::SLOPES];
        k[0] = evolution(t, &y, &dydt)?;
        let mut y_sol = U::new();
        y_sol.push(y.clone());
        let mut dydt_sol = V::new();
        dydt_sol.push(k[0].clone());
        let mut k_sol: Vec<V> = Vec::new();
        let mut y_trial = Y::default();
        while t < t_f {
            match self.slopes_and_error(
                |t: Quantity<T>, y: &Y| evolution(t, y, &dydt),
                &y,
                t,
                dt,
                &mut k,
                &mut y_trial,
            ) {
                Ok(e) => {
                    if let Some(error) = self
                        .step(
                            |t: Quantity<T>, y: &Y| evolution(t, y, &dydt),
                            &mut y,
                            &mut t,
                            &mut y_sol,
                            &mut t_sol,
                            &mut dydt_sol,
                            &mut k_sol,
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
                                self.dt_min().value(),
                                error,
                                format!("{:?}", self),
                            ));
                        }
                    } else {
                        dydt = k[0].clone();
                        dt = dt.min(t_f - t);
                        if dt < self.dt_min() && t < t_f {
                            return Err(IntegrationError::MinimumStepSizeReached(
                                self.dt_min().value(),
                                format!("{:?}", self),
                            ));
                        }
                    }
                }
                Err(error) => {
                    dt *= self.dt_cut();
                    if dt < self.dt_min() {
                        return Err(IntegrationError::MinimumStepSizeUpstream(
                            self.dt_min().value(),
                            error,
                            format!("{:?}", self),
                        ));
                    }
                }
            }
        }
        if time.len() > 2 {
            let t_int = Times::from(time);
            let (y_int, dydt_int) = self.interpolate_implicit_dae_variable_step(
                evolution, &t_int, &t_sol, &y_sol, &dydt_sol,
            )?;
            Ok((t_int, y_int, dydt_int))
        } else {
            Ok((t_sol, y_sol, dydt_sol))
        }
    }
    fn interpolate_implicit_dae_variable_step(
        &self,
        mut evolution: impl FnMut(
            Quantity<T>,
            &Y,
            &Derivative<Y, T>,
        ) -> Result<Derivative<Y, T>, String>,
        time: &Times<T>,
        tp: &Times<T>,
        yp: &U,
        dydtp: &V,
    ) -> Result<(U, V), IntegrationError> {
        let mut dt;
        let mut i;
        let mut k = vec![Derivative::<Y, T>::default(); Self::SLOPES];
        let mut t;
        let mut y;
        let mut dydt;
        let mut y_int = U::new();
        let mut dydt_int = V::new();
        let mut y_trial = Y::default();
        for time_k in time.iter() {
            i = tp.iter().position(|tp_i| tp_i >= time_k).unwrap();
            if time_k == &tp[i] {
                t = tp[i];
                y_trial = yp[i].clone();
                dt = Quantity::default();
            } else {
                t = tp[i - 1];
                y = &yp[i - 1];
                dydt = &dydtp[i - 1];
                dt = *time_k - t;
                k[0] = evolution(t, y, dydt)?;
                Self::slopes(
                    |t: Quantity<T>, y: &Y| evolution(t, y, dydt),
                    y,
                    t,
                    dt,
                    &mut k,
                    &mut y_trial,
                )?;
            }
            dydt_int.push(evolution(t + dt, &y_trial, &k[0])?);
            y_int.push(y_trial.clone());
        }
        Ok((y_int, dydt_int))
    }
}

impl<I, Y, U, V, T> ImplicitDaeVariableStepExplicit<Y, U, V, T> for I
where
    Self: VariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
}

/// Variable-step explicit integrators for implicit differential-algebraic equations using zeroth-order root-finding.
pub trait ImplicitDaeVariableStepExplicitZerothOrderRoot<G, Y, U, V, T = Time>
where
    Self: ImplicitDaeVariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    fn integrate_implicit_dae_variable_step_explicit_root_0(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<G, String>,
        solver: impl ZerothOrderRootFinding<G, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        mut equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        let evolution = |t: Quantity<T>,
                         y: &Y,
                         dydt_0: &Derivative<Y, T>|
         -> Result<Derivative<Y, T>, String> {
            Ok(solver.root(
                |dydt| function(t, y, dydt),
                dydt_0.clone(),
                equality_constraint(t),
            )?)
        };
        self.integrate_implicit_dae_variable_step(evolution, time, initial_condition)
    }
}

impl<I, G, Y, U, V, T> ImplicitDaeVariableStepExplicitZerothOrderRoot<G, Y, U, V, T> for I
where
    I: ImplicitDaeVariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
}

impl<I, G, Y, U, V, T> ImplicitDaeZerothOrderRoot<G, Y, U, V, T> for I
where
    Self: ImplicitDaeVariableStepExplicitZerothOrderRoot<G, Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<G, String>,
        solver: impl ZerothOrderRootFinding<G, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_implicit_dae_variable_step_explicit_root_0(
            function,
            solver,
            time,
            initial_condition,
            equality_constraint,
        )
    }
}

/// Variable-step explicit integrators for implicit differential-algebraic equations using first-order root-finding.
pub trait ImplicitDaeVariableStepExplicitFirstOrderRoot<F, J, Y, U, V, T = Time>
where
    Self: ImplicitDaeVariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    fn integrate_implicit_dae_variable_step_explicit_root_1(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        mut jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<J, String>,
        solver: impl FirstOrderRootFinding<F, J, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        mut equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        let evolution = |t: Quantity<T>,
                         y: &Y,
                         dydt_0: &Derivative<Y, T>|
         -> Result<Derivative<Y, T>, String> {
            Ok(solver.root(
                |dydt| function(t, y, dydt),
                |dydt| jacobian(t, y, dydt),
                dydt_0.clone(),
                equality_constraint(t),
                LinearSolver::Dense,
            )?)
        };
        self.integrate_implicit_dae_variable_step(evolution, time, initial_condition)
    }
}

impl<I, F, J, Y, U, V, T> ImplicitDaeVariableStepExplicitFirstOrderRoot<F, J, Y, U, V, T> for I
where
    I: ImplicitDaeVariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
}

impl<I, F, J, Y, U, V, T> ImplicitDaeFirstOrderRoot<F, J, Y, U, V, T> for I
where
    Self: ImplicitDaeVariableStepExplicitFirstOrderRoot<F, J, Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<J, String>,
        solver: impl FirstOrderRootFinding<F, J, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_implicit_dae_variable_step_explicit_root_1(
            function,
            jacobian,
            solver,
            time,
            initial_condition,
            equality_constraint,
        )
    }
}

/// Variable-step explicit integrators for implicit differential-algebraic equations using first-order minimization.
pub trait ImplicitDaeVariableStepExplicitFirstOrderMinimize<F, G, Y, U, V, T = Time>
where
    Self: ImplicitDaeVariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate_implicit_dae_variable_step_explicit_minimize_1(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        mut jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<G, String>,
        solver: impl FirstOrderOptimization<F, G, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        mut equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        let evolution = |t: Quantity<T>,
                         y: &Y,
                         dydt_0: &Derivative<Y, T>|
         -> Result<Derivative<Y, T>, String> {
            Ok(solver.minimize(
                |dydt| function(t, y, dydt),
                |dydt| jacobian(t, y, dydt),
                dydt_0.clone(),
                equality_constraint(t),
            )?)
        };
        self.integrate_implicit_dae_variable_step(evolution, time, initial_condition)
    }
}

impl<I, F, G, Y, U, V, T> ImplicitDaeVariableStepExplicitFirstOrderMinimize<F, G, Y, U, V, T> for I
where
    I: ImplicitDaeVariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
}

impl<I, F, G, Y, U, V, T> ImplicitDaeFirstOrderMinimize<F, G, Y, U, V, T> for I
where
    Self: ImplicitDaeVariableStepExplicitFirstOrderMinimize<F, G, Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<G, String>,
        solver: impl FirstOrderOptimization<F, G, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_implicit_dae_variable_step_explicit_minimize_1(
            function,
            jacobian,
            solver,
            time,
            initial_condition,
            equality_constraint,
        )
    }
}

/// Variable-step explicit integrators for implicit differential-algebraic equations using second-order minimization.
pub trait ImplicitDaeVariableStepExplicitSecondOrderMinimize<F, J, H, Y, U, V, T = Time>
where
    Self: ImplicitDaeVariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate_implicit_dae_variable_step_explicit_minimize_2(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        mut jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<J, String>,
        mut hessian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<H, String>,
        solver: impl SecondOrderOptimization<F, J, H, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        mut equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
        linear_solver: LinearSolver,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        let evolution = |t: Quantity<T>,
                         y: &Y,
                         dydt_0: &Derivative<Y, T>|
         -> Result<Derivative<Y, T>, String> {
            Ok(solver.minimize(
                |dydt| function(t, y, dydt),
                |dydt| jacobian(t, y, dydt),
                |dydt| hessian(t, y, dydt),
                dydt_0.clone(),
                equality_constraint(t),
                linear_solver.clone(),
            )?)
        };
        self.integrate_implicit_dae_variable_step(evolution, time, initial_condition)
    }
}

impl<I, F, J, H, Y, U, V, T> ImplicitDaeVariableStepExplicitSecondOrderMinimize<F, J, H, Y, U, V, T>
    for I
where
    I: ImplicitDaeVariableStepExplicit<Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
}

impl<I, F, J, H, Y, U, V, T> ImplicitDaeSecondOrderMinimize<F, J, H, Y, U, V, T> for I
where
    Self: ImplicitDaeVariableStepExplicitSecondOrderMinimize<F, J, H, Y, U, V, T>,
    Y: Differentiate<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
    T: UnitInv,
    for<'a> &'a Y: Mul<Scalar, Output = Y>
        + Mul<Quantity<<T as UnitInv>::Output>, Output = Derivative<Y, T>>
        + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<J, String>,
        hessian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<H, String>,
        solver: impl SecondOrderOptimization<F, J, H, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
        linear_solver: LinearSolver,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_implicit_dae_variable_step_explicit_minimize_2(
            function,
            jacobian,
            hessian,
            solver,
            time,
            initial_condition,
            equality_constraint,
            linear_solver,
        )
    }
}
