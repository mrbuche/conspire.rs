use crate::math::{
    Banded, Scalar, Tensor, TensorVec, Vector,
    integrate::{
        DaeSolver, DaeSolverFirstOrderMinimize, DaeSolverFirstOrderRoot,
        DaeSolverSecondOrderMinimize, DaeSolverZerothOrderRoot, IntegrationError,
        VariableStepExplicitDaeSolver, VariableStepExplicitDaeSolverFirstOrderMinimize,
        VariableStepExplicitDaeSolverFirstOrderRoot,
        VariableStepExplicitDaeSolverSecondOrderMinimize,
        VariableStepExplicitDaeSolverZerothOrderRoot, ode::explicit::variable_step::verner_9::*,
    },
    optimize::{
        EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, SecondOrderOptimization,
        ZerothOrderRootFinding,
    },
};
use std::ops::{Mul, Sub};

impl<Y, Z, U, V> DaeSolver<Y, Z, U, V> for Verner9
where
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
}

impl<Y, Z, U, V> VariableStepExplicitDaeSolver<Y, Z, U, V> for Verner9
where
    Self: DaeSolver<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
{
    fn slopes_solve(
        mut evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        mut solution: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        y: &Y,
        z: &Z,
        t: Scalar,
        dt: Scalar,
        k: &mut [Y],
        y_trial: &mut Y,
        z_trial: &mut Z,
    ) -> Result<(), String> {
        k[0] = evolution(t, y, z)?;
        *y_trial = &k[0] * (A_2_1 * dt) + y;
        *z_trial = solution(t + C_2 * dt, y_trial, z)?;
        k[1] = evolution(t + C_2 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_3_1 * dt) + &k[1] * (A_3_2 * dt) + y;
        *z_trial = solution(t + C_3 * dt, y_trial, z_trial)?;
        k[2] = evolution(t + C_3 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_4_1 * dt) + &k[2] * (A_4_3 * dt) + y;
        *z_trial = solution(t + C_4 * dt, y_trial, z_trial)?;
        k[3] = evolution(t + C_4 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_5_1 * dt) + &k[2] * (A_5_3 * dt) + &k[3] * (A_5_4 * dt) + y;
        *z_trial = solution(t + C_5 * dt, y_trial, z_trial)?;
        k[4] = evolution(t + C_5 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_6_1 * dt) + &k[3] * (A_6_4 * dt) + &k[4] * (A_6_5 * dt) + y;
        *z_trial = solution(t + C_6 * dt, y_trial, z_trial)?;
        k[5] = evolution(t + C_6 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_7_1 * dt)
            + &k[3] * (A_7_4 * dt)
            + &k[4] * (A_7_5 * dt)
            + &k[5] * (A_7_6 * dt)
            + y;
        *z_trial = solution(t + C_7 * dt, y_trial, z_trial)?;
        k[6] = evolution(t + C_7 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_8_1 * dt) + &k[5] * (A_8_6 * dt) + &k[6] * (A_8_7 * dt) + y;
        *z_trial = solution(t + C_8 * dt, y_trial, z_trial)?;
        k[7] = evolution(t + C_8 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_9_1 * dt)
            + &k[5] * (A_9_6 * dt)
            + &k[6] * (A_9_7 * dt)
            + &k[7] * (A_9_8 * dt)
            + y;
        *z_trial = solution(t + C_9 * dt, y_trial, z_trial)?;
        k[8] = evolution(t + C_9 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_10_1 * dt)
            + &k[5] * (A_10_6 * dt)
            + &k[6] * (A_10_7 * dt)
            + &k[7] * (A_10_8 * dt)
            + &k[8] * (A_10_9 * dt)
            + y;
        *z_trial = solution(t + C_10 * dt, y_trial, z_trial)?;
        k[9] = evolution(t + C_10 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_11_1 * dt)
            + &k[5] * (A_11_6 * dt)
            + &k[6] * (A_11_7 * dt)
            + &k[7] * (A_11_8 * dt)
            + &k[8] * (A_11_9 * dt)
            + &k[9] * (A_11_10 * dt)
            + y;
        *z_trial = solution(t + C_11 * dt, y_trial, z_trial)?;
        k[10] = evolution(t + C_11 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_12_1 * dt)
            + &k[5] * (A_12_6 * dt)
            + &k[6] * (A_12_7 * dt)
            + &k[7] * (A_12_8 * dt)
            + &k[8] * (A_12_9 * dt)
            + &k[9] * (A_12_10 * dt)
            + &k[10] * (A_12_11 * dt)
            + y;
        *z_trial = solution(t + C_12 * dt, y_trial, z_trial)?;
        k[11] = evolution(t + C_12 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_13_1 * dt)
            + &k[5] * (A_13_6 * dt)
            + &k[6] * (A_13_7 * dt)
            + &k[7] * (A_13_8 * dt)
            + &k[8] * (A_13_9 * dt)
            + &k[9] * (A_13_10 * dt)
            + &k[10] * (A_13_11 * dt)
            + &k[11] * (A_13_12 * dt)
            + y;
        *z_trial = solution(t + C_13 * dt, y_trial, z_trial)?;
        k[12] = evolution(t + C_13 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_14_1 * dt)
            + &k[5] * (A_14_6 * dt)
            + &k[6] * (A_14_7 * dt)
            + &k[7] * (A_14_8 * dt)
            + &k[8] * (A_14_9 * dt)
            + &k[9] * (A_14_10 * dt)
            + &k[10] * (A_14_11 * dt)
            + &k[11] * (A_14_12 * dt)
            + &k[12] * (A_14_13 * dt)
            + y;
        *z_trial = solution(t + C_14 * dt, y_trial, z_trial)?;
        k[13] = evolution(t + C_14 * dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_15_1 * dt)
            + &k[5] * (A_15_6 * dt)
            + &k[6] * (A_15_7 * dt)
            + &k[7] * (A_15_8 * dt)
            + &k[8] * (A_15_9 * dt)
            + &k[9] * (A_15_10 * dt)
            + &k[10] * (A_15_11 * dt)
            + &k[11] * (A_15_12 * dt)
            + &k[12] * (A_15_13 * dt)
            + &k[13] * (A_15_14 * dt)
            + y;
        *z_trial = solution(t + dt, y_trial, z_trial)?;
        k[14] = evolution(t + dt, y_trial, z_trial)?;
        *y_trial = &k[0] * (A_16_1 * dt)
            + &k[5] * (A_16_6 * dt)
            + &k[6] * (A_16_7 * dt)
            + &k[7] * (A_16_8 * dt)
            + &k[8] * (A_16_9 * dt)
            + &k[9] * (A_16_10 * dt)
            + &k[10] * (A_16_11 * dt)
            + &k[11] * (A_16_12 * dt)
            + &k[12] * (A_16_13 * dt)
            + y;
        *z_trial = solution(t + dt, y_trial, z_trial)?;
        k[15] = evolution(t + dt, y_trial, z_trial)?;
        *y_trial = (&k[0] * B_1
            + &k[7] * B_8
            + &k[8] * B_9
            + &k[9] * B_10
            + &k[10] * B_11
            + &k[11] * B_12
            + &k[12] * B_13
            + &k[13] * B_14
            + &k[14] * B_15)
            * dt
            + y;
        *z_trial = solution(t + dt, y_trial, z_trial)?;
        Ok(())
    }
}

super::implement_solvers!(Verner9);
