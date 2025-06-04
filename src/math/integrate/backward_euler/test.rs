use super::{
    super::super::{
        Tensor, TensorRank0, Vector,
        integrate::{Implicit, test::zero_to_one},
        optimize::{GradientDescent, Optimization},
        test::TestError,
    },
    BackwardEuler,
};

const LENGTH: usize = 33;
const TOLERANCE: TensorRank0 = crate::ABS_TOL;

mod gradient_descent {
    use super::*;
    #[test]
    fn first_order_tensor_rank_0() -> Result<(), TestError> {
        let (time, solution, function): (Vector, Vector, _) = BackwardEuler {
            opt_alg: Optimization::GradientDescent(GradientDescent {
                ..Default::default()
            }),
        }
        .integrate(
            |_: TensorRank0, _: &TensorRank0| Ok(1.0),
            |_: TensorRank0, _: &TensorRank0| Ok(0.0),
            &zero_to_one::<LENGTH>(),
            0.0,
        )?;
        time.iter()
            .zip(solution.iter().zip(function.iter()))
            .for_each(|(t, (y, f))| {
                assert!((t - y).abs() < TOLERANCE && (1.0 - f).abs() < TOLERANCE)
            });
        Ok(())
    }
}

mod newton_raphson {
    use super::*;
    #[test]
    fn first_order_tensor_rank_0() -> Result<(), TestError> {
        let (time, solution, function): (Vector, Vector, _) = BackwardEuler::default().integrate(
            |_: TensorRank0, _: &TensorRank0| Ok(1.0),
            |_: TensorRank0, _: &TensorRank0| Ok(0.0),
            &zero_to_one::<LENGTH>(),
            0.0,
        )?;
        time.iter()
            .zip(solution.iter().zip(function.iter()))
            .for_each(|(t, (y, f))| {
                assert!((t - y).abs() < TOLERANCE && (1.0 - f).abs() < TOLERANCE)
            });
        Ok(())
    }
}
