use super::{
    super::super::{
        integrate::{test::zero_to_tau, Implicit},
        optimize::{GradientDescent, Optimization},
        test::TestError,
        Tensor, TensorRank0, Vector,
    },
    BackwardEuler,
};

const LENGTH: usize = 33;
const TOLERANCE: TensorRank0 = crate::ABS_TOL;

mod gradient_descent {
    use super::*;
    #[test]
    fn first_order_tensor_rank_0() -> Result<(), TestError> {
        let (time, solution): (Vector, Vector) = BackwardEuler {
            opt_alg: Optimization::GradientDescent(GradientDescent {
                ..Default::default()
            }),
        }
        .integrate(
            |_: &TensorRank0, _: &TensorRank0| 1.0,
            |_: &TensorRank0, _: &TensorRank0| 0.0,
            &zero_to_tau::<LENGTH>(),
            0.0,
        )?;
        time.iter().zip(solution.iter()).for_each(|(t, y)| {
            assert!(
                (t - y).abs() < TOLERANCE // || (t / y - 1.0).abs() < TOLERANCE
            )
        });
        Ok(())
    }
}

mod newton_raphson {
    use super::*;
    #[test]
    fn first_order_tensor_rank_0() -> Result<(), TestError> {
        let (time, solution): (Vector, Vector) = BackwardEuler::default().integrate(
            |_: &TensorRank0, _: &TensorRank0| 1.0,
            |_: &TensorRank0, _: &TensorRank0| 0.0,
            &zero_to_tau::<LENGTH>(),
            0.0,
        )?;
        time.iter().zip(solution.iter()).for_each(|(t, y)| {
            assert!(
                (t - y).abs() < TOLERANCE // || (t / y - 1.0).abs() < TOLERANCE
            )
        });
        Ok(())
    }
}
