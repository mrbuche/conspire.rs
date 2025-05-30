use super::{
    super::super::{
        Tensor, TensorRank0, Vector,
        integrate::{Implicit, test::zero_to_tau},
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
        let (time, solution): (Vector, Vector) = BackwardEuler {
            opt_alg: Optimization::GradientDescent(GradientDescent {
                ..Default::default()
            }),
        }
        .integrate(
            |_: TensorRank0, _: &TensorRank0| Ok(1.0),
            |_: TensorRank0, _: &TensorRank0| Ok(0.0),
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
            |_: TensorRank0, _: &TensorRank0| Ok(1.0),
            |_: TensorRank0, _: &TensorRank0| Ok(0.0),
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
