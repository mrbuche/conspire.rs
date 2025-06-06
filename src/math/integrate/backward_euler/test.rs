use super::{
    super::super::{Tensor, TensorRank0, Vector, integrate::test::zero_to_one, test::TestError},
    BackwardEuler,
};

const LENGTH: usize = 33;
const TOLERANCE: TensorRank0 = 1.0 / (LENGTH as TensorRank0);

mod gradient_descent {
    use super::{
        super::super::{super::optimize::GradientDescent, ImplicitZerothOrder},
        *,
    };
    #[test]
    fn first_order_tensor_rank_0() -> Result<(), TestError> {
        let (time, solution, function): (Vector, Vector, _) = BackwardEuler::default().integrate(
            |t: TensorRank0, _: &TensorRank0| Ok(t),
            &zero_to_one::<LENGTH>(),
            0.0,
            GradientDescent::default(),
        )?;
        time.iter()
            .zip(solution.iter().zip(function.iter()))
            .for_each(|(t, (y, f))| {
                assert!((0.5 * t * t - y).abs() < TOLERANCE && (t - f).abs() < TOLERANCE)
            });
        Ok(())
    }
}

mod newton_raphson {
    use super::{
        super::super::{super::optimize::NewtonRaphson, ImplicitFirstOrder},
        *,
    };
    #[test]
    fn first_order_tensor_rank_0() -> Result<(), TestError> {
        let (time, solution, function): (Vector, Vector, _) = BackwardEuler::default().integrate(
            |t: TensorRank0, _: &TensorRank0| Ok(t),
            |_: TensorRank0, _: &TensorRank0| Ok(1.0),
            &zero_to_one::<LENGTH>(),
            0.0,
            NewtonRaphson::default(),
        )?;
        time.iter()
            .zip(solution.iter().zip(function.iter()))
            .for_each(|(t, (y, f))| {
                assert!((0.5 * t * t - y).abs() < TOLERANCE && (t - f).abs() < TOLERANCE)
            });
        Ok(())
    }
}
