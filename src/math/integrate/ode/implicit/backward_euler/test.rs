use crate::math::{Scalar, Tensor, Vector, integrate::BackwardEuler, test::TestError};

const TIME_STEP: Scalar = 0.1;
const TOLERANCE: Scalar = TIME_STEP;

mod gradient_descent {
    use super::*;
    use crate::math::{integrate::ImplicitZerothOrder, optimize::GradientDescent};
    #[test]
    fn first_order_tensor_rank_0() -> Result<(), TestError> {
        let (time, solution, function): (Vector, Vector, _) = BackwardEuler { dt: TIME_STEP }
            .integrate(
                |t: Scalar, _: &Scalar| Ok(t),
                &[0.0, 1.0],
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
    use super::*;
    use crate::math::{integrate::ImplicitFirstOrder, optimize::NewtonRaphson};
    #[test]
    fn first_order_tensor_rank_0() -> Result<(), TestError> {
        let (time, solution, function): (Vector, Vector, _) = BackwardEuler { dt: TIME_STEP }
            .integrate(
                |t: Scalar, _: &Scalar| Ok(t),
                |_: Scalar, _: &Scalar| Ok(1.0),
                &[0.0, 1.0],
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
