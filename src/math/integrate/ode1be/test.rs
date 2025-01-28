use super::{
    super::{
        super::optimize::{GradientDescent, NewtonRaphson, Optimization},
        test::test_implicit,
    },
    Ode1be,
};

mod gradient_descent {
    use super::*;
    test_implicit!(Ode1be {
        opt_alg: Optimization::GradientDescent(GradientDescent {
            ..Default::default()
        }),
        ..Default::default()
    });
}

mod newton_raphson {
    use super::*;
    test_implicit!(Ode1be {
        opt_alg: Optimization::NewtonRaphson(NewtonRaphson {
            ..Default::default()
        }),
        ..Default::default()
    });
}
