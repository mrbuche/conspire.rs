crate::math::integrate::ode::explicit::variable_step::test::test_explicit_variable_step!(
    super::DormandPrince::default(),
    crate::math::assert::Assert {
        abs_tol: 1e-9,
        rel_tol: 1e-9,
        ..crate::math::assert::Assert::default()
    }
);

mod bounded {
    use super::super::{DormandPrince, Tableau};
    use crate::math::{
        Quantity, Tensor, TensorVector,
        integrate::{ButcherTableau, Explicit, IntegrationError, Spectrum, Times},
    };
    use crate::units::{Rate, Time};

    type Solution = (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>);

    #[test]
    fn steps_stay_within_the_limit() {
        let scale = 1e-3;
        let safety = 0.5;
        let limit = safety * scale * Tableau::stability().real;
        let (t, y, _): Solution = DormandPrince::default()
            .integrate_bounded(
                |_: Quantity<Time>, y: &Quantity| Ok(y * -Rate::per_second(1e3)),
                |_, _| Ok(Spectrum::Real(Time::seconds(scale))),
                safety,
                &[Quantity::new(0.0), Quantity::new(0.1)],
                Quantity::new(1.0),
            )
            .unwrap();
        assert!(
            t.iter()
                .zip(t.iter().skip(1))
                .all(|(a, b)| (*b - *a).value() <= limit * (1.0 + 1e-9))
        );
        assert!(y.iter().last().unwrap().value().abs() < 1e-6);
    }

    #[test]
    fn reaching_the_minimum_step() {
        let result: Result<Solution, IntegrationError> = DormandPrince::default()
            .integrate_bounded(
                |_: Quantity<Time>, y: &Quantity| Ok(y * -Rate::per_second(1.0)),
                |_, _| Ok(Spectrum::Real(Time::seconds(1e-30))),
                1.0,
                &[Quantity::new(0.0), Quantity::new(1.0)],
                Quantity::new(1.0),
            );
        assert!(matches!(
            result,
            Err(IntegrationError::MinimumStepSizeReached(..))
        ));
    }

    #[test]
    fn invalid_safety_factor() {
        let result: Result<Solution, IntegrationError> = DormandPrince::default()
            .integrate_bounded(
                |_: Quantity<Time>, y: &Quantity| Ok(y * -Rate::per_second(1.0)),
                |_, _| Ok(Spectrum::Real(Time::seconds(1.0))),
                2.0,
                &[Quantity::new(0.0), Quantity::new(1.0)],
                Quantity::new(1.0),
            );
        assert!(matches!(
            result,
            Err(IntegrationError::InvalidSafetyFactor(_))
        ));
    }
}
