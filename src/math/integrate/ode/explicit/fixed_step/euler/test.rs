crate::math::integrate::ode::explicit::fixed_step::test::test_explicit_fixed_step!(super::Euler {
    dt: 0.1
});

mod bounded {
    use super::super::Euler;
    use crate::math::{
        Quantity, Tensor, TensorVector,
        integrate::{Explicit, IntegrationError, Spectrum, Times},
    };
    use crate::units::{Rate, Time};

    type Solution = (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>);

    fn run(
        dt: f64,
        spectrum: Spectrum,
        safety: f64,
        t_f: f64,
    ) -> Result<Solution, IntegrationError> {
        Euler { dt }.integrate_bounded(
            |_: Quantity<Time>, y: &Quantity| Ok(y * -Rate::per_second(10.0)),
            |_, _| Ok(spectrum),
            safety,
            &[Quantity::new(0.0), Quantity::new(t_f)],
            Quantity::new(1.0),
        )
    }

    const REAL: Spectrum = Spectrum::Real(Time::seconds(0.1));

    #[test]
    fn within_limit() {
        assert!(run(0.19, REAL, 1.0, 2.0).is_ok());
    }

    #[test]
    fn beyond_limit() {
        assert!(matches!(
            run(0.21, REAL, 1.0, 2.0),
            Err(IntegrationError::UnstableTimeStep(..))
        ));
    }

    #[test]
    fn limit_is_where_the_method_diverges() {
        let last = |dt| {
            let (_, y, _): Solution = Euler { dt }
                .integrate(
                    |_: Quantity<Time>, y: &Quantity| Ok(y * -Rate::per_second(10.0)),
                    &[Quantity::new(0.0), Quantity::new(20.0)],
                    Quantity::new(1.0),
                )
                .unwrap();
            y.iter().last().unwrap().value().abs()
        };
        assert!(last(0.19) < 1e-3);
        assert!(last(0.21) > 1.0);
    }

    #[test]
    fn safety_factor_scales_the_limit() {
        assert!(run(0.15, REAL, 1.0, 1.5).is_ok());
        assert!(matches!(
            run(0.15, REAL, 0.5, 1.5),
            Err(IntegrationError::UnstableTimeStep(..))
        ));
    }

    #[test]
    fn imaginary_spectrum_is_never_stable() {
        let error = run(1e-6, Spectrum::Imaginary(Time::seconds(1.0)), 1.0, 1e-5).unwrap_err();
        assert!(matches!(error, IntegrationError::UnstableTimeStep(..)));
        assert!(format!("{error}").contains("stability limit"));
    }

    #[test]
    fn invalid_safety_factor() {
        [0.0, -1.0, 1.5, f64::NAN].iter().for_each(|&safety| {
            let error = run(0.1, REAL, safety, 1.0).unwrap_err();
            assert!(matches!(error, IntegrationError::InvalidSafetyFactor(_)));
            assert!(format!("{error}").contains("safety factor"));
        });
    }
}
