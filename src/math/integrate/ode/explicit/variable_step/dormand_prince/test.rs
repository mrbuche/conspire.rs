crate::math::integrate::ode::explicit::variable_step::test::test_explicit_variable_step!(
    super::DormandPrince::default(),
    crate::math::assert::Assert {
        abs_tol: 1e-9,
        rel_tol: 1e-9,
        ..crate::math::assert::Assert::default()
    }
);
