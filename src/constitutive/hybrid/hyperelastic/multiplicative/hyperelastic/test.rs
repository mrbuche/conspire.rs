use crate::{
    constitutive::{
        hybrid::ElasticMultiplicative,
        solid::{
            elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
            hyperelastic::{NeoHookean, SaintVenantKirchhoff},
        },
    },
    math::{
        Vector,
        assert::{Assert, AssertionError},
    },
};

use crate::{
    constitutive::solid::elastic::AppliedLoad,
    math::{
        Norm,
        optimize::{
            Armijo, Direct, Error, Goldstein, NewtonRaphson, NoSearch, ObjectiveMerit, Residual,
            ResidualMerit, SolveStrategy, TrustRegion, Wolfe,
        },
    },
    mechanics::*,
};

const STRETCH: Scalar = 1.5;

fn model() -> ElasticMultiplicative<NeoHookean, SaintVenantKirchhoff> {
    ElasticMultiplicative::from((
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        SaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
    ))
}

fn blocked(
    strategy: SolveStrategy,
) -> Result<(DeformationGradient, DeformationGradient2), AssertionError> {
    use crate::constitutive::solid::hyperelastic::internal_variables::Minimize;
    let (f, f_2) = model().minimize(
        AppliedLoad::UniaxialStress(STRETCH),
        NewtonRaphson::default(),
        strategy,
    )?;
    check(&f, &f_2)?;
    Ok((f, f_2))
}

fn check(f: &DeformationGradient, f_2: &DeformationGradient2) -> Result<(), AssertionError> {
    use crate::constitutive::solid::elastic::internal_variables::ElasticIV;
    Assert::default().zero_within_tols(&model().internal_variables_residual(f, f_2)?)?;
    Assert::default().eq_within_tols(
        Vector::from([
            f[0][0].value(),
            f[0][1].value(),
            f[0][2].value(),
            f[1][2].value(),
        ]),
        &Vector::from([STRETCH, 0.0, 0.0, 0.0]),
    )
}

#[test]
fn minimize_monolithic() -> Result<(), AssertionError> {
    blocked(SolveStrategy::Monolithic { elimination: false })?;
    Ok(())
}

#[test]
fn minimize_monolithic_elimination() -> Result<(), AssertionError> {
    let (f, f_2) = blocked(SolveStrategy::Monolithic { elimination: false })?;
    let (f_elim, f_2_elim) = blocked(SolveStrategy::Monolithic { elimination: true })?;
    Assert::default().eq_within_tols(&f, &f_elim)?;
    Assert::default().eq_within_tols(&f_2, &f_2_elim)
}

#[test]
fn minimize_condensed() -> Result<(), AssertionError> {
    let (f, f_2) = blocked(SolveStrategy::Monolithic { elimination: false })?;
    let (f_condensed, f_2_condensed) = blocked(SolveStrategy::Condensed(NewtonRaphson::default()))?;
    Assert::default().eq_within_tols(&f, &f_condensed)?;
    Assert::default().eq_within_tols(&f_2, &f_2_condensed)
}

fn searched(
    line_search: impl ObjectiveMerit,
    strategy: SolveStrategy,
) -> Result<(), AssertionError> {
    use crate::constitutive::solid::hyperelastic::internal_variables::Minimize;
    let (f, f_2) = model().minimize(
        AppliedLoad::UniaxialStress(STRETCH),
        NewtonRaphson {
            linear_solver: Direct,
            ..Default::default()
        }
        .with_line_search(line_search),
        strategy.clone(),
    )?;
    check(&f, &f_2)?;
    let (f_none, f_2_none) = blocked(strategy)?;
    Assert::default().eq_within_tols(&f, &f_none)?;
    Assert::default().eq_within_tols(&f_2, &f_2_none)
}

#[test]
fn minimize_line_search() -> Result<(), AssertionError> {
    for strategy in [
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
        SolveStrategy::Condensed(NewtonRaphson::default()),
    ] {
        searched(
            Armijo {
                control: 1e-3,
                cut_back: 9e-1,
                max_steps: 100,
            },
            strategy.clone(),
        )?;
        searched(
            Goldstein {
                control: 1e-4,
                cut_back: 5e-1,
                max_steps: 100,
            },
            strategy.clone(),
        )?;
        searched(
            Error {
                cut_back: 5e-1,
                max_steps: 100,
            },
            strategy,
        )?
    }
    Ok(())
}

const STRETCH_FAR: Scalar = 6.0;

fn far(
    trust_region: TrustRegion,
    strategy: SolveStrategy,
) -> Result<(DeformationGradient, DeformationGradient2), AssertionError> {
    far_searched(trust_region, NoSearch, strategy)
}

fn far_searched(
    trust_region: TrustRegion,
    line_search: impl ResidualMerit,
    strategy: SolveStrategy,
) -> Result<(DeformationGradient, DeformationGradient2), AssertionError> {
    use crate::constitutive::solid::elastic::internal_variables::Root;
    let (f, f_2) = model().root(
        AppliedLoad::UniaxialStress(STRETCH_FAR),
        NewtonRaphson {
            trust_region,
            max_steps: 200,
            linear_solver: Direct,
            ..Default::default()
        }
        .with_line_search(line_search),
        strategy,
    )?;
    use crate::constitutive::solid::elastic::internal_variables::ElasticIV;
    Assert::default().zero_within_tols(&model().internal_variables_residual(&f, &f_2)?)?;
    Assert::default().eq_within_tols(
        Vector::from([
            f[0][0].value(),
            f[0][1].value(),
            f[0][2].value(),
            f[1][2].value(),
        ]),
        &Vector::from([STRETCH_FAR, 0.0, 0.0, 0.0]),
    )?;
    Ok((f, f_2))
}

#[test]
fn root_residual_search_is_what_the_trust_region_was() -> Result<(), AssertionError> {
    for strategy in [
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
        SolveStrategy::Condensed(NewtonRaphson::default()),
    ] {
        far_searched(
            TrustRegion::None,
            Residual {
                control: 1e-3,
                cut_back: 5e-1,
                max_steps: 100,
            },
            strategy,
        )?;
    }
    Ok(())
}

#[test]
fn root_trust_region() -> Result<(), AssertionError> {
    let (f, f_2) = far(
        TrustRegion::Fixed {
            radius: 5e-1,
            norm: Norm::Chebyshev,
        },
        SolveStrategy::Monolithic { elimination: false },
    )?;
    for strategy in [
        SolveStrategy::Monolithic { elimination: true },
        SolveStrategy::Condensed(NewtonRaphson::default()),
    ] {
        let (f_other, f_2_other) = far(
            TrustRegion::Fixed {
                radius: 5e-1,
                norm: Norm::Chebyshev,
            },
            strategy,
        )?;
        Assert::default().eq_within_tols(&f, &f_other)?;
        Assert::default().eq_within_tols(&f_2, &f_2_other)?
    }
    Ok(())
}

#[test]
fn root_trust_region_needed() {
    for strategy in [
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
        SolveStrategy::Condensed(NewtonRaphson::default()),
    ] {
        assert!(far(TrustRegion::None, strategy).is_err())
    }
}

#[test]
fn root_line_search_error() -> Result<(), AssertionError> {
    use crate::constitutive::solid::elastic::internal_variables::Root;
    for strategy in [
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
        SolveStrategy::Condensed(NewtonRaphson::default()),
    ] {
        let (f, f_2) = model().root(
            AppliedLoad::UniaxialStress(STRETCH),
            NewtonRaphson {
                linear_solver: Direct,
                ..Default::default()
            }
            .with_line_search(Error {
                cut_back: 5e-1,
                max_steps: 100,
            }),
            strategy,
        )?;
        check(&f, &f_2)?
    }
    Ok(())
}

#[test]
fn root_line_search_residual() -> Result<(), AssertionError> {
    use crate::constitutive::solid::elastic::internal_variables::Root;
    for strategy in [
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
        SolveStrategy::Condensed(NewtonRaphson::default()),
    ] {
        let (f, f_2) = model().root(
            AppliedLoad::UniaxialStress(STRETCH),
            NewtonRaphson {
                linear_solver: Direct,
                ..Default::default()
            }
            .with_line_search(Residual {
                control: 1e-3,
                cut_back: 5e-1,
                max_steps: 100,
            }),
            strategy,
        )?;
        check(&f, &f_2)?
    }
    Ok(())
}

#[test]
#[should_panic(expected = "gradient of the merit function")]
fn minimize_line_search_wolfe() {
    searched(
        Wolfe {
            control_1: 1e-3,
            control_2: 9e-1,
            cut_back: 5e-1,
            max_steps: 100,
            strong: true,
        },
        SolveStrategy::Monolithic { elimination: false },
    )
    .unwrap();
}

#[test]
fn root_fixed_at_point() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic::internal_variables::ElasticIV,
        math::optimize::{EqualityConstraint, RootFinding},
    };
    let model = model();
    let deformation_gradient =
        DeformationGradient::from([[1.2, 0.1, 0.0], [0.0, 0.9, 0.05], [0.03, 0.0, 1.1]]);
    let fixed = model.internal_variables_fixed().to_vec();
    let root = NewtonRaphson::default().root(
        |internal_variables: &DeformationGradient2| {
            Ok(model.internal_variables_residual(&deformation_gradient, internal_variables)?)
        },
        |internal_variables: &DeformationGradient2| {
            Ok(model.tangents(&deformation_gradient, internal_variables)?.3)
        },
        model.internal_variables_initial(),
        vec![EqualityConstraint::Fixed(fixed.clone())],
        None,
    )?;
    let residual = model.internal_variables_residual(&deformation_gradient, &root)?;
    let initial = model.internal_variables_initial();
    (0..9).try_for_each(|index| {
        if fixed.contains(&index) {
            Assert::default()
                .eq_within_tols(root[index / 3][index % 3], &initial[index / 3][index % 3])
        } else {
            Assert::default().zero_within_tols(&residual[index / 3][index % 3])
        }
    })
}

const STRETCH_2: Scalar = 1.2;

#[test]
fn root_biaxial() -> Result<(), AssertionError> {
    use crate::constitutive::solid::elastic::internal_variables::{ElasticIV, Root};
    let (f, f_2) = model().root(
        AppliedLoad::BiaxialStress(STRETCH, STRETCH_2),
        NewtonRaphson::default(),
        SolveStrategy::Monolithic { elimination: false },
    )?;
    Assert::default().zero_within_tols(&model().internal_variables_residual(&f, &f_2)?)?;
    Assert::default().eq_within_tols(
        Vector::from([
            f[0][0].value(),
            f[0][1].value(),
            f[0][2].value(),
            f[1][2].value(),
            f[1][1].value(),
        ]),
        &Vector::from([STRETCH, 0.0, 0.0, 0.0, STRETCH_2]),
    )
}
