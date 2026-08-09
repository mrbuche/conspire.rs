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
        optimize::{GradientDescent, LineSearch, NewtonRaphson, SolveStrategy, TrustRegion},
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
    use crate::constitutive::solid::hyperelastic::internal_variables::SecondOrderMinimize;
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
        Vector::from([f[0][0], f[0][1], f[0][2], f[1][2]]),
        &Vector::from([STRETCH, 0.0, 0.0, 0.0]),
    )
}

#[test]
fn minimize_first_order() -> Result<(), AssertionError> {
    use crate::constitutive::solid::hyperelastic::internal_variables::FirstOrderMinimize;
    let (_f, _f_2) = model().minimize(
        AppliedLoad::UniaxialStress(STRETCH),
        GradientDescent {
            dual: true,
            ..Default::default()
        },
    )?;
    Ok(())
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

fn line_search(name: &str) -> LineSearch {
    match name {
        "armijo" => LineSearch::Armijo {
            control: 1e-3,
            cut_back: 9e-1,
            max_steps: 100,
        },
        "goldstein" => LineSearch::Goldstein {
            control: 1e-4,
            cut_back: 5e-1,
            max_steps: 100,
        },
        "error" => LineSearch::Error {
            cut_back: 5e-1,
            max_steps: 100,
        },
        _ => LineSearch::Wolfe {
            control_1: 1e-3,
            control_2: 9e-1,
            cut_back: 5e-1,
            max_steps: 100,
            strong: true,
        },
    }
}

fn searched(name: &str, strategy: SolveStrategy) -> Result<(), AssertionError> {
    use crate::constitutive::solid::hyperelastic::internal_variables::SecondOrderMinimize;
    let (f, f_2) = model().minimize(
        AppliedLoad::UniaxialStress(STRETCH),
        NewtonRaphson {
            line_search: line_search(name),
            ..Default::default()
        },
        strategy.clone(),
    )?;
    check(&f, &f_2)?;
    let (f_none, f_2_none) = blocked(strategy)?;
    Assert::default().eq_within_tols(&f, &f_none)?;
    Assert::default().eq_within_tols(&f_2, &f_2_none)
}

#[test]
fn minimize_line_search() -> Result<(), AssertionError> {
    for name in ["armijo", "goldstein", "error"] {
        for strategy in [
            SolveStrategy::Monolithic { elimination: false },
            SolveStrategy::Monolithic { elimination: true },
            SolveStrategy::Condensed(NewtonRaphson::default()),
        ] {
            searched(name, strategy)?
        }
    }
    Ok(())
}

const STRETCH_FAR: Scalar = 6.0;

fn far(
    trust_region: TrustRegion,
    strategy: SolveStrategy,
) -> Result<(DeformationGradient, DeformationGradient2), AssertionError> {
    use crate::constitutive::solid::elastic::internal_variables::FirstOrderRoot;
    let (f, f_2) = model().root(
        AppliedLoad::UniaxialStress(STRETCH_FAR),
        NewtonRaphson {
            trust_region,
            max_steps: 200,
            ..Default::default()
        },
        strategy,
    )?;
    use crate::constitutive::solid::elastic::internal_variables::ElasticIV;
    Assert::default().zero_within_tols(&model().internal_variables_residual(&f, &f_2)?)?;
    Assert::default().eq_within_tols(
        Vector::from([f[0][0], f[0][1], f[0][2], f[1][2]]),
        &Vector::from([STRETCH_FAR, 0.0, 0.0, 0.0]),
    )?;
    Ok((f, f_2))
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
    use crate::constitutive::solid::elastic::internal_variables::FirstOrderRoot;
    for strategy in [
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
        SolveStrategy::Condensed(NewtonRaphson::default()),
    ] {
        let (f, f_2) = model().root(
            AppliedLoad::UniaxialStress(STRETCH),
            NewtonRaphson {
                line_search: line_search("error"),
                ..Default::default()
            },
            strategy,
        )?;
        check(&f, &f_2)?
    }
    Ok(())
}

#[test]
#[should_panic(expected = "gradient of the merit function")]
fn minimize_line_search_wolfe() {
    searched("wolfe", SolveStrategy::Monolithic { elimination: false }).unwrap();
}

#[test]
fn root_fixed_at_point() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic::internal_variables::ElasticIV,
        math::optimize::{EqualityConstraint, FirstOrderRootFinding},
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
        EqualityConstraint::Fixed(fixed.clone()),
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
    use crate::constitutive::solid::elastic::internal_variables::{ElasticIV, FirstOrderRoot};
    let (f, f_2) = model().root(
        AppliedLoad::BiaxialStress(STRETCH, STRETCH_2),
        NewtonRaphson::default(),
        SolveStrategy::Monolithic { elimination: false },
    )?;
    Assert::default().zero_within_tols(&model().internal_variables_residual(&f, &f_2)?)?;
    Assert::default().eq_within_tols(
        Vector::from([f[0][0], f[0][1], f[0][2], f[1][2], f[1][1]]),
        &Vector::from([STRETCH, 0.0, 0.0, 0.0, STRETCH_2]),
    )
}

#[test]
fn minimize_biaxial_first_order() -> Result<(), AssertionError> {
    use crate::constitutive::solid::hyperelastic::internal_variables::FirstOrderMinimize;
    let (f, _f_2) = model().minimize(
        AppliedLoad::BiaxialStress(STRETCH, STRETCH_2),
        GradientDescent {
            dual: true,
            ..Default::default()
        },
    )?;
    Assert::default().eq_within_tols(
        Vector::from([f[0][0], f[1][1]]),
        &Vector::from([STRETCH, STRETCH_2]),
    )
}
