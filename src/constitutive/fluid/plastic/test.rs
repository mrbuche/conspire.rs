use super::*;
use crate::{
    EPSILON,
    math::assert::{Assert, AssertionError, perturbation},
    math::{ContractWith, Tensor, TensorArray},
};

fn linear() -> Linear {
    Linear {
        yield_stress: Stress::pascals(2.0),
        hardening_slope: Stress::pascals(1.0),
    }
}

fn model() -> PlasticFlow<VonMises, Linear> {
    PlasticFlow {
        surface: VonMises,
        hardening: linear(),
    }
}

fn deviatoric_mandel_stress() -> MandelStressElastic {
    MandelStressElastic::from([[1.3, 0.7, -0.4], [0.7, -0.9, 1.1], [-0.4, 1.1, -0.4]])
}

#[test]
fn flow_direction_has_unit_norm() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        model().flow_direction(&deviatoric_mandel_stress())?.norm(),
        &Quantity::new(1.0),
    )
}

#[test]
fn flow_direction_is_zero_at_zero_stress() -> Result<(), AssertionError> {
    Assert::zero(&model().flow_direction(&MandelStressElastic::zero())?)
}

#[test]
fn yield_function_reduces_to_yield_stress_offset() -> Result<(), AssertionError> {
    let model = model();
    let stress = deviatoric_mandel_stress();
    Assert::default().eq_within_tols(
        model.yield_function(&stress, Quantity::default())?,
        &(stress.norm() - model.initial_yield_stress()),
    )
}

#[test]
fn yield_function_hardens_with_equivalent_plastic_strain() -> Result<(), AssertionError> {
    let model = model();
    let stress = deviatoric_mandel_stress();
    Assert::default().eq_within_tols(
        model.yield_function(&stress, Quantity::new(1.0))?,
        &(model.yield_function(&stress, Quantity::default())? - model.hardening_slope()),
    )
}

#[test]
fn plastic_stretching_rate_is_multiplier_times_flow_direction() -> Result<(), AssertionError> {
    let model = model();
    let stress = deviatoric_mandel_stress();
    let multiplier = Rate::per_second(0.3);
    Assert::default().eq_within_tols(
        &model.plastic_stretching_rate(&stress, multiplier)?,
        &(model.flow_direction(&stress)? * multiplier),
    )
}

#[test]
fn deviatoric_mandel_stress_from_finite_difference_of_dissipation_potential()
-> Result<(), AssertionError> {
    let model = model();
    let flow_direction = model.flow_direction(&deviatoric_mandel_stress())?;
    let yield_stress = model.initial_yield_stress();
    let plastic_stretching_rate = flow_direction.clone() * Rate::per_second(0.3);
    let mandel_stress_on_surface = flow_direction * yield_stress;
    let mut finite_difference = MandelStressElastic::zero();
    for i in 0..3 {
        for j in 0..3 {
            let mut plus = plastic_stretching_rate.clone();
            plus[i][j] += perturbation(0.5 * EPSILON);
            let mut minus = plastic_stretching_rate.clone();
            minus[i][j] -= perturbation(0.5 * EPSILON);
            finite_difference[i][j] = (model.dissipation_potential(plus, yield_stress)?
                - model.dissipation_potential(minus, yield_stress)?)
                / Quantity::<Rate>::new(EPSILON);
        }
    }
    Assert::default().eq_within_fd_tol(&mandel_stress_on_surface, &finite_difference)
}

fn voce() -> PlasticFlow<VonMises, Voce> {
    PlasticFlow {
        surface: VonMises,
        hardening: Voce {
            yield_stress: Stress::pascals(2.0),
            hardening_slope: Stress::pascals(0.2),
            saturation_stress: Stress::pascals(1.5),
            saturation_rate: 8.0,
        },
    }
}

#[test]
fn the_default_hardening_modulus_is_the_slope() -> Result<(), AssertionError> {
    let model = model();
    [0.0, 0.3, 2.0].into_iter().try_for_each(|strain| {
        Assert::default().eq_within_tols(
            model.hardening_modulus(Quantity::new(strain))?,
            &model.hardening_slope(),
        )
    })
}

#[test]
fn hardening_modulus_is_the_derivative_of_the_yield_stress() -> Result<(), AssertionError> {
    let (model, step) = (voce(), 1e-6);
    for strain in [0.0, 0.01, 0.05, 0.3, 1.0] {
        let finite_difference = (model.yield_stress(Quantity::new(strain + step))?
            - model.yield_stress(Quantity::new(strain - step))?)
            / (2.0 * step);
        let modulus = model.hardening_modulus(Quantity::new(strain))?;
        assert!(
            (modulus.value() - finite_difference.value()).abs()
                <= 1e-6 * (1.0 + modulus.value().abs()),
            "strain {strain}: modulus {} vs finite difference {}",
            modulus.value(),
            finite_difference.value(),
        );
    }
    Ok(())
}

#[test]
fn voce_hardening_starts_at_the_initial_yield_stress_and_saturates() -> Result<(), AssertionError> {
    let model = voce();
    Assert::default().eq_within_tols(
        model.yield_stress(Quantity::default())?,
        &model.initial_yield_stress(),
    )?;
    // the initial slope is H + Q b, and past saturation only the linear part H remains
    Assert::default().eq_within_tols(
        model.hardening_modulus(Quantity::default())?,
        &model.hardening_slope(),
    )?;
    Assert::default().eq_within_tols(
        model.hardening_slope(),
        &(Stress::pascals(0.2) + Stress::pascals(1.5) * 8.0),
    )?;
    Assert::default().eq_within_tols(
        model.hardening_modulus(Quantity::new(10.0))?,
        &Stress::pascals(0.2),
    )
}

#[test]
fn fenchel_equality() -> Result<(), AssertionError> {
    let model = model();
    let flow_direction = model.flow_direction(&deviatoric_mandel_stress())?;
    let yield_stress = model.initial_yield_stress();
    let plastic_stretching_rate = flow_direction.clone() * Rate::per_second(0.3);
    let mandel_stress_on_surface = flow_direction * yield_stress;
    Assert::default().eq_within_tols(
        model.dissipation_potential(plastic_stretching_rate.clone(), yield_stress)?,
        &ContractWith::contract_with(&mandel_stress_on_surface, &plastic_stretching_rate),
    )
}

fn hill() -> PlasticFlow<Hill, Linear> {
    PlasticFlow {
        surface: Hill {
            f: 0.4,
            g: 0.25,
            h: 0.3,
            l: 1.3,
            m: 0.8,
            n: 1.1,
        },
        hardening: linear(),
    }
}

fn isotropic_hill() -> PlasticFlow<Hill, Linear> {
    PlasticFlow {
        surface: Hill {
            f: 1.0 / 3.0,
            g: 1.0 / 3.0,
            h: 1.0 / 3.0,
            l: 1.0,
            m: 1.0,
            n: 1.0,
        },
        hardening: linear(),
    }
}

fn stress_increment() -> MandelStressElastic {
    MandelStressElastic::from([[0.3, -0.2, 0.5], [-0.2, 0.4, 0.1], [0.5, 0.1, -0.7]])
}

fn assert_flow_direction_is_the_gradient_of_the_equivalent_stress(
    model: &impl RateIndependentPlastic,
) -> Result<(), AssertionError> {
    let stress = deviatoric_mandel_stress();
    let mut finite_difference = FlowDirectionPlastic::zero();
    for i in 0..3 {
        for j in 0..3 {
            let mut plus = stress.clone();
            plus[i][j] += perturbation(0.5 * EPSILON);
            let mut minus = stress.clone();
            minus[i][j] -= perturbation(0.5 * EPSILON);
            finite_difference[i][j] = (model.equivalent_stress(&plus)?
                - model.equivalent_stress(&minus)?)
                / Quantity::<Stress>::new(EPSILON);
        }
    }
    Assert::default().eq_within_fd_tol(&model.flow_direction(&stress)?, &finite_difference)
}

fn assert_slope_matches_finite_difference(
    model: &impl RateIndependentPlastic,
) -> Result<(), AssertionError> {
    let (stress, increment) = (deviatoric_mandel_stress(), stress_increment());
    let step = 1e-6;
    let direction_at =
        |sign: f64| model.flow_direction(&(stress.clone() + increment.clone() * (sign * step)));
    let finite_difference = (direction_at(1.0)? - direction_at(-1.0)?) / (2.0 * step);
    Assert {
        abs_tol: 1e-6,
        rel_tol: 1e-6,
        ..Default::default()
    }
    .eq_within_tols(
        &model.flow_direction_slope(&stress, &increment)?,
        &finite_difference,
    )
}

#[test]
fn von_mises_flow_direction_is_the_gradient_of_the_equivalent_stress() -> Result<(), AssertionError>
{
    assert_flow_direction_is_the_gradient_of_the_equivalent_stress(&model())
}

#[test]
fn hill_flow_direction_is_the_gradient_of_the_equivalent_stress() -> Result<(), AssertionError> {
    assert_flow_direction_is_the_gradient_of_the_equivalent_stress(&hill())
}

#[test]
fn von_mises_flow_direction_slope_matches_finite_difference() -> Result<(), AssertionError> {
    assert_slope_matches_finite_difference(&model())
}

#[test]
fn hill_flow_direction_slope_matches_finite_difference() -> Result<(), AssertionError> {
    assert_slope_matches_finite_difference(&hill())
}

#[test]
fn an_isotropic_hill_model_is_von_mises() -> Result<(), AssertionError> {
    let (hill, mises) = (isotropic_hill(), model());
    let (stress, increment) = (deviatoric_mandel_stress(), stress_increment());
    let yield_stress = mises.initial_yield_stress();
    let assert = Assert::default();
    assert.eq_within_tols(
        hill.equivalent_stress(&stress)?,
        &mises.equivalent_stress(&stress)?,
    )?;
    assert.eq_within_tols(
        &hill.flow_direction(&stress)?,
        &mises.flow_direction(&stress)?,
    )?;
    assert.eq_within_tols(
        &hill.flow_direction_slope(&stress, &increment)?,
        &mises.flow_direction_slope(&stress, &increment)?,
    )?;
    let rate = mises.flow_direction(&stress)? * Rate::per_second(0.3);
    assert.eq_within_tols(
        hill.dissipation_potential(rate.clone(), yield_stress)?,
        &mises.dissipation_potential(rate, yield_stress)?,
    )
}

#[test]
fn hill_is_anisotropic_and_its_flow_is_deviatoric() -> Result<(), AssertionError> {
    let (hill, mises) = (hill(), model());
    let stress = deviatoric_mandel_stress();
    assert!(
        (hill.equivalent_stress(&stress)? - mises.equivalent_stress(&stress)?)
            .value()
            .abs()
            > 1e-2,
        "the coefficients must move the surface for this test to mean anything"
    );
    let direction = hill.flow_direction(&stress)?;
    Assert::default().eq_within_tols((0..3).map(|i| direction[i][i].value()).sum::<f64>(), &0.0)
}

#[test]
fn hill_equivalent_stress_is_homogeneous_of_degree_one() -> Result<(), AssertionError> {
    let hill = hill();
    let stress = deviatoric_mandel_stress();
    let scaled = stress.clone() * 2.5;
    Assert::default().eq_within_tols(
        hill.equivalent_stress(&scaled)?,
        &(hill.equivalent_stress(&stress)? * 2.5),
    )?;
    Assert::default().eq_within_tols(
        hill.equivalent_stress(&stress)?,
        &ContractWith::contract_with(&stress, &hill.flow_direction(&stress)?),
    )
}

fn hill_stress_on_the_surface(
    model: &PlasticFlow<Hill, Linear>,
) -> Result<MandelStressElastic, AssertionError> {
    let stress = deviatoric_mandel_stress();
    let scale = model.initial_yield_stress().value() / model.equivalent_stress(&stress)?.value();
    Ok(stress * scale)
}

#[test]
fn hill_dissipation_potential_is_the_yield_stress_times_the_multiplier()
-> Result<(), AssertionError> {
    let model = hill();
    let yield_stress = model.initial_yield_stress();
    let rate = model.flow_direction(&deviatoric_mandel_stress())? * Rate::per_second(0.3);
    Assert::default().eq_within_tols(
        model.dissipation_potential(rate, yield_stress)?,
        &(yield_stress * Quantity::<Rate>::new(0.3)),
    )
}

#[test]
fn hill_deviatoric_mandel_stress_from_finite_difference_of_dissipation_potential()
-> Result<(), AssertionError> {
    let model = hill();
    let yield_stress = model.initial_yield_stress();
    let plastic_stretching_rate =
        model.flow_direction(&deviatoric_mandel_stress())? * Rate::per_second(0.3);
    let mut finite_difference = MandelStressElastic::zero();
    for i in 0..3 {
        for j in 0..3 {
            let mut plus = plastic_stretching_rate.clone();
            plus[i][j] += perturbation(0.5 * EPSILON);
            let mut minus = plastic_stretching_rate.clone();
            minus[i][j] -= perturbation(0.5 * EPSILON);
            finite_difference[i][j] = (model.dissipation_potential(plus, yield_stress)?
                - model.dissipation_potential(minus, yield_stress)?)
                / Quantity::<Rate>::new(EPSILON);
        }
    }
    Assert::default().eq_within_fd_tol(&hill_stress_on_the_surface(&model)?, &finite_difference)
}

#[test]
fn hill_fenchel_equality() -> Result<(), AssertionError> {
    let model = hill();
    let yield_stress = model.initial_yield_stress();
    let plastic_stretching_rate =
        model.flow_direction(&deviatoric_mandel_stress())? * Rate::per_second(0.3);
    Assert::default().eq_within_tols(
        model.dissipation_potential(plastic_stretching_rate.clone(), yield_stress)?,
        &ContractWith::contract_with(
            &hill_stress_on_the_surface(&model)?,
            &plastic_stretching_rate,
        ),
    )
}
