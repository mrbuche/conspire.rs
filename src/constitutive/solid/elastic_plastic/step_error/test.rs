use crate::math::integrate::StepError;
use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::plastic::{
            PlasticFlow, PlasticStateVariables, RateIndependentPlastic, Voce, VonMises,
        },
        solid::{
            elastic_plastic::ElasticPlastic,
            hyperelastic::{Hencky, NeoHookean},
        },
    },
    math::{Quantity, TensorArray},
    mechanics::{DeformationGradient, DeformationGradientPlastic},
    units::Stress,
};

fn flow() -> PlasticFlow<VonMises, Voce> {
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

fn uniaxial(stretch: f64) -> DeformationGradient {
    let transverse = stretch.powf(-0.5);
    [
        [stretch, 0.0, 0.0],
        [0.0, transverse, 0.0],
        [0.0, 0.0, transverse],
    ]
    .into()
}

fn sheared(stretch: f64, shear: f64) -> DeformationGradient {
    let transverse = stretch.powf(-0.5);
    [
        [stretch, shear, 0.0],
        [0.0, transverse, 0.0],
        [0.0, 0.0, transverse],
    ]
    .into()
}

fn turning(h: f64) -> DeformationGradient {
    sheared(1.5 + 0.5 * h, 0.6 * h)
}

fn strain(state: &PlasticStateVariables) -> f64 {
    let (_, &strain): (&DeformationGradientPlastic, &Quantity) = state.into();
    strain.value()
}

macro_rules! test_models {
    ($elastic:ident) => {
        use super::*;
        fn model() -> Canonical<$elastic, PlasticFlow<VonMises, Voce>> {
            (
                $elastic {
                    bulk_modulus: Stress::pascals(13.0),
                    shear_modulus: Stress::pascals(3.0),
                },
                flow(),
            ).into()
        }
        fn step(
            old: (&DeformationGradient, &PlasticStateVariables),
            f: &DeformationGradient,
        ) -> Result<(StepError, PlasticStateVariables), ConstitutiveError> {
            let model = model();
            let state = model.return_map(f, old.1)?;
            Ok((model.step_error(old.0, f, old.1, &state)?, state))
        }
        fn yielding() -> Result<(DeformationGradient, PlasticStateVariables), ConstitutiveError> {
            let f = uniaxial(1.5);
            let (_, state) = step(
                (&DeformationGradient::identity(), &model().initial_state()),
                &f,
            )?;
            Ok((f, state))
        }
        #[test]
        fn elastic_step_has_no_error() -> Result<(), ConstitutiveError> {
            let (error, state) = step(
                (&DeformationGradient::identity(), &model().initial_state()),
                &uniaxial(1.05),
            )?;
            assert_eq!(strain(&state), 0.0);
            assert_eq!(error.direction.value(), 0.0);
            assert_eq!(error.overshoot.value(), 0.0);
            Ok(())
        }
        #[test]
        fn proportional_loading_has_no_direction_error_at_any_step() -> Result<(), ConstitutiveError>
        {
            let (f, state) = yielding()?;
            for stretch in [1.6, 2.0, 3.0, 6.0] {
                let (error, next) = step((&f, &state), &uniaxial(stretch))?;
                assert!(strain(&next) > strain(&state));
                assert!(error.direction.value() < 1e-12, "{stretch}: {error:?}");
            }
            Ok(())
        }
        #[test]
        fn turning_load_has_a_direction_error_that_falls_with_the_square_of_the_step()
        -> Result<(), ConstitutiveError> {
            let (f, state) = yielding()?;
            let directions = [1.0, 0.5, 0.25, 0.125]
                .map(|h| step((&f, &state), &turning(h)).map(|(error, _)| error.direction.value()));
            let directions = directions.into_iter().collect::<Result<Vec<_>, _>>()?;
            assert!(directions[3] > 1e-4, "{directions:?}");
            for pair in directions.windows(2) {
                let ratio = pair[0] / pair[1];
                assert!((3.0..5.0).contains(&ratio), "{ratio}: {directions:?}");
            }
            Ok(())
        }
        #[test]
        fn first_plastic_step_overshoots_and_a_continued_one_does_not() -> Result<(), ConstitutiveError>
        {
            let model = model();
            let rest = (&DeformationGradient::identity(), &model.initial_state());
            let (error, state) = step(rest, &uniaxial(1.5))?;
            assert_eq!(error.overshoot.value(), strain(&state));
            assert!(error.overshoot.value() > 0.0);
            let (error, _) = step((&uniaxial(1.5), &state), &uniaxial(2.0))?;
            assert_eq!(error.overshoot.value(), 0.0);
            Ok(())
        }
        #[test]
        fn overshoot_falls_as_the_step_ends_nearer_the_onset() -> Result<(), ConstitutiveError> {
            let rest = (&DeformationGradient::identity(), &model().initial_state());
            let near = step(rest, &uniaxial(1.42))?.0.overshoot.value();
            let far = step(rest, &uniaxial(1.5))?.0.overshoot.value();
            assert!(0.0 < near && near < far, "{near} {far}");
            Ok(())
        }
        #[test]
        fn yielding_again_after_unloading_is_an_onset() -> Result<(), ConstitutiveError> {
            let (f, state) = yielding()?;
            let unloaded = uniaxial(1.2);
            let (error, state_unloaded) = step((&f, &state), &unloaded)?;
            assert_eq!(strain(&state_unloaded), strain(&state));
            assert_eq!(error.overshoot.value(), 0.0);
            let (error, _) = step((&unloaded, &state_unloaded), &uniaxial(2.0))?;
            assert!(error.overshoot.value() > 0.0, "{error:?}");
            Ok(())
        }
    };
}

mod hencky {
    test_models!(Hencky);
}

mod neo_hookean {
    test_models!(NeoHookean);
}

#[test]
fn the_estimate_is_the_local_error_of_the_step_as_it_shrinks() -> Result<(), ConstitutiveError> {
    use crate::math::Tensor;
    let model: Canonical<NeoHookean, PlasticFlow<VonMises, Voce>> = Canonical::from((
        NeoHookean {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        flow(),
    ));
    let f0 = uniaxial(1.5);
    let s0 = model.return_map(&f0, &model.initial_state())?;
    let plastic = |state: &PlasticStateVariables| -> DeformationGradientPlastic {
        let (f_p, _): (&DeformationGradientPlastic, &Quantity) = state.into();
        f_p.clone()
    };
    let f_p_0 = plastic(&s0);
    let ratios = [0.25, 0.125, 0.0625, 0.03125]
        .into_iter()
        .map(|h| {
            let f = turning(h);
            let step = model.return_map(&f, &s0)?;
            let estimate = model.step_error(&f0, &f, &s0, &step)?.direction.value();
            let substeps = 128;
            let mut reference = s0.clone();
            for k in 1..=substeps {
                reference =
                    model.return_map(&turning(h * k as f64 / substeps as f64), &reference)?;
            }
            let error = ((plastic(&step) - plastic(&reference)) * f_p_0.inverse())
                .norm()
                .value();
            Ok(error / estimate)
        })
        .collect::<Result<Vec<_>, ConstitutiveError>>()?;
    ratios
        .windows(2)
        .for_each(|pair| assert!(pair[0] < pair[1], "{ratios:?}"));
    assert!(ratios.iter().all(|&ratio| ratio < 1.02), "{ratios:?}");
    assert!(ratios[3] > 0.92, "{ratios:?}");
    Ok(())
}
