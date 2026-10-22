use super::Load;
use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::plastic::{
            Linear, PlasticFlow, PlasticStateVariables, RateIndependentPlastic, VonMises,
        },
        solid::{elastic_plastic::AppliedLoad, hyperelastic::NeoHookean},
    },
    math::{
        Quantity, Tensor, TensorArray,
        optimize::{NewtonRaphson, SolveStrategy},
    },
    mechanics::{DeformationGradient, DeformationGradientPlastic},
    units::{Stress, Time},
};

type Model = Canonical<NeoHookean, PlasticFlow<VonMises, Linear>>;

fn model() -> Model {
    Canonical::from((
        NeoHookean {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        PlasticFlow {
            surface: VonMises,
            hardening: Linear {
                yield_stress: Stress::pascals(2.0),
                hardening_slope: Stress::pascals(1.0),
            },
        },
    ))
}

fn ramp(t: Quantity<Time>) -> f64 {
    1.0 + t.value()
}

fn advance(
    load: &mut Load,
    time: f64,
) -> Result<(DeformationGradient, PlasticStateVariables), ConstitutiveError> {
    let model = model();
    load.advance(
        &model,
        &NewtonRaphson::default(),
        &SolveStrategy::Condensed(NewtonRaphson::default()),
        Quantity::new(time),
        &DeformationGradient::identity(),
        &model.initial_state(),
    )
}

fn strain(state: &PlasticStateVariables) -> f64 {
    let (_, &strain): (&DeformationGradientPlastic, &Quantity) = state.into();
    strain.value()
}

#[test]
fn a_uniaxial_load_holds_one_component_and_a_biaxial_load_two() {
    let times = [0.0, 1.0].map(Quantity::<Time>::new);
    let (uniaxial, span) = Load::new(AppliedLoad::UniaxialStress(ramp, &times));
    assert_eq!(uniaxial.prescribed.len(), 1);
    assert_eq!(uniaxial.vector.len(), 4);
    assert_eq!(span, times);
    let (biaxial, _) = Load::new(AppliedLoad::BiaxialStress(ramp, ramp, &times));
    assert_eq!(biaxial.prescribed.len(), 2);
    assert_eq!(biaxial.vector.len(), 5);
}

#[test]
fn the_local_constraint_is_empty() {
    let times = [0.0, 1.0].map(Quantity::<Time>::new);
    let (load, _) = Load::new(AppliedLoad::UniaxialStress(ramp, &times));
    assert_eq!(load.constraint.1.len(), 0);
}

#[test]
fn advancing_sets_the_prescribed_values_at_the_time() -> Result<(), ConstitutiveError> {
    let times = [0.0, 1.0].map(Quantity::<Time>::new);
    let (mut load, _) = Load::new(AppliedLoad::UniaxialStress(ramp, &times));
    let (f, _) = advance(&mut load, 0.04)?;
    assert_eq!(load.vector[0], 1.04);
    assert!((f[0][0].value() - 1.04).abs() < 1e-10);
    Ok(())
}

#[test]
fn advancing_below_yield_leaves_the_plastic_state_alone() -> Result<(), ConstitutiveError> {
    let times = [0.0, 1.0].map(Quantity::<Time>::new);
    let (mut load, _) = Load::new(AppliedLoad::UniaxialStress(ramp, &times));
    let (_, state) = advance(&mut load, 0.04)?;
    assert_eq!(strain(&state), 0.0);
    Ok(())
}

#[test]
fn advancing_past_yield_accumulates_plastic_strain() -> Result<(), ConstitutiveError> {
    let times = [0.0, 1.0].map(Quantity::<Time>::new);
    let (mut load, _) = Load::new(AppliedLoad::UniaxialStress(ramp, &times));
    let (_, state) = advance(&mut load, 0.8)?;
    assert!(strain(&state) > 0.0);
    Ok(())
}
