#[cfg(test)]
mod test;

use super::ElasticPlastic;
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariables},
    math::{Quantity, Rank2, Tensor, integrate::StepError},
    mechanics::{DeformationGradient, FlowDirectionPlastic, Scalar},
};

const ONSET_TOLERANCE: Scalar = 1e-6;

pub(super) fn evaluate<C: ElasticPlastic>(
    model: &C,
    deformation_gradient_old: &DeformationGradient,
    deformation_gradient: &DeformationGradient,
    state_old: &PlasticStateVariables,
    state_new: &PlasticStateVariables,
) -> Result<StepError, ConstitutiveError> {
    let (f_p_old, &eqps_old) = state_old.into();
    let (f_p_new, eqps_new) = state_new.into();
    let increment = eqps_new - eqps_old;
    if increment <= Quantity::default() {
        return Ok(StepError {
            direction: Quantity::default(),
            overshoot: Quantity::default(),
        });
    }
    let direction_at = |f, f_p| -> Result<FlowDirectionPlastic, ConstitutiveError> {
        Ok(model
            .flow_direction(&model.mandel_stress(f, f_p)?.deviatoric())?
            .symmetric_part())
    };
    let direction = (direction_at(deformation_gradient, f_p_new)?
        - direction_at(deformation_gradient_old, f_p_old)?)
    .norm()
        * (increment * 0.5);
    let yield_function = model.yield_function(
        &model
            .mandel_stress(deformation_gradient_old, f_p_old)?
            .deviatoric(),
        eqps_old,
    )?;
    let overshoot = if yield_function < model.yield_stress(eqps_old)? * -ONSET_TOLERANCE {
        increment
    } else {
        Quantity::default()
    };
    Ok(StepError {
        direction,
        overshoot,
    })
}
