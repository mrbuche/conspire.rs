#[cfg(test)]
mod test;

use super::ElasticPlastic;
use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::PlasticStateVariables},
    math::{Quantity, Rank2, Tensor},
    mechanics::{DeformationGradient, FlowDirectionPlastic, Scalar},
};

const ONSET_TOLERANCE: Scalar = 1e-6;

/// The error indicators of one plastic step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepError {
    /// The change of the flow direction over the step, weighted by the plastic increment.
    ///
    /// ```math
    /// \Delta\gamma\,\|\mathbf{N}^{n+1} - \mathbf{N}^{n}\|
    /// ```
    ///
    /// The directions are those at the two ends of the step, so it measures the turning
    /// of the load path as well as of the flow. A direction frozen at the trial state
    /// instead would miss the turning of the load, and vanishes identically for a
    /// radial return such as Hencky with a von Mises surface.
    ///
    /// It vanishes when the direction is constant, as under proportional loading, however
    /// large the step, and falls as the square of the step otherwise.
    pub direction: Quantity,
    /// The plastic increment $`\Delta\gamma`$ of a step that begins inside the yield
    /// surface, how far the step runs past the onset of yielding, and zero otherwise.
    pub overshoot: Quantity,
}

impl StepError {
    /// The larger of each of the two, as for the points of a mesh.
    pub fn max(self, other: Self) -> Self {
        Self {
            direction: self.direction.max(other.direction),
            overshoot: self.overshoot.max(other.overshoot),
        }
    }
}

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
        * increment;
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
