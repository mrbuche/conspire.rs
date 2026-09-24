use super::YieldSurface;
use crate::{
    constitutive::ConstitutiveError,
    math::{Quantity, Tensor, TensorArray},
    mechanics::{FlowDirectionPlastic, MandelStressElastic, Scalar, StretchingRatePlastic},
    units::{Dissipation, Stress},
};
use std::array::from_fn;

/// The von Mises yield surface.
///
/// ```math
/// \phi(\mathbf{M}_\mathrm{e}') = |\mathbf{M}_\mathrm{e}'|
/// ```
#[derive(Clone, Debug)]
pub struct VonMises;

impl YieldSurface for VonMises {
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(deviatoric_mandel_stress.norm())
    }
    /// ```math
    /// \mathbf{N} = \frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm();
        if magnitude.is_zero() {
            Ok(FlowDirectionPlastic::zero())
        } else {
            Ok(deviatoric_mandel_stress / magnitude)
        }
    }
    /// ```math
    /// \frac{\partial\mathbf{N}}{\partial\mathbf{M}_\mathrm{e}'}:\mathrm{d}\mathbf{M}_\mathrm{e}'
    /// = \frac{1}{|\mathbf{M}_\mathrm{e}'|}\left(\mathrm{d}\mathbf{M}_\mathrm{e}' - \mathbf{N}\,(\mathbf{N}:\mathrm{d}\mathbf{M}_\mathrm{e}')\right)
    /// ```
    fn flow_direction_slope(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        increment: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm().value();
        if magnitude == 0.0 {
            return Ok(FlowDirectionPlastic::zero());
        }
        let direction: [[Scalar; 3]; 3] =
            from_fn(|i| from_fn(|j| deviatoric_mandel_stress[i][j].value() / magnitude));
        let slope: Scalar = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| direction[i][j] * increment[i][j].value())
                    .sum::<Scalar>()
            })
            .sum();
        Ok(FlowDirectionPlastic::from(from_fn::<_, 3, _>(|i| {
            from_fn::<_, 3, _>(|j| (increment[i][j].value() - direction[i][j] * slope) / magnitude)
        })))
    }
    /// ```math
    /// \phi_\mathrm{d}(\mathbf{D}_\mathrm{p}) = Y\,|\mathbf{D}_\mathrm{p}|
    /// ```
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        Ok(yield_stress * plastic_stretching_rate.norm())
    }
}
