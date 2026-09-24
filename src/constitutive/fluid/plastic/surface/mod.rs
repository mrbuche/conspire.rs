//! Yield surfaces for rate-independent plastic fluid constitutive models.

mod hill;
mod von_mises;

pub use hill::Hill;
pub use von_mises::VonMises;

use crate::{
    constitutive::ConstitutiveError,
    math::Quantity,
    mechanics::{FlowDirectionPlastic, MandelStressElastic, StretchingRatePlastic},
    units::{Dissipation, Stress},
};
use std::fmt::Debug;

/// Required methods for yield surfaces.
///
/// The yield surface is $`\phi(\mathbf{M}_\mathrm{e}') = Y(\varepsilon_\mathrm{p})`$, with
/// the equivalent stress $`\phi`$ positively homogeneous of degree one, smooth away from
/// the origin, and dependent on the deviatoric Mandel stress only, and the flow is
/// associative. The hardening law $`Y`$ is independent of the surface, and the two are
/// combined by [`PlasticFlow`](super::PlasticFlow).
pub trait YieldSurface
where
    Self: Clone + Debug,
{
    /// Calculates and returns the equivalent stress $`\phi(\mathbf{M}_\mathrm{e}')`$.
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError>;
    /// Calculates and returns the associative plastic flow direction.
    ///
    /// ```math
    /// \mathbf{N} = \frac{\partial\phi}{\partial\mathbf{M}_\mathrm{e}'}
    /// ```
    ///
    /// It is zero where the equivalent stress vanishes.
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError>;
    /// Calculates and returns the derivative of the flow direction along a deviatoric
    /// Mandel stress increment.
    ///
    /// ```math
    /// \frac{\partial\mathbf{N}}{\partial\mathbf{M}_\mathrm{e}'}:\mathrm{d}\mathbf{M}_\mathrm{e}'
    /// ```
    fn flow_direction_slope(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        increment: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError>;
    /// Calculates and returns the plastic dissipation potential, the dual gauge of the
    /// equivalent stress scaled by the yield stress.
    ///
    /// ```math
    /// \phi_\mathrm{d}(\mathbf{D}_\mathrm{p}) = Y\dot\gamma
    /// \quad\text{for}\quad \mathbf{D}_\mathrm{p} = \dot\gamma\,\mathbf{N}
    /// ```
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError>;
}
