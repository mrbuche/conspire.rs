mod more_sorensen;

use crate::math::{Norm, Scalar};
use more_sorensen::MORE_SORENSEN_TOLERANCE;
pub(crate) use more_sorensen::{more_sorensen_dense, more_sorensen_sparse};

/// How far a step is trusted to follow the model it was built from.
#[derive(Clone, Copy, Debug, Default)]
pub enum TrustRegion {
    /// The step grows or shrinks based on model predication.
    Adaptive { radius: Scalar, max_radius: Scalar },
    /// The step is shortened to a radius that never adapts.
    Fixed { radius: Scalar, norm: Norm },
    /// The step is taken whole, however far the model carries it.
    #[default]
    None,
}

pub(crate) const TRUST_REGION_ETA: Scalar = 0.1;
const TRUST_REGION_SHRINK_THRESHOLD: Scalar = 0.25;
const TRUST_REGION_GROW_THRESHOLD: Scalar = 0.75;
pub(crate) const TRUST_REGION_SHRINK: Scalar = 0.25;
const TRUST_REGION_GROW: Scalar = 2.0;
pub(crate) const TRUST_REGION_MAX_REJECTIONS: usize = 25;

pub(crate) fn update_radius(
    radius: Scalar,
    max_radius: Scalar,
    rho: Scalar,
    primal_step_norm: Scalar,
) -> Scalar {
    if rho < TRUST_REGION_SHRINK_THRESHOLD {
        TRUST_REGION_SHRINK * primal_step_norm
    } else if rho > TRUST_REGION_GROW_THRESHOLD
        && primal_step_norm >= radius * (1.0 - MORE_SORENSEN_TOLERANCE)
    {
        (TRUST_REGION_GROW * radius).min(max_radius)
    } else {
        radius
    }
}
