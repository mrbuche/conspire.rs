use super::super::Scalar;

/// How far a step is trusted to follow the model it was built from.
#[derive(Clone, Copy, Debug, Default)]
pub enum TrustRegion {
    /// The step is shortened to a radius that never adapts.
    Fixed(Scalar),
    /// The step is taken whole, however far the model carries it.
    #[default]
    None,
}
