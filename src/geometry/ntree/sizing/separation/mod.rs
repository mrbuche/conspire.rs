use crate::{
    math::{Quantity, Scalar},
    units::Length,
};

/// Parameters controlling refinement driven by narrow features: places where
/// two creases run close in space but are not part of the same corner or
/// smoothly-continuing edge (e.g. the two edges bounding a thin rib, wall, or
/// stair-step). Unlike thickness (SDF) and curvature, this catches features
/// that are narrow as a *surface* patch without the solid on either side
/// being thin at all, and without needing any continuous curvature to key
/// off of.
pub struct SeparationSizing {
    /// The maximum crease-to-crease distance considered a narrow feature.
    /// `None` disables separation-driven refinement entirely.
    pub radius: Option<Quantity<Length>>,
    /// Creases within this many steps of one another along the crease graph
    /// are not compared, so a crease is not mistaken for lying close to its
    /// own smoothly-continuing neighbors (e.g. the other edges meeting it at
    /// a corner).
    pub hops: usize,
    /// Cells-per-gap for separation-driven refinement, independent of
    /// `scale` (which still controls cells-per-thickness for the SDF term).
    /// `None` falls back to `scale`, matching prior behavior.
    pub scale: Option<Scalar>,
}

impl Default for SeparationSizing {
    fn default() -> Self {
        Self {
            radius: None,
            hops: 1,
            scale: None,
        }
    }
}
