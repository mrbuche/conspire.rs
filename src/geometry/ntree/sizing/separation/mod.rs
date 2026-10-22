use crate::{
    math::{Quantity, Scalar},
    units::Length,
};

/// Parameters controlling refinement driven by narrow features: places where
/// two creases run close in space but far apart along the creases themselves
/// (e.g. the two edges bounding a thin rib, wall, or stair-step, or the two
/// sides of a wall that thins to a point or ends in a curve). Unlike thickness
/// (SDF) and curvature, this catches features that are narrow as a *surface*
/// patch without the solid on either side being thin at all, and without
/// needing any continuous curvature to key off of.
#[derive(Default)]
pub struct SeparationSizing {
    /// The minimum crease-to-crease distance considered a narrow feature:
    /// creases closer together than this are taken for noise and ignored.
    /// `None` disables separation-driven refinement entirely, and zero
    /// enables it with nothing ignored. There is no maximum, so creases are
    /// compared with every other crease and the cost grows with the square of
    /// their number.
    pub minimum: Option<Quantity<Length>>,
    /// Cells-per-gap for separation-driven refinement, independent of
    /// `scale` (which still controls cells-per-thickness for the SDF term).
    /// `None` falls back to `scale`, matching prior behavior.
    pub scale: Option<Scalar>,
}

/// The ratio of the gap between two creases to the distance along the
/// creases from one to the other, at and below which they are fully a narrow
/// feature. Where two creases open from a point at an angle, the ratio is the
/// sine of half the angle, so this is about thirty degrees.
pub(crate) const THIN: Scalar = 0.25;

/// The ratio at and above which they are not one at all. This is about sixty
/// degrees, and the corner of a box is wider still.
pub(crate) const WIDE: Scalar = 0.5;

/// The least of a narrow feature that creases bound for them to ask anything
/// of a cell.
pub(crate) const NARROWEST: Scalar = 1.0e-3;

/// How much of a narrow feature two creases bound, given the ratio of their
/// gap to the distance along the creases between them: one when they run far
/// apart along the creases for how close they lie, zero when they lie about as
/// far apart as they run, and smooth between.
pub(crate) fn narrowness(ratio: Scalar) -> Scalar {
    let t = ((WIDE - ratio) / (WIDE - THIN)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Whether creases bound enough of a narrow feature, at a ratio of their gap to
/// the distance along the creases between them, to ask anything of a cell.
pub(crate) fn asks(ratio: Scalar) -> bool {
    narrowness(ratio) > NARROWEST
}
