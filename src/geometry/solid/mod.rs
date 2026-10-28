mod partition;

pub use partition::Partition;

use crate::{
    geometry::{Coordinate, Direction},
    math::Quantity,
    units::Length,
};

/// A solid known through its signed distance, negative inside.
pub trait SolidOracle {
    /// Signed distance from the point to the boundary, negative inside the solid.
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length>;
    /// The closest boundary point to the point and the outward unit normal there.
    fn project(&self, point: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)>;
}
