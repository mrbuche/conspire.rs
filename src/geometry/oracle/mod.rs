use crate::{geometry::Coordinate, math::Quantity, units::Length};

/// A region of space known through its signed distance, negative inside.
pub trait Oracle {
    /// Signed distance from the point to the boundary.
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length>;
    /// Closest point on the boundary to the point.
    fn project(&self, point: &Coordinate<3>) -> Coordinate<3>;
}
