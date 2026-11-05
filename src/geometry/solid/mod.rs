mod partition;

pub use partition::Partition;

use crate::{
    geometry::{Coordinate, Direction},
    math::{Quantity, Tensor},
    units::Length,
};

/// A closest boundary point of a solid, with the outward unit normal there.
pub struct Projection {
    /// A point on the boundary of the solid.
    pub point: Coordinate<3>,
    /// The outward unit normal of the boundary at the point.
    pub normal: Direction<3>,
    /// Whether the point is provably the closest boundary point, not only one on the boundary.
    pub exact: bool,
}

/// A solid known through its signed distance, negative inside.
pub trait SolidOracle {
    /// Signed distance from the point to the boundary, negative inside the solid.
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length>;
    /// A boundary point near the point, and the outward unit normal there.
    fn project(&self, point: &Coordinate<3>) -> Option<Projection>;
}

pub(crate) type Candidate = Option<(Projection, bool)>;

/// The nearest candidate that survives, which is exact when the operands are and no candidate that
/// does not survive is nearer.
///
/// The distance of each candidate is found once and compared as found, so that the nearest of them
/// is never judged farther than itself by a distance that rounds another way.
pub(crate) fn nearest_surviving(
    query: &Coordinate<3>,
    candidates: impl IntoIterator<Item = Candidate>,
) -> Option<Projection> {
    let candidates: Vec<(Projection, bool, Quantity<Length>)> = candidates
        .into_iter()
        .flatten()
        .map(|(projection, survives)| {
            let distance = (&projection.point - query).norm();
            (projection, survives, distance)
        })
        .collect();
    let nearest = candidates
        .iter()
        .map(|(.., distance)| *distance)
        .min_by(|a, b| a.total_cmp(b))?;
    let operands_exact = candidates.iter().all(|(projection, ..)| projection.exact);
    candidates
        .into_iter()
        .filter(|(_, survives, _)| *survives)
        .min_by(|(.., a), (.., b)| a.total_cmp(b))
        .map(|(projection, _, distance)| Projection {
            exact: operands_exact && distance <= nearest,
            ..projection
        })
}
