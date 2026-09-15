//! Exact 1D crease curves, extracted from B-rep topology rather than guessed
//! by nearest-face distance. A boundary mesh node genuinely on a sharp edge
//! should be pulled onto this curve, never left to pick between the two (or
//! more) faces that meet there — right at a crease those faces are
//! near-equidistant by definition, so a nearest-*face* choice is unstable
//! under the sub-cell node motion a fit sweep makes, and flips.
//!
//! This module only extracts the curves; querying them (nearest point, for a
//! fit to constrain a node onto) lives in [`crate::geometry::mesh::buffer::fit`]
//! as a plain-polyline utility — `mesh` must not depend on `cad`.

#[cfg(test)]
mod test;

use super::{Brep, D, curve::chords};
use crate::geometry::Coordinate;

impl Brep {
    /// The exact chord polyline of every sharp edge ([`Brep::features`]),
    /// paired with the [`Brep::faces`] indices it borders -- the faces a
    /// crease-owned node is allowed to come from; a node whose nearest face is
    /// none of these (an unrelated wall merely sitting close by, such as the
    /// far side of a thin flange) must never be pulled onto this curve.
    pub fn crease_curves(&self) -> Vec<(Vec<Coordinate<D>>, Vec<usize>)> {
        self.features()
            .creases
            .into_iter()
            .map(|index| {
                let edge = &self.edges[index];
                let [a, b] = edge.vertices;
                let curve = chords(
                    &edge.curve,
                    &self.vertices[a],
                    &self.vertices[b],
                    true,
                    a == b,
                );
                (curve, self.incident_faces(index))
            })
            .collect()
    }

    /// World positions of every hard point ([`Brep::features`] corners) --
    /// where two or more creases (and three or more faces) meet. A crease
    /// curve alone still lets a node near one of these slide freely along
    /// its own tangent; right at the junction that freedom is the problem
    /// (several creases converge, so "along the curve" is not one direction),
    /// so the fit pins a node that lands here onto the exact point instead.
    pub fn corner_points(&self) -> Vec<Coordinate<D>> {
        self.features()
            .corners
            .into_iter()
            .map(|vertex| self.vertices[vertex].clone())
            .collect()
    }
}
