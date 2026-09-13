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
    /// The exact chord polyline of every sharp edge ([`Brep::features`]).
    pub fn crease_curves(&self) -> Vec<Vec<Coordinate<D>>> {
        self.features()
            .creases
            .into_iter()
            .map(|index| {
                let edge = &self.edges[index];
                let [a, b] = edge.vertices;
                chords(
                    &edge.curve,
                    &self.vertices[a],
                    &self.vertices[b],
                    true,
                    a == b,
                )
            })
            .collect()
    }
}
