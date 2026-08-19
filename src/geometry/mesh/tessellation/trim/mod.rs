#[cfg(test)]
pub mod test;

use crate::{
    geometry::{
        mesh::{
            Mesh,
            tessellation::{D, Tessellation},
        },
        primitive::Solid,
    },
    math::{Quantity, Scalar},
    units::Length,
};

const TRIM_RATIO: Scalar = 0.1;

impl Tessellation {
    /// Discards the cells of a background mesh lying outside this
    /// tessellation, leaving a mesh that covers the volume it encloses.
    ///
    /// A cell survives when the signed distances at its nodes satisfy
    /// `minimum + 0.1 * maximum >= 0`, so the cells straddling the surface
    /// are kept for [`buffer`](Mesh::buffer) to fit onto it.
    pub fn trim(&self, mesh: &mut Mesh<D>) -> Result<(), &'static str> {
        trim_to(self, mesh)
    }
}

/// Discards the cells of a background mesh lying outside a solid, by the rule
/// [`Tessellation::trim`] describes.
///
/// Written against what a solid answers rather than against a surface, so that
/// a shape known in closed form is trimmed to as readily as a tessellated one,
/// and by the very same rule.
pub fn trim_to<S: Solid<D>>(solid: &S, mesh: &mut Mesh<D>) -> Result<(), &'static str> {
    let zero = Quantity::default();
    let signed = solid.signed_distances(mesh.coordinates());
    mesh.keep_hexes(|_, hex, _| {
        let (minimum, maximum) = hex.iter().fold(
            (
                Quantity::<Length>::new(Scalar::INFINITY),
                Quantity::new(Scalar::NEG_INFINITY),
            ),
            |(minimum, maximum), &node| (minimum.min(signed[node]), maximum.max(signed[node])),
        );
        maximum + minimum * TRIM_RATIO <= zero
    })
}
