mod triangle;

use crate::fem::block::element::surface::SurfaceElement;

pub use triangle::Triangle;

pub type LinearSurfaceElement<const G: usize, const N: usize> = SurfaceElement<G, N, 1>;
