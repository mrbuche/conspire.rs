pub mod base;
pub mod iter;
pub mod polytopal;
pub mod primitive;

use self::{polytopal::PolytopalConnectivity, primitive::PrimitiveConnectivity};

// Can bring in Sets, but should generalize across two concrete types
// (with/without id numbers stored) and avoid extra storage.

pub enum Connectivity {
    Hexahedral(PrimitiveConnectivity<3, 8>),
    Polyhedral(PolytopalConnectivity<3>),
    Polygonal(PolytopalConnectivity<2>),
    Quadrilateral(PrimitiveConnectivity<2, 4>),
    Tetrahedral(PrimitiveConnectivity<3, 4>),
    Triangular(PrimitiveConnectivity<2, 3>),
}

pub type Connectivities = Vec<Connectivity>;
