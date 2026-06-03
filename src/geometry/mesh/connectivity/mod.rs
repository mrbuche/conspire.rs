pub mod base;
pub mod iter;
pub mod polytopal;
pub mod primitive;

use crate::{
    geometry::mesh::connectivity::{
        polytopal::PolytopalConnectivity, primitive::PrimitiveConnectivity,
    },
    math::Set,
};

pub enum Connectivity {
    Hexahedral(PrimitiveConnectivity<3, 8>),
    Polyhedral(PolytopalConnectivity<3>),
    Polygonal(PolytopalConnectivity<2>),
    Quadrilateral(PrimitiveConnectivity<2, 4>),
    Tetrahedral(PrimitiveConnectivity<3, 4>),
    Triangular(PrimitiveConnectivity<2, 3>),
}

pub type Connectivities = Set<Vec<Connectivity>>;
