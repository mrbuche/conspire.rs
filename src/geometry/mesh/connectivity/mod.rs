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
    Pyramidal(PrimitiveConnectivity<3, 5>),
    Quadrilateral(PrimitiveConnectivity<2, 4>),
    Tetrahedral(PrimitiveConnectivity<3, 4>),
    Triangular(PrimitiveConnectivity<2, 3>),
    Wedge(PrimitiveConnectivity<3, 6>),
}

pub type Connectivities = Set<Vec<Connectivity>>;

macro_rules! try_from_connectivity {
    ($variant: ident, $m: literal, $n: literal, $error: literal) => {
        impl TryFrom<Connectivity> for PrimitiveConnectivity<$m, $n> {
            type Error = &'static str;
            fn try_from(connectivity: Connectivity) -> Result<Self, Self::Error> {
                match connectivity {
                    Connectivity::$variant(connectivity) => Ok(connectivity),
                    _ => Err($error),
                }
            }
        }
    };
}
try_from_connectivity!(Hexahedral, 3, 8, "block is not hexahedral");
try_from_connectivity!(Pyramidal, 3, 5, "block is not pyramidal");
try_from_connectivity!(Quadrilateral, 2, 4, "block is not quadrilateral");
try_from_connectivity!(Tetrahedral, 3, 4, "block is not tetrahedral");
try_from_connectivity!(Triangular, 2, 3, "block is not triangular");
try_from_connectivity!(Wedge, 3, 6, "block is not a wedge");

impl TryFrom<Connectivity> for PolytopalConnectivity<3> {
    type Error = &'static str;
    fn try_from(connectivity: Connectivity) -> Result<Self, Self::Error> {
        match connectivity {
            Connectivity::Polyhedral(connectivity) => Ok(connectivity),
            _ => Err("block is not polyhedral"),
        }
    }
}

impl TryFrom<Connectivity> for PolytopalConnectivity<2> {
    type Error = &'static str;
    fn try_from(connectivity: Connectivity) -> Result<Self, Self::Error> {
        match connectivity {
            Connectivity::Polygonal(connectivity) => Ok(connectivity),
            _ => Err("block is not polygonal"),
        }
    }
}
