mod base;
// mod connectivity;
mod from;
mod into;
mod tessellation;
mod write;

#[cfg(feature = "netcdf")]
mod exodus;

pub use self::tessellation::Tessellation;

#[cfg(feature = "netcdf")]
pub use self::exodus::WriteExodus;

use crate::{geometry::Coordinates, math::Tensor};

// Can bring in Sets, but should generalize across two concrete types
// (with/without id numbers stored) and avoid extra storage.

pub struct PrimitiveConnectivity<const M: usize, const N: usize>(pub(crate) Vec<[usize; N]>);
pub struct PolytopalConnectivity<const M: usize>(pub(crate) Vec<Vec<usize>>);

impl<const M: usize, const N: usize> From<Vec<[usize; N]>> for PrimitiveConnectivity<M, N> {
    fn from(connectivity: Vec<[usize; N]>) -> Self {
        PrimitiveConnectivity(connectivity)
    }
}

trait ConnectivityImpl {
    fn len(&self) -> usize;
    fn number_of_nodes_per_element(&self) -> Option<usize>;
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str;
    #[cfg(feature = "netcdf")]
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>>;
}

impl<const M: usize, const N: usize> ConnectivityImpl for PrimitiveConnectivity<M, N> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn number_of_nodes_per_element(&self) -> Option<usize> {
        Some(N)
    }
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str {
        match (M, N) {
            (2, 3) => "tri3",
            (2, 4) => "quad4",
            (3, 4) => "tet4",
            (3, 8) => "hex8",
            _ => panic!("unknown primitive element type: M={M}, N={N}"),
        }
    }
    #[cfg(feature = "netcdf")]
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>> {
        Some(
            self.0
                .iter()
                .flat_map(|nodes| nodes.iter().map(|&node| node as i32 + 1))
                .collect(),
        )
    }
}

impl<const M: usize> ConnectivityImpl for PolytopalConnectivity<M> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn number_of_nodes_per_element(&self) -> Option<usize> {
        None
    }
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str {
        match M {
            2 => "nsided",
            3 => "nfaced",
            _ => panic!("unknown polytopal element type: M={M}"),
        }
    }
    #[cfg(feature = "netcdf")]
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>> {
        None
    }
}

pub enum Connectivity {
    Hexahedral(PrimitiveConnectivity<3, 8>),
    Polyhedral(PolytopalConnectivity<3>),
    Polygonal(PolytopalConnectivity<2>),
    Quadrilateral(PrimitiveConnectivity<2, 4>),
    Tetrahedral(PrimitiveConnectivity<3, 4>),
    Triangular(PrimitiveConnectivity<2, 3>),
}

pub type Connectivities = Vec<Connectivity>;

impl Connectivity {
    fn as_impl(&self) -> &dyn ConnectivityImpl {
        match self {
            Connectivity::Hexahedral(c) => c,
            Connectivity::Polyhedral(c) => c,
            Connectivity::Polygonal(c) => c,
            Connectivity::Quadrilateral(c) => c,
            Connectivity::Tetrahedral(c) => c,
            Connectivity::Triangular(c) => c,
        }
    }
    fn len(&self) -> usize {
        self.as_impl().len()
    }
    fn number_of_nodes_per_element(&self) -> Option<usize> {
        self.as_impl().number_of_nodes_per_element()
    }
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str {
        self.as_impl().exodus_element_type()
    }
    #[cfg(feature = "netcdf")]
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>> {
        self.as_impl().primitive_connectivity_flattened()
    }
}

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Coordinates<D>,
}
