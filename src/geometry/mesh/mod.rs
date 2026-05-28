#[cfg(feature = "netcdf")]
pub mod exodus;

pub mod base;
pub mod from;
pub mod into;
pub mod tessellation;
pub mod write;

use crate::geometry::Coordinates;

pub struct Mesh<const D: usize, const I: usize, const M: usize, T> {
    connectivity: T,
    coordinates: Coordinates<D, I>,
}

pub type PrimitiveMesh<const D: usize, const I: usize, const M: usize, const N: usize, T> =
    Mesh<D, I, M, Vec<[T; N]>>;

pub type HexahedralMesh<const I: usize, T> = PrimitiveMesh<3, I, 3, 8, T>;
pub type TetrahedralMesh<const I: usize, T> = PrimitiveMesh<3, I, 3, 4, T>;

pub type QuadrilateralMesh<const D: usize, const I: usize, T> = PrimitiveMesh<D, I, 2, 4, T>;
pub type TriangularMesh<const I: usize, T> = PrimitiveMesh<3, I, 2, 3, T>;

// Can bring in Sets, but should generalize across two concrete types
// (with/without id numbers stored) and avoid extra storage.

use crate::math::Tensor;

struct PrimitiveConnectivity<const D: usize, const M: usize, const N: usize, T>(Vec<[T; N]>);
struct PolytopalConnectivity<const D: usize, const M: usize, T>(Vec<Vec<T>>);

enum Connectivity<const D: usize, T> {
    Hexahedral(PrimitiveConnectivity<3, 3, 8, T>),
    Polyhedral(PolytopalConnectivity<3, 3, T>),
    Polygonal(PolytopalConnectivity<D, 2, T>),
    Quadrilateral(PrimitiveConnectivity<D, 2, 4, T>),
    Tetrahedral(PrimitiveConnectivity<3, 3, 4, T>),
    Triangular(PrimitiveConnectivity<D, 2, 3, T>),
}

type Connectivities<const D: usize, T> = Vec<Connectivity<D, T>>;

impl<const D: usize, T> Connectivity<D, T>
where
    T: Copy + Into<i32>,
{
    fn len(&self) -> usize {
        match self {
            Connectivity::Hexahedral(connectivity) => connectivity.0.len(),
            Connectivity::Polyhedral(connectivity) => connectivity.0.len(),
            Connectivity::Polygonal(connectivity) => connectivity.0.len(),
            Connectivity::Quadrilateral(connectivity) => connectivity.0.len(),
            Connectivity::Tetrahedral(connectivity) => connectivity.0.len(),
            Connectivity::Triangular(connectivity) => connectivity.0.len(),
        }
    }
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str {
        match self {
            Connectivity::Hexahedral(_) => "hex8",
            Connectivity::Polyhedral(_) => "nfaced",
            Connectivity::Polygonal(_) => "nsided",
            Connectivity::Quadrilateral(_) => "quad4",
            Connectivity::Tetrahedral(_) => "tet4",
            Connectivity::Triangular(_) => "tri3",
        }
    }
    fn number_of_nodes_per_element(&self) -> Option<usize> {
        match self {
            Connectivity::Hexahedral(connectivity) => Some(connectivity.0[0].len()),
            Connectivity::Polyhedral(_) => None,
            Connectivity::Polygonal(_) => None,
            Connectivity::Quadrilateral(connectivity) => Some(connectivity.0[0].len()),
            Connectivity::Tetrahedral(connectivity) => Some(connectivity.0[0].len()),
            Connectivity::Triangular(connectivity) => Some(connectivity.0[0].len()),
        }
    }
    #[cfg(feature = "netcdf")]
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>> {
        match self {
            Connectivity::Hexahedral(connectivity) => Some(
                connectivity
                    .0
                    .iter()
                    .flat_map(|nodes| nodes.iter().map(|&node| node.into() + 1))
                    .collect(),
            ),
            Connectivity::Polyhedral(_) => None,
            Connectivity::Polygonal(_) => None,
            Connectivity::Quadrilateral(connectivity) => Some(
                connectivity
                    .0
                    .iter()
                    .flat_map(|nodes| nodes.iter().map(|&node| node.into() + 1))
                    .collect(),
            ),
            Connectivity::Tetrahedral(connectivity) => Some(
                connectivity
                    .0
                    .iter()
                    .flat_map(|nodes| nodes.iter().map(|&node| node.into() + 1))
                    .collect(),
            ),
            Connectivity::Triangular(connectivity) => Some(
                connectivity
                    .0
                    .iter()
                    .flat_map(|nodes| nodes.iter().map(|&node| node.into() + 1))
                    .collect(),
            ),
        }
    }
}

pub struct MeshNew<const D: usize, T> {
    connectivities: Connectivities<D, T>,
    coordinates: Coordinates<D, 0>,
}

impl<const D: usize, T> MeshNew<D, T>
where
    T: Copy + Into<i32>,
{
    fn connectivities(&self) -> &[Connectivity<D, T>] {
        &self.connectivities
    }
    fn coordinates(&self) -> &Coordinates<D, 0> {
        &self.coordinates
    }
    fn number_of_blocks(&self) -> usize {
        self.connectivities.len()
    }
    fn number_of_elements(&self) -> usize {
        self.connectivities
            .iter()
            .map(|connectivity| connectivity.len())
            .sum()
    }
    fn number_of_nodes(&self) -> usize {
        self.coordinates.len()
    }
}
