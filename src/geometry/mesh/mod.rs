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

struct PrimitiveConnectivity<const M: usize, const N: usize, T>(Vec<[T; N]>);
struct PolytopalConnectivity<const M: usize, T>(Vec<Vec<T>>);

trait ConnectivityImpl<T> {
    fn len(&self) -> usize;
    fn number_of_nodes_per_element(&self) -> Option<usize>;
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str;
    #[cfg(feature = "netcdf")]
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>>;
}

impl<const M: usize, const N: usize, T> ConnectivityImpl<T> for PrimitiveConnectivity<M, N, T>
where
    T: Copy + Into<i32>,
{
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
                .flat_map(|nodes| nodes.iter().map(|&node| node.into() + 1))
                .collect(),
        )
    }
}

impl<const M: usize, T> ConnectivityImpl<T> for PolytopalConnectivity<M, T>
where
    T: Copy + Into<i32>,
{
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

enum Connectivity<T> {
    Hexahedral(PrimitiveConnectivity<3, 8, T>),
    Polyhedral(PolytopalConnectivity<3, T>),
    Polygonal(PolytopalConnectivity<2, T>),
    Quadrilateral(PrimitiveConnectivity<2, 4, T>),
    Tetrahedral(PrimitiveConnectivity<3, 4, T>),
    Triangular(PrimitiveConnectivity<2, 3, T>),
}

type Connectivities<T> = Vec<Connectivity<T>>;

impl<T> Connectivity<T>
where
    T: Copy + Into<i32>,
{
    fn as_impl(&self) -> &dyn ConnectivityImpl<T> {
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

pub struct MeshNew<const D: usize, T> {
    connectivities: Connectivities<T>,
    coordinates: Coordinates<D, 0>,
}

impl<const D: usize, T> MeshNew<D, T>
where
    T: Copy + Into<i32>,
{
    fn connectivities(&self) -> &[Connectivity<T>] {
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
