use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, tessellation::Tessellation, test::mesh},
    },
    math::{
        Tensor,
        test::{TestError, assert_eq},
    },
};

pub const NORMALS: [Coordinate<3>; 12] = [
    Coordinate::const_from([0.0, 0.0, -1.0]),
    Coordinate::const_from([0.0, 0.0, -1.0]),
    Coordinate::const_from([0.0, 0.0, 1.0]),
    Coordinate::const_from([0.0, 0.0, 1.0]),
    Coordinate::const_from([0.0, -1.0, 0.0]),
    Coordinate::const_from([0.0, -1.0, 0.0]),
    Coordinate::const_from([0.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 1.0, 0.0]),
    Coordinate::const_from([-1.0, 0.0, 0.0]),
    Coordinate::const_from([-1.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 0.0, 0.0]),
];

pub fn tessellation() -> Tessellation {
    Tessellation::from(mesh())
}

#[test]
fn triangular_mesh() -> Result<(), TestError> {
    let connectivities = vec![Connectivity::Triangular(
        vec![[0_usize, 1, 2], [0, 3, 1]].into(),
    )];
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]);
    let mesh = Mesh::from((connectivities, coordinates));
    let tessellation = Tessellation::from(mesh);
    let up = Coordinate::const_from([0.0, 0.0, 1.0]);
    tessellation
        .normals()
        .iter()
        .flat_map(|block| block.iter())
        .try_for_each(|normal| assert_eq(normal, &up))
}
