use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Mesh, from::test::mesh, tessellation::Tessellation},
    },
    math::{
        Tensor,
        test::{TestError, assert_eq},
    },
};

pub use crate::geometry::mesh::from::test::{CONNECTIVITY, COORDINATES};

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
fn triangluar_mesh() -> Result<(), TestError> {
    let connectivity: Vec<[usize; _]> = vec![[0, 1, 2], [0, 3, 1]];
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]);
    let mesh = Mesh::from((connectivity, coordinates));
    let tessellation = Tessellation::from(mesh);
    tessellation
        .normals
        .iter()
        .try_for_each(|normal| assert_eq(normal, &[0.0, 0.0, 1.0].into()))
}
