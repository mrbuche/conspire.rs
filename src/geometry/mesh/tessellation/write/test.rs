use crate::{
    geometry::{
        Coordinate, Write,
        mesh::{TriangularMesh, tessellation::Tessellation},
    },
    math::test::{TestError, assert_eq},
};

pub const CONNECTIVITY: [[usize; 3]; 12] = [
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [3, 6, 2],
    [3, 7, 6],
    [0, 4, 7],
    [0, 7, 3],
    [1, 2, 6],
    [1, 6, 5],
];

pub const COORDINATES: [Coordinate<3, 1>; 8] = [
    Coordinate::const_from([0.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 1.0, 1.0]),
    Coordinate::const_from([0.0, 1.0, 1.0]),
];

pub const NORMALS: [Coordinate<3, 1>; 12] = [
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

#[test]
fn consistency() -> Result<(), TestError> {
    let mesh = TriangularMesh::from((CONNECTIVITY.into(), COORDINATES.into()));
    let tessellation_1 = Tessellation::from(mesh);
    assert_eq!(tessellation_1.mesh.connectivity, CONNECTIVITY);
    assert_eq(&tessellation_1.mesh.coordinates, &COORDINATES.into())?;
    assert_eq(&tessellation_1.normals, &NORMALS.into())?;
    Ok(tessellation_1.write("target/foo.stl")?)
}
