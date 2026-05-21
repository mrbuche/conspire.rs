use crate::{
    geometry::{
        Coordinate, Write,
        mesh::{TriangularMesh, tessellation::Tessellation},
    },
    math::test::TestError,
};
use std::path::Path;

const CONNECTIVITY: [[usize; 3]; 12] = [
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

const COORDINATES: [Coordinate<3, 1>; 8] = [
    Coordinate::const_from([0.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 1.0, 1.0]),
    Coordinate::const_from([0.0, 1.0, 1.0]),
];

const NORMALS: [Coordinate<3, 1>; 12] = [
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
fn todo() -> Result<(), TestError> {
    let mesh = TriangularMesh::from((CONNECTIVITY.into(), COORDINATES.into()));
    let tessellation_1 = Tessellation::from(mesh);
    // can test three fields against the above here
    tessellation_1.write("target/foo.stl")?;
    let tessellation_2 = Tessellation::<1, usize>::try_from(Path::new("target/foo.stl"))?;
    // impl PartialEq for Tessellation to make test at the end much easier
    // and/or something that returns TestError
    Ok(())
}
