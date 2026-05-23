use crate::{
    geometry::{
        Coordinate, Coordinates, Write,
        mesh::{
            TriangularMesh,
            tessellation::Tessellation,
            write::test::{CONNECTIVITY, COORDINATES},
        },
    },
    math::test::{TestError, assert_eq},
};

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
    let connectivity = CONNECTIVITY.to_vec();
    let coordinates = Coordinates::from(COORDINATES);
    let mesh = TriangularMesh::from((connectivity, coordinates));
    let tessellation = Tessellation::from(mesh);
    assert_eq!(tessellation.mesh.connectivity, CONNECTIVITY);
    assert_eq(&tessellation.mesh.coordinates, &COORDINATES.into())?;
    assert_eq(&tessellation.normals, &NORMALS.into())?;
    Ok(tessellation.write("target/foo.stl")?)
}
