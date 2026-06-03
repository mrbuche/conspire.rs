use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity,
            tessellation::from::test::{CONNECTIVITY, COORDINATES, NORMALS, tessellation},
        },
    },
    io::Write,
    math::{
        Tensor,
        test::{TestError, assert_eq},
    },
};

#[test]
fn consistency() -> Result<(), TestError> {
    let tessellation = tessellation();
    match &tessellation.mesh().connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
    let coords_expected: Coordinates<3> = COORDINATES.into();
    assert_eq(tessellation.mesh().coordinates(), &coords_expected)?;
    tessellation.normals()[0]
        .iter()
        .zip(NORMALS.iter())
        .try_for_each(|(a, b)| assert_eq(a, b))?;
    Ok(tessellation.write("target/foo.stl")?)
}
