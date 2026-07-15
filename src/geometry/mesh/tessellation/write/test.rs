use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity,
            tessellation::from::test::{NORMALS, tessellation},
            test::{CONNECTIVITY, COORDINATES},
        },
    },
    io::Write,
    math::{
        Tensor,
        assert::{AssertionError, assert_eq},
    },
};

#[test]
fn consistency() -> Result<(), AssertionError> {
    let tessellation = tessellation();
    match &tessellation.mesh().connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
    let coords_expected = Coordinates::from(COORDINATES);
    assert_eq(tessellation.mesh().coordinates(), &coords_expected)?;
    tessellation.normals()[0]
        .iter()
        .zip(NORMALS.iter())
        .try_for_each(|(a, b)| assert_eq(a, b))?;
    Ok(tessellation.write("target/foo.stl")?)
}
