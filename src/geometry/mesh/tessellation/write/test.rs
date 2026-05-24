use crate::{
    geometry::{
        Write,
        mesh::tessellation::from::test::{CONNECTIVITY, COORDINATES, NORMALS, tessellation},
    },
    math::test::{TestError, assert_eq},
};

#[test]
fn consistency() -> Result<(), TestError> {
    let tessellation = tessellation();
    assert_eq!(tessellation.mesh.connectivity, CONNECTIVITY);
    assert_eq(&tessellation.mesh.coordinates, &COORDINATES.into())?;
    assert_eq(&tessellation.normals, &NORMALS.into())?;
    Ok(tessellation.write("target/foo.stl")?)
}
