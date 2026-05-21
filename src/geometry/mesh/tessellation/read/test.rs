use crate::{
    geometry::mesh::tessellation::{
        Tessellation,
        write::test::{CONNECTIVITY, COORDINATES, NORMALS},
    },
    math::test::{TestError, assert_eq},
};
use std::path::Path;

#[test]
fn consistency() -> Result<(), TestError> {
    let tessellation_1 = Tessellation::<1, usize>::try_from(Path::new("target/foo.stl"))?;
    assert_eq!(tessellation_1.mesh.connectivity, CONNECTIVITY);
    assert_eq(&tessellation_1.mesh.coordinates, &COORDINATES.into())?;
    assert_eq(&tessellation_1.normals, &NORMALS.into())
}
