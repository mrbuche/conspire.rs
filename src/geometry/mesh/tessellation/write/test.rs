use crate::math::assert::Assert;
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
    math::{Tensor, assert::AssertionError},
};

use super::Stl;
use std::fs::{metadata, read_to_string};

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
    Assert::eq(tessellation.mesh().coordinates(), &coords_expected)?;
    tessellation.normals()[0]
        .iter()
        .zip(NORMALS.iter())
        .try_for_each(|(a, b)| Assert::eq(a, b))?;
    Ok(tessellation.write(Stl::Binary("target/foo.stl"))?)
}

#[test]
fn ascii() -> Result<(), AssertionError> {
    tessellation().write(Stl::Ascii("target/foo_ascii.stl"))?;
    let contents = read_to_string("target/foo_ascii.stl")?;
    assert!(contents.starts_with("solid conspire\n"));
    assert!(contents.ends_with("endsolid conspire\n"));
    assert_eq!(contents.matches("facet normal").count(), CONNECTIVITY.len());
    assert_eq!(contents.matches("vertex").count(), 3 * CONNECTIVITY.len());
    Ok(())
}

#[test]
fn binary_is_not_detected_as_ascii() -> Result<(), AssertionError> {
    tessellation().write(Stl::Binary("target/foo_binary.stl"))?;
    assert_eq!(
        metadata("target/foo_binary.stl")?.len(),
        84 + 50 * CONNECTIVITY.len() as u64
    );
    Ok(())
}
