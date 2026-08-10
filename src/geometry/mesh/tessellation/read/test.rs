use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Connectivity,
            tessellation::{Output, Tessellation, from::test::tessellation as fixture},
        },
    },
    io::{Encoding, Write},
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
};
use std::{
    fs::{read_to_string, write},
    path::Path,
};

fn facets(tessellation: &Tessellation) -> Vec<(Coordinate<3>, [Coordinate<3>; 3])> {
    tessellation
        .mesh()
        .connectivities()
        .iter()
        .zip(tessellation.normals().iter())
        .flat_map(|(connectivity, normals)| match connectivity {
            Connectivity::Triangular(triangles) => triangles
                .iter()
                .zip(normals.iter())
                .map(|(nodes, normal)| {
                    (
                        normal.clone(),
                        nodes.map(|node| tessellation.mesh().coordinates()[node].clone()),
                    )
                })
                .collect::<Vec<_>>(),
            _ => panic!("expected Triangular block"),
        })
        .collect()
}

fn assert_facets(tessellation: &Tessellation) -> Result<(), AssertionError> {
    facets(tessellation)
        .iter()
        .zip(facets(&fixture()).iter())
        .try_for_each(
            |((normal, vertices), (normal_expected, vertices_expected))| {
                Assert::eq(normal, normal_expected)?;
                vertices
                    .iter()
                    .zip(vertices_expected.iter())
                    .try_for_each(|(vertex, vertex_expected)| Assert::eq(vertex, vertex_expected))
            },
        )
}

#[test]
fn binary() -> Result<(), AssertionError> {
    fixture().write(Output::Stl(Encoding::Binary("target/read_binary.stl")))?;
    assert_facets(&Tessellation::try_from(Path::new(
        "target/read_binary.stl",
    ))?)
}

#[test]
fn ascii_file() -> Result<(), AssertionError> {
    fixture().write(Output::Stl(Encoding::Ascii("target/read_ascii.stl")))?;
    assert_facets(&Tessellation::try_from(Path::new("target/read_ascii.stl"))?)
}

#[test]
fn ascii_missing_keyword() -> Result<(), AssertionError> {
    fixture().write(Output::Stl(Encoding::Ascii(
        "target/read_ascii_invalid.stl",
    )))?;
    let contents = read_to_string("target/read_ascii_invalid.stl")?.replace("endloop", "loopend");
    write("target/read_ascii_invalid.stl", contents)?;
    assert!(Tessellation::try_from(Path::new("target/read_ascii_invalid.stl")).is_err());
    Ok(())
}
