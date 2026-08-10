use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Connectivity,
            tessellation::{Tessellation, from::test::tessellation as fixture},
        },
    },
    io::Write,
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
};
use std::{fmt::Write as WriteFmt, fs::write, path::Path};

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

fn ascii(tessellation: &Tessellation) -> String {
    let mut contents = "solid conspire\n".to_string();
    facets(tessellation).iter().for_each(|(normal, vertices)| {
        writeln!(
            contents,
            "  facet normal {} {} {}\n    outer loop",
            normal[0], normal[1], normal[2]
        )
        .unwrap();
        vertices.iter().for_each(|vertex| {
            writeln!(
                contents,
                "      vertex {} {} {}",
                vertex[0], vertex[1], vertex[2]
            )
            .unwrap()
        });
        contents.push_str("    endloop\n  endfacet\n");
    });
    contents.push_str("endsolid conspire\n");
    contents
}

#[test]
fn binary() -> Result<(), AssertionError> {
    fixture().write("target/read_binary.stl")?;
    assert_facets(&Tessellation::try_from(Path::new(
        "target/read_binary.stl",
    ))?)
}

#[test]
fn ascii_file() -> Result<(), AssertionError> {
    write("target/read_ascii.stl", ascii(&fixture()))?;
    assert_facets(&Tessellation::try_from(Path::new("target/read_ascii.stl"))?)
}

#[test]
fn ascii_missing_keyword() {
    let contents = ascii(&fixture()).replace("endloop", "loopend");
    write("target/read_ascii_invalid.stl", contents).unwrap();
    assert!(Tessellation::try_from(Path::new("target/read_ascii_invalid.stl")).is_err())
}
