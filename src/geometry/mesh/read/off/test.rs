use super::ReadOff;
use crate::{
    geometry::{
        Coordinate,
        mesh::{Input, Mesh},
    },
    math::Quantity,
};
use std::{fs::write, path::Path};

const TETRAHEDRON: &str = "OFF\n4 4 6\n\
    0 0 0\n1 0 0\n0 1 0\n0 0 1\n\
    3 0 2 1\n3 0 1 3\n3 1 2 3\n3 0 3 2\n";

fn first_element(mesh: &Mesh<3>, block: usize) -> &[usize] {
    mesh.iter().nth(block).unwrap().iter().next().unwrap()
}

fn read(name: &str, text: &str) -> std::io::Result<Mesh<3>> {
    let path = format!("target/{name}.off");
    write(&path, text).unwrap();
    Mesh::<3>::read_off(path)
}

#[test]
fn reads_triangles() {
    let mesh = read("tetrahedron", TETRAHEDRON).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(mesh.number_of_elements(), 4);
    assert_eq!(first_element(&mesh, 0), [0, 2, 1]);
    let coordinates = mesh.coordinates();
    assert_eq!(coordinates[1], Coordinate::const_from([1.0, 0.0, 0.0]));
    assert_eq!(coordinates[3][2], Quantity::new(1.0));
}

#[test]
fn reads_through_input() {
    let path = "target/through_input.off";
    write(path, TETRAHEDRON).unwrap();
    assert_eq!(Input::Off(path).as_ref(), Path::new(path));
    let mesh = Mesh::<3>::try_from(Input::Off(path)).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_elements(), 4);
}

#[test]
fn ignores_the_edge_count() {
    let zero = read("edges_zero", TETRAHEDRON.replace("4 4 6", "4 4 0").as_str()).unwrap();
    let wrong = read(
        "edges_wrong",
        TETRAHEDRON.replace("4 4 6", "4 4 999").as_str(),
    )
    .unwrap();
    for mesh in [zero, wrong] {
        assert_eq!(mesh.number_of_nodes(), 4);
        assert_eq!(mesh.number_of_elements(), 4);
    }
}

#[test]
fn accepts_the_counts_on_the_magic_line() {
    let mesh = read("one_line", &TETRAHEDRON.replace("OFF\n", "OFF ")).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_elements(), 4);
}

#[test]
fn skips_comments_and_blank_lines() {
    let text = "# a tetrahedron\nOFF\n\n4 4 6 # counts\n\
        0 0 0\n\n1 0 0 # x\n0 1 0\n0 0 1\n\
        # faces\n3 0 2 1\n3 0 1 3\n3 1 2 3\n3 0 3 2\n\n";
    let mesh = read("comments", text).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_elements(), 4);
}

#[test]
fn accepts_windows_line_endings() {
    let mesh = read("crlf", &TETRAHEDRON.replace('\n', "\r\n")).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_elements(), 4);
}

#[test]
fn ignores_face_colors() {
    let text = TETRAHEDRON
        .replace("3 0 2 1", "3 0 2 1 255 0 0")
        .replace("3 0 1 3", "3 0 1 3 0.5");
    let mesh = read("colors", &text).unwrap();
    assert_eq!(mesh.number_of_elements(), 4);
    assert_eq!(first_element(&mesh, 0), [0, 2, 1]);
}

#[test]
fn keeps_an_unreferenced_vertex() {
    let text = "OFF\n4 1 3\n0 0 0\n1 0 0\n0 1 0\n5 5 5\n3 0 1 2\n";
    let mesh = read("unreferenced", text).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_elements(), 1);
}

#[test]
fn splits_triangles_and_quadrilaterals_into_blocks() {
    let text = "OFF\n5 3 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n2 0 0\n\
        4 0 1 2 3\n3 1 4 2\n3 0 1 3\n";
    let mesh = read("mixed", text).unwrap();
    assert_eq!(mesh.number_of_nodes(), 5);
    assert_eq!(mesh.number_of_element_blocks(), 2);
    assert_eq!(mesh.number_of_elements(), 3);
    assert_eq!(first_element(&mesh, 0), [1, 4, 2]);
    assert_eq!(first_element(&mesh, 1), [0, 1, 2, 3]);
}

#[test]
fn reads_vertices_without_faces() {
    let mesh = read("no_faces", "OFF\n2 0 0\n0 0 0\n1 1 1\n").unwrap();
    assert_eq!(mesh.number_of_nodes(), 2);
    assert_eq!(mesh.number_of_element_blocks(), 0);
}

#[test]
fn rejects_a_missing_or_unsupported_magic_word() {
    for (name, magic) in [
        ("coff", "COFF"),
        ("noff", "NOFF"),
        ("stoff", "STOFF"),
        ("four", "4OFF"),
        ("lower_coff", "coff"),
        ("mixed_noff", "Noff"),
        ("mesh", "MeshVersionFormatted"),
    ] {
        let text = TETRAHEDRON.replacen("OFF", magic, 1);
        assert!(read(name, &text).is_err(), "{magic} should fail");
    }
    assert!(read("no_magic", "4 4 6\n").is_err());
}

#[test]
fn accepts_the_magic_word_in_any_case() {
    for (name, magic) in [("lower", "off"), ("title", "Off"), ("mixed", "oFF")] {
        let text = TETRAHEDRON.replacen("OFF", magic, 1);
        let mesh = read(name, &text).unwrap();
        assert_eq!(mesh.number_of_nodes(), 4, "{magic}");
        assert_eq!(mesh.number_of_elements(), 4, "{magic}");
    }
}

#[test]
fn rejects_a_coordinate_that_is_not_finite() {
    for (index, value) in [
        "nan", "NaN", "inf", "-inf", "+inf", "Infinity", "1e999", "-1e999",
    ]
    .iter()
    .enumerate()
    {
        let text = TETRAHEDRON.replace("0 1 0\n", &format!("0 {value} 0\n"));
        let error = read(&format!("not_finite_{index}"), &text).err();
        assert!(
            error.is_some_and(|error| error.to_string().contains("not finite")),
            "{value} should fail as not finite"
        );
    }
}

#[test]
fn accepts_large_small_and_signed_zero_coordinates() {
    let text = TETRAHEDRON.replace("0 1 0\n", "-0.0 1e300 1.5E-3\n");
    let mesh = read("extreme", &text).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.coordinates()[2][1], Quantity::new(1e300));
}

#[test]
fn rejects_an_empty_file() {
    assert!(read("empty", "").is_err());
    assert!(read("only_comments", "# nothing\n\n").is_err());
    assert!(read("magic_only", "OFF\n").is_err());
}

#[test]
fn rejects_a_dimension_mismatch() {
    let path = "target/wrong_dim.off";
    write(path, TETRAHEDRON).unwrap();
    assert!(Mesh::<2>::read_off(path).is_err());
}

#[test]
fn rejects_bad_counts() {
    assert!(read("two_counts", &TETRAHEDRON.replace("4 4 6", "4 4")).is_err());
    assert!(read("four_counts", &TETRAHEDRON.replace("4 4 6", "4 4 6 1")).is_err());
    assert!(read("text_count", &TETRAHEDRON.replace("4 4 6", "four 4 6")).is_err());
    assert!(read("text_faces", &TETRAHEDRON.replace("4 4 6", "4 four 6")).is_err());
    assert!(read("text_edges", &TETRAHEDRON.replace("4 4 6", "4 4 six")).is_err());
    assert!(read("negative_count", &TETRAHEDRON.replace("4 4 6", "4 -4 6")).is_err());
}

#[test]
fn rejects_a_file_that_ends_in_the_vertices() {
    assert!(read("ends_in_vertices", "OFF\n3 0 0\n0 0 0\n1 0 0\n").is_err());
}

#[test]
fn rejects_a_bad_vertex() {
    assert!(read("two_values", &TETRAHEDRON.replace("0 1 0\n", "0 1\n")).is_err());
    assert!(read("four_values", &TETRAHEDRON.replace("0 1 0\n", "0 1 0 1\n")).is_err());
    assert!(read("text_value", &TETRAHEDRON.replace("0 1 0\n", "0 up 0\n")).is_err());
}

#[test]
fn rejects_a_bad_face() {
    assert!(read("out_of_range", &TETRAHEDRON.replace("3 0 2 1", "3 0 2 4")).is_err());
    assert!(read("too_short", &TETRAHEDRON.replace("3 0 2 1", "3 0 2")).is_err());
    assert!(read("pentagon", &TETRAHEDRON.replace("3 0 2 1", "5 0 1 2 3 0")).is_err());
    assert!(read("size_one", &TETRAHEDRON.replace("3 0 2 1", "1 0")).is_err());
    assert!(read("negative", &TETRAHEDRON.replace("3 0 2 1", "3 0 2 -1")).is_err());
    assert!(read("size_text", &TETRAHEDRON.replace("3 0 2 1", "three 0 2 1")).is_err());
}

#[test]
fn rejects_counts_that_disagree_with_the_content() {
    let short = TETRAHEDRON.replace("4 4 6", "4 5 6");
    assert!(read("too_few_faces", &short).is_err());
    let long = TETRAHEDRON.replace("4 4 6", "4 3 6");
    assert!(read("too_many_faces", &long).is_err());
    let vertices = TETRAHEDRON.replace("4 4 6", "5 4 6");
    assert!(read("too_few_vertices", &vertices).is_err());
}

#[test]
fn reports_the_line_of_an_error() {
    let error = read("line_number", &TETRAHEDRON.replace("0 1 0\n", "0 up 0\n"))
        .err()
        .unwrap();
    assert!(error.to_string().contains("line 5"), "{error}");
}

#[test]
fn a_huge_vertex_count_errors_instead_of_reserving_memory() {
    assert!(read("huge_vertices", "OFF\n18446744073709551615 0 0\n").is_err());
    assert!(read("many_vertices", "OFF\n99999999999999 0 0\n0 0 0\n").is_err());
}

#[test]
fn a_huge_face_size_errors_instead_of_panicking() {
    let text = TETRAHEDRON.replace("3 0 2 1", "18446744073709551615 0 2 1");
    assert!(read("huge_face", &text).is_err());
}

#[test]
fn missing_file_errors() {
    assert!(Mesh::<3>::read_off("target/does_not_exist.off").is_err());
}
