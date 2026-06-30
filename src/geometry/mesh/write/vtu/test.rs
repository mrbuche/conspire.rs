use crate::{
    geometry::mesh::{Connectivity, Mesh, Output},
    io::Write,
};
use std::fs::read_to_string;

fn mixed() -> Mesh<3> {
    let connectivities = vec![
        Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
        Connectivity::Wedge(vec![[4, 5, 8, 7, 6, 9]].into()),
        Connectivity::Pyramidal(vec![[1, 2, 6, 5, 10]].into()),
        Connectivity::Tetrahedral(vec![[1, 2, 10, 11]].into()),
    ];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 2.0],
        [0.5, 1.0, 2.0],
        [2.0, 0.5, 0.5],
        [1.5, 0.5, -1.0],
    ]
    .into();
    Mesh::from((connectivities, coordinates))
}

fn payload<'a>(contents: &'a str, name: &str) -> &'a str {
    let marker = format!("Name=\"{name}\" format=\"binary\">");
    let start = contents.find(&marker).unwrap() + marker.len();
    let end = contents[start..].find("</DataArray>").unwrap() + start;
    &contents[start..end]
}

fn unbase64(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => (c - b'A') as u32,
        b'a'..=b'z' => (c - b'a' + 26) as u32,
        b'0'..=b'9' => (c - b'0' + 52) as u32,
        b'+' => 62,
        b'/' => 63,
        _ => 0,
    };
    let chars: Vec<u8> = text.bytes().filter(|&b| b != b'=').collect();
    let mut out = Vec::new();
    for chunk in chars.chunks(4) {
        let mut block = 0u32;
        for &c in chunk {
            block = (block << 6) | value(c);
        }
        block <<= 6 * (4 - chunk.len());
        for shift in (0..chunk.len() - 1).map(|i| 16 - 8 * i) {
            out.push((block >> shift) as u8);
        }
    }
    out
}

#[test]
fn mixed_unstructured_grid() {
    let path = "target/mixed.vtu";
    mixed().write(Output::Vtu(path)).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("type=\"UnstructuredGrid\""));
    assert!(contents.contains("byte_order=\"LittleEndian\""));
    assert!(contents.contains("NumberOfPoints=\"12\" NumberOfCells=\"4\""));
    let types = unbase64(payload(&contents, "types"));
    assert_eq!(u64::from_le_bytes(types[0..8].try_into().unwrap()), 4);
    assert_eq!(&types[8..], &[12, 13, 14, 10]);
    let offsets = unbase64(payload(&contents, "offsets"));
    let offsets: Vec<i64> = offsets[8..]
        .chunks(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(offsets, [8, 14, 19, 23]);
    let connectivity = unbase64(payload(&contents, "connectivity"));
    let first8: Vec<i64> = connectivity[8..]
        .chunks(8)
        .take(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(first8, [0, 1, 2, 3, 4, 5, 6, 7]);
}
