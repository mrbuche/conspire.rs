use super::WriteHtg;
use crate::geometry::{
    grid::Voxels,
    ntree::{
        Balancing, Octree, Pairing, Quadtree, Rescaling,
        node::{Kind, Node},
    },
};
use std::fs::read_to_string;

fn octree() -> Octree<u16, usize> {
    let mut octree = Octree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0, 0],
            length: 8,
            facets: [None; 6],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        pairing_vertices: Default::default(),
        rescale: Rescaling {
            center: [4.0, 4.0, 4.0],
            cell: 1.0,
            half: 4.0,
        },
    };
    octree.subdivide(0).unwrap();
    octree
}

fn quadtree() -> Quadtree<u16, usize> {
    let mut quadtree = Quadtree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0],
            length: 8,
            facets: [None; 4],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        pairing_vertices: Default::default(),
        rescale: Rescaling {
            center: [4.0, 4.0],
            cell: 1.0,
            half: 4.0,
        },
    };
    quadtree.subdivide(0).unwrap();
    quadtree
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
    let chars: Vec<u8> = text.bytes().filter(|b| *b != b'=').collect();
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

fn bytes(contents: &str, name: &str) -> Vec<u8> {
    let at = contents.find(&format!("Name=\"{name}\"")).unwrap();
    let key = "format=\"binary\">";
    let start = contents[at..].find(key).unwrap() + at + key.len();
    let end = contents[start..].find('<').unwrap() + start;
    unbase64(&contents[start..end]).split_off(8)
}

fn ints(contents: &str, name: &str) -> Vec<i64> {
    bytes(contents, name)
        .chunks(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn floats(contents: &str, name: &str) -> Vec<f64> {
    bytes(contents, name)
        .chunks(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn descriptor_bits(contents: &str, nbits: usize) -> Vec<u8> {
    let packed = bytes(contents, "Descriptor");
    (0..nbits)
        .map(|i| packed[i / 8] >> (7 - i % 8) & 1)
        .collect()
}

#[test]
fn octree_once() {
    let path = "target/octree.htg";
    octree().write_htg(path).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("Dimensions=\"2 2 2\""));
    assert!(contents.contains("NumberOfLevels=\"2\" NumberOfVertices=\"9\""));
    assert_eq!(floats(&contents, "XCoordinates"), [0.0, 8.0]);
    assert_eq!(ints(&contents, "NbVerticesByLevel"), [1, 8]);
    assert_eq!(descriptor_bits(&contents, 1), [1]);
    assert_eq!(ints(&contents, "Depth"), [0, 1, 1, 1, 1, 1, 1, 1, 1]);
}

#[test]
fn octree_asymmetric() {
    let mut octree = octree();
    let children = *octree.nodes[0].orthants().unwrap();
    for &child in &[children[1], children[2], children[4]] {
        octree.subdivide(child).unwrap();
    }
    let path = "target/octree_asymmetric.htg";
    octree.write_htg(path).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("NumberOfLevels=\"3\" NumberOfVertices=\"33\""));
    assert_eq!(ints(&contents, "NbVerticesByLevel"), [1, 8, 24]);
    assert_eq!(descriptor_bits(&contents, 9), [1, 0, 1, 1, 0, 1, 0, 0, 0]);
}

#[test]
fn geometric_tree_has_no_value_array() {
    let path = "target/octree_no_value.htg";
    octree().write_htg(path).unwrap();
    assert!(!read_to_string(path).unwrap().contains("Name=\"Value\""));
}

#[test]
fn valued_octree_writes_value_array() {
    let data: Vec<u8> = (1..=8).collect();
    let octree = Octree::<u16, usize, u8>::from(Voxels::new(data, [2, 2, 2]));
    let path = "target/octree_value.htg";
    octree.write_htg(path).unwrap();
    let contents = read_to_string(path).unwrap();
    let values = floats(&contents, "Value");
    assert!(values[0].is_nan());
    assert_eq!(values[1..], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn quadtree_once() {
    let path = "target/quadtree.htg";
    quadtree().write_htg(path).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("Dimensions=\"2 2 1\""));
    assert!(contents.contains("NumberOfLevels=\"2\" NumberOfVertices=\"5\""));
    assert_eq!(floats(&contents, "XCoordinates"), [0.0, 8.0]);
    assert_eq!(floats(&contents, "ZCoordinates"), [0.0]);
    assert_eq!(ints(&contents, "NbVerticesByLevel"), [1, 4]);
    assert_eq!(descriptor_bits(&contents, 1), [1]);
    assert_eq!(ints(&contents, "Depth"), [0, 1, 1, 1, 1]);
}
