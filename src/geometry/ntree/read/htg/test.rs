use super::ReadHtg;
use crate::geometry::ntree::{
    Balancing, Input, Octree, Pairing, Quadtree, Rescaling,
    node::{Kind, Node},
    write::htg::WriteHtg,
};
use std::fs::{read_to_string, write};

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
        rescale: Rescaling {
            center: [4.0, 4.0, 4.0],
            cell: 1.0,
            half: 4.0,
        },
    };
    octree.subdivide(0).unwrap();
    let children = *octree.nodes[0].orthants().unwrap();
    for &child in &[children[1], children[2], children[4]] {
        octree.subdivide(child).unwrap();
    }
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
        rescale: Rescaling {
            center: [4.0, 4.0],
            cell: 1.0,
            half: 4.0,
        },
    };
    quadtree.subdivide(0).unwrap();
    quadtree
}

#[test]
fn round_trip_octree() {
    let original = "target/htg_octree_a.htg";
    octree().write_htg(original).unwrap();
    let read = Octree::<u16, usize>::read_htg(original).unwrap();
    assert_eq!(read.len(), 33);
    let reread = "target/htg_octree_b.htg";
    read.write_htg(reread).unwrap();
    assert_eq!(
        read_to_string(original).unwrap(),
        read_to_string(reread).unwrap()
    );
}

#[test]
fn round_trip_octree_compressed() {
    let path = "target/htg_octree_compressed.htg";
    octree().write_htg_compressed(path).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("compressor=\"vtkZLibDataCompressor\""));
    let read = Octree::<u16, usize>::read_htg(path).unwrap();
    assert_eq!(read.len(), 33);
}

#[test]
fn round_trip_quadtree_via_input() {
    let original = "target/htg_quadtree_a.htg";
    quadtree().write_htg(original).unwrap();
    let read = Quadtree::<u16, usize>::try_from(Input::Htg(original)).unwrap();
    assert_eq!(read.len(), 5);
    let reread = "target/htg_quadtree_b.htg";
    read.write_htg(reread).unwrap();
    assert_eq!(
        read_to_string(original).unwrap(),
        read_to_string(reread).unwrap()
    );
}

#[test]
fn unknown_compressor_is_unsupported() {
    let path = "target/htg_unknown_compressor.htg";
    write(
        path,
        "<VTKFile type=\"HyperTreeGrid\" compressor=\"vtkLZ4DataCompressor\"></VTKFile>",
    )
    .unwrap();
    let error = match Octree::<u16, usize>::read_htg(path) {
        Ok(_) => panic!("expected an unsupported-compressor error"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
}
