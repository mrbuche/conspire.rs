use crate::geometry::{
    grid::{Pixels, Voxels},
    ntree::{Octree, Quadtree},
};

fn octree_leaves(octree: &Octree<u16, usize, u8>) -> usize {
    octree.iter().filter(|node| node.is_leaf()).count()
}

fn quadtree_leaves(quadtree: &Quadtree<u16, usize, u8>) -> usize {
    quadtree.iter().filter(|node| node.is_leaf()).count()
}

#[test]
fn octree_homogeneous_stays_root() {
    let octree = Octree::<u16, usize, u8>::from(Voxels::new(vec![1u8; 8], [2, 2, 2]));
    assert_eq!(octree.len(), 1);
    assert_eq!(octree_leaves(&octree), 1);
}

#[test]
fn octree_heterogeneous_subdivides_once() {
    let mut data = vec![1u8; 8];
    data[7] = 2;
    let octree = Octree::<u16, usize, u8>::from(Voxels::new(data, [2, 2, 2]));
    assert_eq!(octree.len(), 9);
    assert_eq!(octree_leaves(&octree), 8);
}

#[test]
fn octree_non_power_of_two_pads() {
    let octree = Octree::<u16, usize, u8>::from(Voxels::new(vec![1u8; 27], [3, 3, 3]));
    assert_eq!(octree.rescale().center, [2.0; 3]);
    assert!(octree_leaves(&octree) > 1);
}

#[test]
fn quadtree_homogeneous_stays_root() {
    let quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(vec![1u8; 4], [2, 2]));
    assert_eq!(quadtree.len(), 1);
    assert_eq!(quadtree_leaves(&quadtree), 1);
}

#[test]
fn quadtree_heterogeneous_subdivides_once() {
    let mut data = vec![1u8; 4];
    data[3] = 2;
    let quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(data, [2, 2]));
    assert_eq!(quadtree.len(), 5);
    assert_eq!(quadtree_leaves(&quadtree), 4);
}

#[test]
fn quadtree_non_power_of_two_pads() {
    let quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(vec![1u8; 9], [3, 3]));
    assert_eq!(quadtree.rescale().center, [2.0; 2]);
    assert!(quadtree_leaves(&quadtree) > 1);
}

#[test]
fn octree_round_trip() {
    let data: Vec<u8> = (1..=8).collect();
    let octree = Octree::<u16, usize, u8>::from(Voxels::new(data.clone(), [2, 2, 2]));
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(*back.nel(), [2, 2, 2]);
    assert_eq!(back.data(), data);
}

#[test]
fn octree_round_trip_non_power_of_two() {
    let data: Vec<u8> = (0..27).collect();
    let octree = Octree::<u16, usize, u8>::from(Voxels::new(data.clone(), [3, 3, 3]));
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(*back.nel(), [3, 3, 3]);
    assert_eq!(back.data(), data);
}

#[test]
fn refining_a_valued_leaf_inherits_and_clears_parent() {
    let mut octree = Octree::<u16, usize, u8>::from(Voxels::new(vec![7u8; 8], [2, 2, 2]));
    assert_eq!(octree.nodes[0].value, Some(7));
    octree.subdivide(0).unwrap();
    assert_eq!(octree.nodes[0].value, None);
    let children = *octree.nodes[0].orthants().unwrap();
    assert!(children.iter().all(|&c| octree.nodes[c].value == Some(7)));
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(*back.nel(), [2, 2, 2]);
    assert_eq!(back.data(), [7; 8]);
}

#[test]
fn quadtree_round_trip_non_power_of_two() {
    let data: Vec<u8> = (0..6).collect();
    let quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(data.clone(), [3, 2]));
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(*back.nel(), [3, 2]);
    assert_eq!(back.data(), data);
}
