use crate::geometry::{
    grid::{Pixels, Voxels},
    ntree::{Octree, Quadtree},
};

fn speckled(n: usize) -> Voxels<u8> {
    let mut data = vec![1u8; n * n * n];
    for z in (0..n).step_by(2) {
        for y in (0..n).step_by(2) {
            for x in (0..n).step_by(2) {
                data[x + n * y + n * n * z] = 2;
            }
        }
    }
    Voxels::new(data, [n, n, n])
}

#[test]
fn absorbs_single_pixel_blob() {
    let mut data = vec![1u8; 16];
    data[1 + 4] = 2;
    let mut quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(data, [4, 4]));
    quadtree.defeature(2);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(*back.nel(), [4, 4]);
    assert_eq!(back.data(), [1u8; 16]);
}

#[test]
fn absorbs_single_voxel_blob() {
    let mut data = vec![1u8; 64];
    data[1 + 4 + 16] = 2;
    let mut octree = Octree::<u16, usize, u8>::from(Voxels::new(data, [4, 4, 4]));
    octree.defeature(2);
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(*back.nel(), [4, 4, 4]);
    assert_eq!(back.data(), [1u8; 64]);
}

#[test]
fn keeps_blob_at_or_above_threshold() {
    let mut data = vec![1u8; 16];
    data[1 + 4] = 2;
    let mut quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(data.clone(), [4, 4]));
    quadtree.defeature(1);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(back.data(), data);
}

#[test]
fn absorbs_after_balancing() {
    use crate::geometry::ntree::{Balance, balance::Balancing};
    let mut data = vec![1u8; 64];
    data[1 + 4 + 16] = 2;
    let mut octree = Octree::<u16, usize, u8>::from(Voxels::new(data, [4, 4, 4]));
    octree.balance(Balancing::Strong);
    octree.defeature(2);
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(back.data(), [1u8; 64]);
}

#[test]
fn absorbs_into_largest_shared_area_neighbor() {
    let mut data = vec![1u8; 16];
    for &flat in &[0, 4, 8, 12, 1, 9, 13] {
        data[flat] = 3;
    }
    data[5] = 2;
    let mut quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(data, [4, 4]));
    quadtree.defeature(2);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(back.data()[5], 3);
    assert!(!back.data().contains(&2));
}
