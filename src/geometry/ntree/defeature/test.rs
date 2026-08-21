use crate::geometry::{
    grid::{Pixels, Voxels},
    ntree::{Octree, Quadtree},
};

#[test]
fn absorbs_single_pixel_blob() {
    let mut data = vec![1u8; 16];
    data[1 + 4] = 2;
    let mut quadtree = Quadtree::<u16, usize, u8>::try_from(Pixels::new(data, [4, 4])).unwrap();
    quadtree.defeature(2);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(*back.nel(), [4, 4]);
    assert_eq!(back.data(), [1u8; 16]);
}

#[test]
fn absorbs_single_voxel_blob() {
    let mut data = vec![1u8; 64];
    data[1 + 4 + 16] = 2;
    let mut octree = Octree::<u16, usize, u8>::try_from(Voxels::new(data, [4, 4, 4])).unwrap();
    octree.defeature(2);
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(*back.nel(), [4, 4, 4]);
    assert_eq!(back.data(), [1u8; 64]);
}

#[test]
fn keeps_blob_at_or_above_threshold() {
    let mut data = vec![1u8; 16];
    data[0] = 2;
    let mut quadtree =
        Quadtree::<u16, usize, u8>::try_from(Pixels::new(data.clone(), [4, 4])).unwrap();
    quadtree.defeature(1);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(back.data(), data);
}

#[test]
fn absorbs_after_balancing() {
    use crate::geometry::ntree::{Balance, balance::Balancing};
    let mut data = vec![1u8; 64];
    data[1 + 4 + 16] = 2;
    let mut octree = Octree::<u16, usize, u8>::try_from(Voxels::new(data, [4, 4, 4])).unwrap();
    octree.balance(Balancing::Strong(1));
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
    let mut quadtree = Quadtree::<u16, usize, u8>::try_from(Pixels::new(data, [4, 4])).unwrap();
    quadtree.defeature(2);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(back.data()[5], 3);
    assert!(!back.data().contains(&2));
}

#[test]
fn absorbs_pixel_protrusion_within_a_large_cluster() {
    let mut data = vec![1u8; 16];
    data[5] = 2;
    let mut quadtree = Quadtree::<u16, usize, u8>::try_from(Pixels::new(data, [4, 4])).unwrap();
    quadtree.defeature(1);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(back.data(), [1u8; 16]);
}

#[test]
fn absorbs_voxel_protrusion_within_a_large_cluster() {
    let mut data = vec![1u8; 64];
    data[5] = 2;
    let mut octree = Octree::<u16, usize, u8>::try_from(Voxels::new(data, [4, 4, 4])).unwrap();
    octree.defeature(1);
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(back.data(), [1u8; 64]);
}

#[test]
fn keeps_voxels_below_the_protrusion_threshold() {
    let mut data = vec![1u8; 64];
    data[1] = 2;
    data[2] = 2;
    let mut octree =
        Octree::<u16, usize, u8>::try_from(Voxels::new(data.clone(), [4, 4, 4])).unwrap();
    octree.defeature(1);
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(back.data(), data);
}
