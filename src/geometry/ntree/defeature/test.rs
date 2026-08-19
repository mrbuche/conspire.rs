use crate::geometry::{
    grid::{Pixels, Voxels},
    ntree::{Octree, Quadtree},
};

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
    // Pixel 0 sits in a domain corner, so only two of its four facets have
    // neighbors: below the M - 1 = 3 protrusion threshold, so a minimum of 1
    // (satisfied by its own single-pixel volume) leaves it untouched.
    let mut data = vec![1u8; 16];
    data[0] = 2;
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
    let mut quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(data, [4, 4]));
    quadtree.defeature(2);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(back.data()[5], 3);
    assert!(!back.data().contains(&2));
}

#[test]
fn absorbs_pixel_protrusion_within_a_large_cluster() {
    // Pixel 5 (interior, all four faces present) differs from three of its
    // four neighbors, meeting the M - 1 = 3 threshold even though its
    // cluster (itself) already satisfies the volume minimum.
    let mut data = vec![1u8; 16];
    data[5] = 2;
    let mut quadtree = Quadtree::<u16, usize, u8>::from(Pixels::new(data, [4, 4]));
    quadtree.defeature(1);
    let back = Pixels::<u8>::from(&quadtree);
    assert_eq!(back.data(), [1u8; 16]);
}

#[test]
fn absorbs_voxel_protrusion_within_a_large_cluster() {
    // Voxel 5 = (1, 1, 0) sits on a single domain face (only its z- facet is
    // a boundary), so five of its six facets have neighbors; all five
    // differ, meeting the M - 1 = 5 threshold even though its cluster
    // already satisfies the volume minimum.
    let mut data = vec![1u8; 64];
    data[5] = 2;
    let mut octree = Octree::<u16, usize, u8>::from(Voxels::new(data, [4, 4, 4]));
    octree.defeature(1);
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(back.data(), [1u8; 64]);
}

#[test]
fn keeps_voxels_below_the_protrusion_threshold() {
    // Voxels 1 = (1, 0, 0) and 2 = (2, 0, 0) sit on a domain edge (only four
    // of six facets present) and share a facet with each other, so each
    // differs from only three of its present neighbors: below the M - 1 = 5
    // threshold, so neither is a protrusion.
    let mut data = vec![1u8; 64];
    data[1] = 2;
    data[2] = 2;
    let mut octree = Octree::<u16, usize, u8>::from(Voxels::new(data.clone(), [4, 4, 4]));
    octree.defeature(1);
    let back = Voxels::<u8>::from(&octree);
    assert_eq!(back.data(), data);
}
