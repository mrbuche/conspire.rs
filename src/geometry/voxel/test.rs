use crate::{
    geometry::voxel::{Input, Output, Voxels},
    io::{Npy, Write},
};

#[test]
fn round_trip_npy() {
    let data: Vec<u8> = (0..24).collect();
    let path = "target/voxels.npy";
    Voxels::new(data.clone(), [2, 3, 4])
        .write(Output::Npy(path))
        .unwrap();
    let read = Voxels::<3, u8>::try_from(Input::Npy(path)).unwrap();
    assert_eq!(read.data(), data);
    assert_eq!(read.nel(), &[2, 3, 4]);
}

#[test]
fn reads_c_order_with_transpose() {
    let path = "target/voxels_c.npy";
    Npy {
        data: vec![0u8, 1, 10, 11],
        shape: vec![2, 2],
        fortran_order: false,
    }
    .write(path)
    .unwrap();
    let voxels = Voxels::<2, u8>::try_from(Input::Npy(path)).unwrap();
    assert_eq!(voxels[[0, 0]], 0);
    assert_eq!(voxels[[1, 0]], 10);
    assert_eq!(voxels[[0, 1]], 1);
    assert_eq!(voxels[[1, 1]], 11);
}

#[test]
fn dimension_mismatch_errors() {
    let path = "target/voxels_2d.npy";
    Voxels::new(vec![0u8; 6], [2, 3])
        .write(Output::Npy(path))
        .unwrap();
    assert!(Voxels::<3, u8>::try_from(Input::Npy(path)).is_err());
}
