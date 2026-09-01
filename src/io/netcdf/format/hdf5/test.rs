use crate::io::netcdf::{GetVariable, NetCDF};
use std::fs;

fn open(name: &str, bytes: &[u8]) -> NetCDF {
    let path = format!("target/{name}");
    fs::write(&path, bytes).unwrap();
    NetCDF::open(&path).unwrap()
}

const PLAIN: &[u8] = include_bytes!("fixtures/mesh_plain.nc");
const DEFLATE: &[u8] = include_bytes!("fixtures/mesh_deflate.nc");
const DEFLATE_SHUFFLE: &[u8] = include_bytes!("fixtures/mesh_deflate_shuffle.nc");
const SMALL: &[u8] = include_bytes!("fixtures/small.nc");
const DENSE_INDIRECT: &[u8] = include_bytes!("fixtures/dense_indirect.nc");
const CHUNK_MULTILEVEL: &[u8] = include_bytes!("fixtures/chunk_multilevel.nc");
const UNALLOCATED: &[u8] = include_bytes!("fixtures/unallocated.nc");

fn check_mesh(nc: &NetCDF) {
    assert_eq!(nc.dimension_length("num_dim").unwrap(), 3);
    assert_eq!(nc.dimension_length("num_nodes").unwrap(), 6);
    assert_eq!(nc.dimension_length("num_el_in_blk1").unwrap(), 2);
    assert_eq!(nc.try_dimension_length("nope").unwrap(), None);
    assert_eq!(
        nc.get_variable::<f64>("coordx", 6).unwrap(),
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        nc.get_variable::<f64>("coordz", 6).unwrap(),
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(
        nc.get_variable::<f32>("nodal", 6).unwrap(),
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    );
    assert_eq!(
        nc.get_variable::<i32>("connect1", 8).unwrap(),
        [1, 2, 3, 4, 2, 3, 4, 5]
    );
    assert_eq!(nc.get_variable::<i32>("eb_prop1", 1).unwrap(), [100]);
    assert_eq!(
        nc.get_variable_attribute_text("connect1", "elem_type")
            .unwrap(),
        "TETRA4"
    );
    assert_eq!(nc.try_get_variable::<i32>("eb_names", 1).unwrap(), None);
    assert_eq!(nc.try_get_variable::<i32>("missing", 1).unwrap(), None);
}

#[test]
fn contiguous() {
    check_mesh(&open("hdf5_plain.nc", PLAIN));
}

#[test]
fn chunked_deflate() {
    check_mesh(&open("hdf5_deflate.nc", DEFLATE));
}

#[test]
fn chunked_deflate_shuffle() {
    check_mesh(&open("hdf5_deflate_shuffle.nc", DEFLATE_SHUFFLE));
}

#[test]
fn compact_links_coordinate_var_and_big_endian() {
    let nc = open("hdf5_small.nc", SMALL);
    assert_eq!(nc.dimension_length("n").unwrap(), 4);
    assert_eq!(
        nc.get_variable::<f64>("n", 4).unwrap(),
        [0.25, 0.5, 0.75, 1.0]
    );
    assert_eq!(
        nc.get_variable::<f64>("x", 4).unwrap(),
        [1.5, 2.5, 3.5, 4.5]
    );
    assert_eq!(nc.get_variable::<i32>("id", 4).unwrap(), [10, 20, 30, 40]);
    assert_eq!(
        nc.get_variable_attribute_text("x", "units").unwrap(),
        "meters"
    );
}

#[test]
fn dense_links_with_indirect_block() {
    let nc = open("hdf5_dense_indirect.nc", DENSE_INDIRECT);
    assert_eq!(nc.dimension_length("dimension_number_0").unwrap(), 1);
    assert_eq!(nc.dimension_length("dimension_number_9").unwrap(), 10);
    assert_eq!(
        nc.get_variable::<i32>("variable_number_9", 10).unwrap(),
        [7; 10]
    );
}

#[test]
fn chunked_multilevel_btree() {
    let nc = open("hdf5_chunk_multilevel.nc", CHUNK_MULTILEVEL);
    let seq = nc.get_variable::<i32>("seq", 400).unwrap();
    assert_eq!(seq, (0..400).collect::<Vec<_>>());
}

#[test]
fn unallocated_variable_reads_as_zero() {
    let nc = open("hdf5_unallocated.nc", UNALLOCATED);
    assert_eq!(
        nc.get_variable::<f64>("a", 4).unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(nc.get_variable::<f64>("b", 4).unwrap(), [0.0; 4]);
}

#[test]
#[should_panic(expected = "unsupported HDF5 superblock version")]
fn rejects_unknown_superblock_version() {
    let mut bytes = vec![0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];
    bytes.extend([9, 8, 8, 0]);
    bytes.resize(64, 0);
    let path = "target/hdf5_bad_superblock.nc";
    fs::write(path, &bytes).unwrap();
    NetCDF::open(path).unwrap();
}

#[test]
#[should_panic(expected = "expected a version 2 object header")]
fn rejects_non_v2_object_header() {
    let mut bytes = vec![0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];
    bytes.extend([2, 8, 8, 0]);
    bytes.extend((0u64).to_le_bytes());
    bytes.extend((!0u64).to_le_bytes());
    bytes.extend((0u64).to_le_bytes());
    bytes.extend((64u64).to_le_bytes());
    bytes.resize(72, 0);
    let path = "target/hdf5_bad_header.nc";
    fs::write(path, &bytes).unwrap();
    NetCDF::open(path).unwrap();
}
