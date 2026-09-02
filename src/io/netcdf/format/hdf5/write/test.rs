use super::message::{attr_f32, attr_i32, datatype, fill_value};
use crate::io::netcdf::{
    DefineVariable, GetVariable, NetCDF, PutVariable,
    format::{DimSpec, NC_DOUBLE, NC_INT, Storage, VarSpec, hdf5::write},
};

fn spec(name: &str, xtype: i32, dimids: Vec<usize>) -> VarSpec {
    VarSpec {
        name: name.to_string(),
        xtype,
        dimids,
        atts: Vec::new(),
        begin: 0,
        vsize: 0,
        storage: Storage::Classic,
    }
}

fn le<T: Copy>(values: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(values);
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), n) }.to_vec()
}

fn write_to_vec(dims: &[DimSpec], vars: &[VarSpec], data: Vec<Vec<u8>>, threads: usize) -> Vec<u8> {
    let mut out = Vec::new();
    write(dims, &[], vars, data, threads, &mut out).unwrap();
    out
}

#[test]
fn api_round_trip_scalar_contiguous_and_chunked() {
    let path = "target/hdf5_write_api.nc";
    let coordx = vec![0.5f64, 1.5, 2.5, 3.5];
    let temperature = vec![-1.0f32, 0.0, 1.0, 2.0];
    let connect = vec![1i32, 2, 3, 4, 2, 3];
    let count = vec![42i32];
    {
        let mut nc = NetCDF::create_netcdf4(path, 1).unwrap();
        nc.global();
        nc.define_dimension("nodes", 4).unwrap();
        nc.define_dimension("elems", 2).unwrap();
        nc.define_dimension("corners", 3).unwrap();
        nc.define_variable::<f64>("coordx", 1, &["nodes"]).unwrap();
        nc.define_variable::<f32>("temperature", 1, &["nodes"])
            .unwrap();
        nc.define_variable::<i32>("connect1", 2, &["elems", "corners"])
            .unwrap();
        nc.define_variable::<i32>("count", 0, &[]).unwrap();
        nc.put_variable_attribute_text("connect1", "elem_type", "TRI3")
            .unwrap();
        nc.end_definition();
        nc.put_variable("coordx", &coordx).unwrap();
        nc.put_variable("temperature", &temperature).unwrap();
        nc.put_variable("connect1", &connect).unwrap();
        nc.put_variable("count", &count).unwrap();
    }
    let nc = NetCDF::open(path).unwrap();
    assert_eq!(nc.dimension_length("nodes").unwrap(), 4);
    assert_eq!(nc.dimension_length("corners").unwrap(), 3);
    assert_eq!(nc.get_variable::<f64>("coordx", 4).unwrap(), coordx);
    assert_eq!(
        nc.get_variable::<f32>("temperature", 4).unwrap(),
        temperature
    );
    assert_eq!(
        nc.get_variable_slice::<f32>("temperature", &[1], &[2])
            .unwrap(),
        &temperature[1..3]
    );
    assert_eq!(nc.get_variable::<i32>("connect1", 6).unwrap(), connect);
    assert_eq!(nc.get_variable::<i32>("count", 1).unwrap(), count);
    assert_eq!(
        nc.get_variable_attribute_text("connect1", "elem_type")
            .unwrap(),
        "TRI3"
    );
}

#[test]
fn multi_chunk_variable_round_trips() {
    let path = "target/hdf5_write_multichunk.nc";
    let n = 400_000usize;
    let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    {
        let mut nc = NetCDF::create_netcdf4(path, 4).unwrap();
        nc.define_dimension("row", n).unwrap();
        nc.define_variable::<f64>("x", 1, &["row"]).unwrap();
        nc.end_definition();
        nc.put_variable("x", &values).unwrap();
    }
    let nc = NetCDF::open(path).unwrap();
    assert_eq!(nc.get_variable::<f64>("x", n).unwrap(), values);
}

#[test]
fn btree_final_key_is_chunk_aligned_past_the_end() {
    use super::chunk::{btree_key_size, btree_node, plan};
    let rows = 1_030_301u64;
    let elem = 8usize;
    let p = plan(&[rows], &vec![0u8; rows as usize * elem], elem, 1);
    let slab = p.chunk_dims[0];
    let n = p.blobs.len();
    assert!(
        n > 1 && !rows.is_multiple_of(slab),
        "want a partial trailing chunk"
    );
    let node = btree_node(&p, &vec![0u64; n]);
    let off = 24 + n * (btree_key_size(1) + 8) + 8;
    let final_dim0 = u64::from_le_bytes(node[off..off + 8].try_into().unwrap());
    assert_eq!(final_dim0 % slab, 0);
    assert!(final_dim0 / slab > (n as u64 - 1));
}

#[test]
fn multi_chunk_output_is_deterministic() {
    let n = 300_000usize;
    let values: Vec<f64> = (0..n).map(|i| (i % 13) as f64).collect();
    let dims = [DimSpec {
        name: "row".to_string(),
        len: n as u64,
    }];
    let vars = [spec("x", NC_DOUBLE, vec![0])];
    let a = write_to_vec(&dims, &vars, vec![le(&values)], 2);
    let b = write_to_vec(&dims, &vars, vec![le(&values)], 5);
    assert_eq!(a, b);
}

#[test]
fn incompressible_chunk_is_stored_raw() {
    let path = "target/hdf5_write_raw.nc";
    let values: Vec<i32> = vec![0x1234_5678, -0x0765_4321, 0x0abc_def0, 0x7fff_fffe];
    {
        let mut nc = NetCDF::create_netcdf4(path, 1).unwrap();
        nc.define_dimension("k", 4).unwrap();
        nc.define_variable::<i32>("v", 1, &["k"]).unwrap();
        nc.end_definition();
        nc.put_variable("v", &values).unwrap();
    }
    let nc = NetCDF::open(path).unwrap();
    assert_eq!(nc.get_variable::<i32>("v", 4).unwrap(), values);
}

#[test]
fn multi_element_numeric_attributes_use_an_array_dataspace() {
    assert!(attr_i32("dims", &[3, 1]).len() > attr_i32("dims", &[3]).len());
    assert!(attr_f32("scale", &[0.5, 2.0]).len() > attr_f32("scale", &[0.5]).len());
}

#[test]
fn write_function_smoke() {
    let dims = [DimSpec {
        name: "n".to_string(),
        len: 3,
    }];
    let vars = [spec("x", NC_DOUBLE, vec![0]), spec("s", NC_INT, vec![])];
    let bytes = write_to_vec(&dims, &vars, vec![le(&[1.0f64, 2.0, 3.0]), le(&[7i32])], 1);
    assert_eq!(
        &bytes[..8],
        &[0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n']
    );
    let parsed = crate::io::netcdf::format::hdf5::parse(&bytes, 0);
    assert!(matches!(
        parsed.vars.iter().find(|v| v.name == "s").unwrap().storage,
        Storage::Hdf5 { .. }
    ));
}

#[test]
#[should_panic(expected = "cannot write netCDF external type")]
fn write_rejects_unsupported_variable_type() {
    let dims = [DimSpec {
        name: "n".to_string(),
        len: 2,
    }];
    write_to_vec(&dims, &[spec("bad", 99, vec![0])], vec![le(&[0u8, 0])], 0);
}

#[test]
#[should_panic(expected = "cannot write netCDF external type")]
fn datatype_rejects_unsupported_type() {
    datatype(99);
}

#[test]
#[should_panic(expected = "cannot write netCDF external type")]
fn fill_value_rejects_unsupported_type() {
    fill_value(99);
}

#[test]
fn hyperslab_across_chunk_boundaries() {
    let path = "target/hdf5_write_slab.nc";
    let n = 400_000usize;
    let values: Vec<i32> = (0..n as i32).collect();
    {
        let mut nc = NetCDF::create_netcdf4(path, 4).unwrap();
        nc.define_dimension("row", n).unwrap();
        nc.define_variable::<i32>("x", 1, &["row"]).unwrap();
        nc.end_definition();
        nc.put_variable("x", &values).unwrap();
    }
    let nc = NetCDF::open(path).unwrap();
    let slab = nc
        .get_variable_slice::<i32>("x", &[262_000], &[500])
        .unwrap();
    assert_eq!(slab, &values[262_000..262_500]);
    assert_eq!(
        nc.get_variable_slice::<i32>("x", &[0], &[3]).unwrap(),
        &values[0..3]
    );
    assert_eq!(
        nc.get_variable_slice::<i32>("x", &[n - 4], &[4]).unwrap(),
        &values[n - 4..]
    );
}

#[test]
fn hyperslab_two_dimensional() {
    let path = "target/hdf5_write_slab2d.nc";
    let (rows, cols) = (5usize, 4usize);
    let values: Vec<f64> = (0..(rows * cols) as i32).map(f64::from).collect();
    {
        let mut nc = NetCDF::create_netcdf4(path, 1).unwrap();
        nc.define_dimension("r", rows).unwrap();
        nc.define_dimension("c", cols).unwrap();
        nc.define_variable::<f64>("m", 2, &["r", "c"]).unwrap();
        nc.end_definition();
        nc.put_variable("m", &values).unwrap();
    }
    let nc = NetCDF::open(path).unwrap();
    let block = nc.get_variable_slice::<f64>("m", &[1, 1], &[3, 2]).unwrap();
    assert_eq!(block, [5.0, 6.0, 9.0, 10.0, 13.0, 14.0]);
}

#[test]
#[should_panic(expected = "slice out of bounds")]
fn hyperslab_out_of_bounds_panics() {
    let path = "target/hdf5_write_slab_oob.nc";
    {
        let mut nc = NetCDF::create_netcdf4(path, 2).unwrap();
        nc.define_dimension("k", 4).unwrap();
        nc.define_variable::<i32>("v", 1, &["k"]).unwrap();
        nc.end_definition();
        nc.put_variable("v", &[1i32, 2, 3, 4]).unwrap();
    }
    let nc = NetCDF::open(path).unwrap();
    nc.get_variable_slice::<i32>("v", &[2], &[5]).unwrap();
}
