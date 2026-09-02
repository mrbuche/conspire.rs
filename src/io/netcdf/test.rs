use crate::io::netcdf::{DefineVariable, GetVariable, NetCDF, PutVariable};
use std::path::Path;

#[test]
fn round_trip() {
    let path = "target/netcdf_round_trip.nc";
    let coordx = vec![0.0_f64, 1.0, 2.0];
    let temperature = vec![0.5_f32, 1.5, 2.5];
    let connectivity = vec![0_i32, 1, 2];
    {
        let mut netcdf = NetCDF::try_from(Path::new(path)).unwrap();
        netcdf.define_dimension("nodes", 3).unwrap();
        netcdf
            .define_variable::<f64>("coordx", 1, &["nodes"])
            .unwrap();
        netcdf
            .define_variable::<f32>("temperature", 1, &["nodes"])
            .unwrap();
        netcdf
            .define_variable::<i32>("connectivity", 1, &["nodes"])
            .unwrap();
        netcdf
            .put_variable_attribute_text("coordx", "units", "meters")
            .unwrap();
        netcdf.end_definition();
        netcdf.put_variable("coordx", &coordx).unwrap();
        netcdf.put_variable("temperature", &temperature).unwrap();
        netcdf.put_variable("connectivity", &connectivity).unwrap();
    }
    let netcdf = NetCDF::open(path).unwrap();
    assert_eq!(netcdf.dimension_length("nodes").unwrap(), 3);
    assert_eq!(netcdf.try_dimension_length("nodes").unwrap(), Some(3));
    assert_eq!(netcdf.try_dimension_length("missing").unwrap(), None);
    assert_eq!(netcdf.get_variable::<f64>("coordx", 3).unwrap(), coordx);
    assert_eq!(
        netcdf
            .get_variable_slice::<f64>("coordx", &[1], &[2])
            .unwrap(),
        &coordx[1..3]
    );
    assert_eq!(
        netcdf
            .get_variable_slice::<f32>("temperature", &[0], &[2])
            .unwrap(),
        &temperature[0..2]
    );
    assert_eq!(
        netcdf
            .get_variable_slice::<i32>("connectivity", &[2], &[1])
            .unwrap(),
        &connectivity[2..3]
    );
    assert_eq!(
        netcdf.get_variable::<f32>("temperature", 3).unwrap(),
        temperature
    );
    assert_eq!(
        netcdf.get_variable::<i32>("connectivity", 3).unwrap(),
        connectivity
    );
    assert_eq!(
        netcdf.try_get_variable::<f64>("coordx", 3).unwrap(),
        Some(coordx)
    );
    assert_eq!(netcdf.try_get_variable::<i32>("missing", 3).unwrap(), None);
    assert_eq!(
        netcdf.try_get_variable::<i32>("connectivity", 3).unwrap(),
        Some(connectivity)
    );
    assert_eq!(
        netcdf
            .get_variable_attribute_text("coordx", "units")
            .unwrap(),
        "meters"
    );
}

#[test]
#[cfg(unix)]
fn non_utf8_path_errors() {
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};
    let path = OsStr::from_bytes(&[0xff, 0x2f, 0x66]);
    let error = NetCDF::try_from(Path::new(path)).err().unwrap();
    assert_eq!(error.to_string(), "path is not valid UTF-8");
    assert_eq!(format!("{error:?}"), "path is not valid UTF-8");
}

#[test]
fn try_from_nul_path_errors() {
    let error = NetCDF::try_from(Path::new("bad\0.nc")).err().unwrap();
    let nul = std::ffi::CString::new("bad\0.nc").unwrap_err();
    assert_eq!(error.to_string(), nul.to_string());
    assert_eq!(format!("{error:?}"), nul.to_string());
}

#[test]
fn interior_nul_errors() {
    assert!(NetCDF::create("bad\0.nc").is_err());
    assert!(NetCDF::open("bad\0.nc").is_err());
    let mut netcdf = NetCDF::create("target/netcdf_nul.nc").unwrap();
    assert!(netcdf.define_dimension("d\0", 1).is_err());
    assert!(netcdf.dimension_length("d\0").is_err());
    assert!(netcdf.try_dimension_length("d\0").is_err());
    assert!(netcdf.put_variable_attribute_text("v\0", "a", "x").is_err());
    assert!(netcdf.get_variable_attribute_text("v\0", "a").is_err());
    assert!(netcdf.define_variable::<i32>("v\0", 0, &[]).is_err());
    assert!(netcdf.define_variable::<i32>("v", 1, &["d\0"]).is_err());
    assert!(netcdf.put_variable("v\0", &[1_i32]).is_err());
    assert!(netcdf.get_variable::<i32>("v\0", 1).is_err());
    assert!(netcdf.try_get_variable::<i32>("v\0", 1).is_err());

    assert!(netcdf.define_variable::<f64>("v\0", 0, &[]).is_err());
    assert!(netcdf.define_variable::<f64>("v", 1, &["d\0"]).is_err());
    assert!(netcdf.put_variable("v\0", &[1.0_f64]).is_err());
    assert!(netcdf.get_variable::<f64>("v\0", 1).is_err());
    assert!(netcdf.try_get_variable::<f64>("v\0", 1).is_err());

    assert!(netcdf.define_variable::<f32>("v\0", 0, &[]).is_err());
    assert!(netcdf.define_variable::<f32>("v", 1, &["d\0"]).is_err());
    assert!(netcdf.put_variable("v\0", &[1.0_f32]).is_err());
    assert!(netcdf.get_variable::<f32>("v\0", 1).is_err());
    assert!(netcdf.try_get_variable::<f32>("v\0", 1).is_err());
}

#[test]
fn interior_nul_attribute_of_existing_variable_errors() {
    let path = "target/netcdf_nul_attr.nc";
    let mut netcdf = NetCDF::create(path).unwrap();
    netcdf.define_dimension("nodes", 1).unwrap();
    netcdf.define_variable::<i32>("var", 1, &["nodes"]).unwrap();
    netcdf.end_definition();
    assert!(netcdf.get_variable_attribute_text("var", "a\0").is_err());
    assert!(
        netcdf
            .put_variable_attribute_text("var", "a\0", "x")
            .is_err()
    );
    assert!(
        netcdf
            .put_variable_attribute_text("var", "a", "x\0")
            .is_err()
    );
}

#[test]
fn poisoned_lock_recovers() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _guard = super::nc_lock();
        panic!("poison the lock");
    }));
    let _guard = super::nc_lock();
}

#[test]
fn queries_before_end_definition() {
    let mut netcdf = NetCDF::create("target/netcdf_writer_queries.nc").unwrap();
    netcdf.define_dimension("nodes", 5).unwrap();
    netcdf
        .define_variable::<f64>("coordx", 1, &["nodes"])
        .unwrap();
    netcdf
        .put_variable_attribute_text("coordx", "units", "meters")
        .unwrap();
    assert_eq!(netcdf.dimension_length("nodes").unwrap(), 5);
    assert_eq!(netcdf.try_dimension_length("missing").unwrap(), None);
    assert_eq!(
        netcdf
            .get_variable_attribute_text("coordx", "units")
            .unwrap(),
        "meters"
    );
}

#[test]
#[should_panic(expected = "no text attribute")]
fn missing_variable_attribute_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_missing_attr.nc").unwrap();
    netcdf.define_dimension("n", 1).unwrap();
    netcdf.define_variable::<i32>("v", 1, &["n"]).unwrap();
    let _ = netcdf.get_variable_attribute_text("v", "nope");
}

#[test]
#[should_panic(expected = "unknown dimension")]
fn end_definition_with_unknown_dimension_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_unknown_dim.nc").unwrap();
    netcdf
        .define_variable::<f64>("v", 1, &["never_defined"])
        .unwrap();
    netcdf.end_definition();
}

#[test]
#[should_panic(expected = "end_definition on a NetCDF opened for reading")]
fn end_definition_on_reader_panics() {
    let path = "target/netcdf_reader_enddef.nc";
    {
        let mut netcdf = NetCDF::create(path).unwrap();
        netcdf.define_dimension("n", 1).unwrap();
        netcdf.end_definition();
    }
    let mut netcdf = NetCDF::open(path).unwrap();
    netcdf.end_definition();
}

#[test]
#[should_panic(expected = "not allowed after end_definition")]
fn define_after_end_definition_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_define_after.nc").unwrap();
    netcdf.define_dimension("n", 1).unwrap();
    netcdf.end_definition();
    let _ = netcdf.define_dimension("m", 2);
}

#[test]
#[should_panic(expected = "write operation on a NetCDF opened for reading")]
fn write_on_reader_panics() {
    let path = "target/netcdf_reader_write.nc";
    {
        let mut netcdf = NetCDF::create(path).unwrap();
        netcdf.define_dimension("n", 1).unwrap();
        netcdf.end_definition();
    }
    let mut netcdf = NetCDF::open(path).unwrap();
    let _ = netcdf.define_dimension("m", 2);
}

fn one_var_file(path: &str) -> NetCDF {
    {
        let mut netcdf = NetCDF::create(path).unwrap();
        netcdf.define_dimension("n", 2).unwrap();
        netcdf.define_variable::<f64>("x", 1, &["n"]).unwrap();
        netcdf.end_definition();
        netcdf.put_variable("x", &[1.0_f64, 2.0]).unwrap();
    }
    NetCDF::open(path).unwrap()
}

#[test]
#[should_panic(expected = "no dimension named")]
fn dimension_length_missing_panics() {
    let netcdf = one_var_file("target/netcdf_missing_dim.nc");
    let _ = netcdf.dimension_length("nope");
}

#[test]
#[should_panic(expected = "no variable named")]
fn get_variable_missing_panics() {
    let netcdf = one_var_file("target/netcdf_get_missing.nc");
    let _ = netcdf.get_variable::<f64>("nope", 1);
}

#[test]
#[should_panic(expected = "no variable named")]
fn get_variable_slice_missing_panics() {
    let netcdf = one_var_file("target/netcdf_slice_missing.nc");
    let _ = netcdf.get_variable_slice::<f64>("nope", &[0], &[1]);
}

#[test]
#[should_panic(expected = "get_variable on a NetCDF opened for writing")]
fn get_variable_on_writer_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_get_on_writer.nc").unwrap();
    netcdf.define_dimension("n", 1).unwrap();
    netcdf.define_variable::<i32>("v", 1, &["n"]).unwrap();
    netcdf.end_definition();
    let _ = netcdf.get_variable::<i32>("v", 1);
}

#[test]
#[should_panic(expected = "get_variable_slice on a NetCDF opened for writing")]
fn get_variable_slice_on_writer_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_slice_on_writer.nc").unwrap();
    netcdf.define_dimension("n", 1).unwrap();
    netcdf.define_variable::<i32>("v", 1, &["n"]).unwrap();
    netcdf.end_definition();
    let _ = netcdf.get_variable_slice::<i32>("v", &[0], &[1]);
}

#[test]
#[should_panic(expected = "put_variable on a NetCDF opened for reading")]
fn put_variable_on_reader_panics() {
    let mut netcdf = one_var_file("target/netcdf_put_on_reader.nc");
    let _ = netcdf.put_variable("x", &[0.0_f64, 0.0]);
}

#[test]
#[should_panic(expected = "put_variable before end_definition")]
fn put_variable_before_end_definition_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_put_early.nc").unwrap();
    netcdf.define_dimension("n", 1).unwrap();
    netcdf.define_variable::<i32>("v", 1, &["n"]).unwrap();
    let _ = netcdf.put_variable("v", &[1_i32]);
}

#[test]
#[should_panic(expected = "no variable named")]
fn put_variable_missing_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_put_missing.nc").unwrap();
    netcdf.define_dimension("n", 1).unwrap();
    netcdf.end_definition();
    let _ = netcdf.put_variable("nope", &[1_i32]);
}

#[test]
#[should_panic(expected = "type mismatch writing variable")]
fn put_variable_type_mismatch_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_put_type.nc").unwrap();
    netcdf.define_dimension("n", 1).unwrap();
    netcdf.define_variable::<f64>("v", 1, &["n"]).unwrap();
    netcdf.end_definition();
    let _ = netcdf.put_variable("v", &[1_i32]);
}

#[test]
#[should_panic(expected = "wrong element count for variable")]
fn put_variable_wrong_count_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_put_count.nc").unwrap();
    netcdf.define_dimension("n", 3).unwrap();
    netcdf.define_variable::<i32>("v", 1, &["n"]).unwrap();
    netcdf.end_definition();
    let _ = netcdf.put_variable("v", &[1_i32, 2]);
}

#[test]
#[should_panic(expected = "type mismatch reading variable")]
fn get_variable_type_mismatch_panics() {
    let netcdf = one_var_file("target/netcdf_get_type.nc");
    let _ = netcdf.get_variable::<i32>("x", 2);
}

#[test]
#[should_panic(expected = "runs past end of file")]
fn get_variable_out_of_range_panics() {
    let netcdf = one_var_file("target/netcdf_get_range.nc");
    let _ = netcdf.get_variable::<f64>("x", 9999);
}

#[test]
#[should_panic(expected = "no variable named")]
fn get_variable_attribute_missing_variable_panics() {
    let netcdf = one_var_file("target/netcdf_attr_missing_var.nc");
    let _ = netcdf.get_variable_attribute_text("absent", "any");
}

#[test]
#[should_panic(expected = "no variable named")]
fn put_variable_attribute_missing_variable_panics() {
    let mut netcdf = NetCDF::create("target/netcdf_put_attr_missing.nc").unwrap();
    netcdf.define_dimension("n", 1).unwrap();
    let _ = netcdf.put_variable_attribute_text("absent", "a", "x");
}
