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
#[should_panic]
#[cfg(unix)]
fn non_utf8_path_panics() {
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};
    let _ = NetCDF::try_from(Path::new(OsStr::from_bytes(&[0xff, 0x2f, 0x66])));
}

#[test]
fn try_from_nul_path_errors() {
    assert!(NetCDF::try_from(Path::new("bad\0.nc")).is_err());
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
