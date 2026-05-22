use std::ffi::{CStr, c_char};

// maybe an exodus module in geometry/mesh/ and another in fem/ and in vem/ etc.
// the one in mesh would be more like genesis files
// the ones in fem/ and vem/ would be more like exodus files (with output variables and timesteps)

unsafe extern "C" {
    pub fn ex_create_int(path: *const c_char, mode: i32, comp_ws: *mut i32, io_ws: *mut i32)
    -> i32;
    pub fn ex_close(exoid: i32) -> i32;
}

#[test]
fn test_exodus_create_close() {
    use std::ffi::CString;

    let path = CString::new("target/test.exo").unwrap();
    let mut comp_ws = 8;
    let mut io_ws = 8;

    let exoid = unsafe { ex_create_int(path.as_ptr(), 0, &mut comp_ws, &mut io_ws) };
    assert!(exoid >= 0, "ex_create failed with exoid={exoid}");

    let status = unsafe { ex_close(exoid) };
    assert_eq!(status, 0, "ex_close failed with status={status}");
}

unsafe extern "C" {
    pub fn nc_inq_libvers() -> *const c_char;
}

#[test]
fn test_netcdf_lib_version() {
    let ptr = unsafe { nc_inq_libvers() };
    assert!(!ptr.is_null(), "nc_inq_libvers returned null");

    let version = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("netCDF version string was not valid UTF-8");

    println!("netCDF version: {version}");
    assert!(!version.is_empty(), "netCDF version string was empty");
}
