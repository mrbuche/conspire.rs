use std::{env::var_os, path::Path};

const PREFIXES: [&str; 3] = ["/opt/seacas", "/usr", "/usr/local"];

fn main() {
    if var_os("CARGO_FEATURE_EXODUS").is_some() {
        for prefix in PREFIXES {
            let lib_dir = format!("{}/lib", prefix);
            // let exodus = format!("{}/libexodus.so", lib_dir);
            let netcdf = format!("{}/libnetcdf.so", lib_dir);
            // if Path::new(&exodus).exists() && Path::new(&netcdf).exists() {
            if Path::new(&netcdf).exists() {
                println!("cargo:rustc-link-search=native={}", lib_dir);
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
                println!("cargo:rustc-link-lib=exodus");
                println!("cargo:rustc-link-lib=netcdf");
                return;
            }
        }
        // panic!("Could not find Exodus/netCDF libraries in known locations");
        panic!("Could not find netCDF library in known locations");
    }
}
