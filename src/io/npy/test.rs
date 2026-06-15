use super::{Npy, read};
use crate::io::Write;

#[test]
fn round_trip_fortran_order() {
    let data: Vec<u16> = (0..24).collect();
    let path = "target/npy_f.npy";
    Npy {
        data: data.clone(),
        shape: vec![2, 3, 4],
        fortran_order: true,
    }
    .write(path)
    .unwrap();
    let npy: Npy<u16> = read(path).unwrap();
    assert_eq!(npy.data, data);
    assert_eq!(npy.shape, [2, 3, 4]);
    assert!(npy.fortran_order);
}

#[test]
fn round_trip_c_order_f64() {
    let data = vec![1.5_f64, -2.0, 3.25];
    let path = "target/npy_c.npy";
    Npy {
        data: data.clone(),
        shape: vec![3],
        fortran_order: false,
    }
    .write(path)
    .unwrap();
    let npy: Npy<f64> = read(path).unwrap();
    assert_eq!(npy.data, data);
    assert_eq!(npy.shape, [3]);
    assert!(!npy.fortran_order);
}

#[test]
fn dtype_mismatch_errors() {
    let path = "target/npy_dtype.npy";
    Npy {
        data: vec![1u8, 2, 3],
        shape: vec![3],
        fortran_order: true,
    }
    .write(path)
    .unwrap();
    assert!(read::<u16, _>(path).is_err());
}
