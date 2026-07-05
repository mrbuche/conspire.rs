use super::{Npy, NpyType};
use crate::io::Write;
use std::{
    fs::write,
    io::{self, Error, Result},
};

fn npy_bytes(version: u8, descr: &str, count: usize, data: &[u8]) -> Vec<u8> {
    let header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({count}, ), }}");
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.push(version);
    bytes.push(0);
    if version == 2 {
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    } else {
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    }
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

#[test]
fn not_npy_errors() {
    let path = "target/npy_not.npy";
    write(path, b"this is not a numpy file").unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn reads_version_2() {
    let path = "target/npy_v2.npy";
    write(path, npy_bytes(2, "|u1", 3, &[7, 8, 9])).unwrap();
    let npy = Npy::<u8>::read(path).unwrap();
    assert_eq!(npy.data, [7, 8, 9]);
    assert_eq!(npy.shape, [3]);
}

#[test]
fn unsupported_version_errors() {
    let path = "target/npy_v3.npy";
    write(path, npy_bytes(3, "|u1", 3, &[7, 8, 9])).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn truncated_errors() {
    let path = "target/npy_truncated.npy";
    write(path, npy_bytes(1, "|u1", 10, &[1, 2])).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

fn npy_raw(header: &[u8], data: &[u8]) -> Vec<u8> {
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(data);
    bytes
}

#[test]
fn non_utf8_header_errors() {
    let path = "target/npy_badutf8.npy";
    write(path, npy_raw(&[0xff, 0xfe, 0x00], &[])).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn missing_descr_errors() {
    let path = "target/npy_nodescr.npy";
    write(
        path,
        npy_raw(b"{'fortran_order': False, 'shape': (3, ), }", &[1, 2, 3]),
    )
    .unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn big_endian_descr_errors() {
    let path = "target/npy_be.npy";
    let header = b"{'descr': '>u2', 'fortran_order': False, 'shape': (1, ), }";
    write(path, npy_raw(header, &[0, 1])).unwrap();
    assert!(Npy::<u16>::read(path).is_err());
}

#[test]
fn missing_shape_errors() {
    let path = "target/npy_noshape.npy";
    write(
        path,
        npy_raw(b"{'descr': '|u1', 'fortran_order': False, }", &[1]),
    )
    .unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn shape_without_paren_errors() {
    let path = "target/npy_noparen.npy";
    write(path, npy_raw(b"{'descr': '|u1', 'shape': 3, }", &[1, 2, 3])).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn unclosed_shape_errors() {
    let path = "target/npy_unclosed.npy";
    write(
        path,
        npy_raw(b"{'descr': '|u1', 'shape': (3, }", &[1, 2, 3]),
    )
    .unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn bad_shape_entry_errors() {
    let path = "target/npy_badentry.npy";
    write(path, npy_raw(b"{'descr': '|u1', 'shape': (x, ), }", &[1])).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn descr_without_value_quote_errors() {
    let path = "target/npy_descr_noquote.npy";
    write(path, npy_raw(b"{'descr': }", &[])).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn descr_unterminated_value_errors() {
    let path = "target/npy_descr_unterminated.npy";
    write(path, npy_raw(b"{'descr': '|u1 }", &[])).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn write_to_bad_path_errors() {
    let npy = Npy {
        data: vec![1u8, 2, 3],
        shape: vec![3],
        fortran_order: true,
    };
    assert!(npy.write("target/nonexistent_dir/x.npy").is_err());
}

#[test]
fn read_missing_file_errors() {
    assert!(Npy::<u8>::read("target/does_not_exist.npy").is_err());
}

struct FailOnFlush;

impl io::Write for FailOnFlush {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> Result<()> {
        Err(Error::other("boom"))
    }
}

#[test]
fn write_to_propagates_flush_error() {
    let npy = Npy {
        data: vec![1u8, 2, 3],
        shape: vec![3],
        fortran_order: true,
    };
    assert!(npy.write_to(&mut FailOnFlush).is_err());
}

struct FailOnNthWrite {
    allowed: usize,
    count: usize,
}

impl io::Write for FailOnNthWrite {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        if self.count >= self.allowed {
            return Err(Error::other("boom"));
        }
        self.count += 1;
        Ok(buf.len())
    }
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn write_to_propagates_write_errors_at_every_step() {
    let npy = Npy {
        data: vec![1u8, 2, 3],
        shape: vec![3],
        fortran_order: true,
    };
    for allowed in 0..5 {
        assert!(
            npy.write_to(&mut FailOnNthWrite { allowed, count: 0 })
                .is_err()
        );
    }
    assert!(
        npy.write_to(&mut FailOnNthWrite {
            allowed: 5,
            count: 0
        })
        .is_ok()
    );
}

#[test]
fn read_truncated_prefix_errors() {
    let path = "target/npy_short_prefix.npy";
    write(path, &b"\x93NUMPY\x01\x00"[..]).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn read_truncated_version_2_header_length_errors() {
    let path = "target/npy_short_v2_prefix.npy";
    write(path, &b"\x93NUMPY\x02\x00\x00\x00"[..]).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

#[test]
fn read_truncated_header_errors() {
    let path = "target/npy_short_header.npy";
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&100u16.to_le_bytes());
    bytes.extend_from_slice(b"{'descr'");
    write(path, bytes).unwrap();
    assert!(Npy::<u8>::read(path).is_err());
}

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
    let npy = Npy::<u16>::read(path).unwrap();
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
    let npy = Npy::<f64>::read(path).unwrap();
    assert_eq!(npy.data, data);
    assert_eq!(npy.shape, [3]);
    assert!(!npy.fortran_order);
}

#[test]
fn write_le_bytes_and_read_le_all_round_trip() {
    let data = vec![1.5_f64, -2.0, 3.25];
    let bytes = f64::write_le_bytes(&data);
    assert_eq!(f64::read_le_all(&bytes), data);

    let data = vec![1u8, 2, 3];
    let bytes = u8::write_le_bytes(&data);
    assert_eq!(u8::read_le_all(&bytes), data);

    let data = vec![-1i8, 2, -3];
    let bytes = i8::write_le_bytes(&data);
    assert_eq!(i8::read_le_all(&bytes), data);

    let data = vec![1u16, 2, 3];
    let bytes = u16::write_le_bytes(&data);
    assert_eq!(u16::read_le_all(&bytes), data);

    let data = vec![-1i16, 2, -3];
    let bytes = i16::write_le_bytes(&data);
    assert_eq!(i16::read_le_all(&bytes), data);

    let data = vec![1u32, 2, 3];
    let bytes = u32::write_le_bytes(&data);
    assert_eq!(u32::read_le_all(&bytes), data);

    let data = vec![-1i32, 2, -3];
    let bytes = i32::write_le_bytes(&data);
    assert_eq!(i32::read_le_all(&bytes), data);

    let data = vec![1u64, 2, 3];
    let bytes = u64::write_le_bytes(&data);
    assert_eq!(u64::read_le_all(&bytes), data);

    let data = vec![-1i64, 2, -3];
    let bytes = i64::write_le_bytes(&data);
    assert_eq!(i64::read_le_all(&bytes), data);

    let data = vec![1.5_f32, -2.0, 3.25];
    let bytes = f32::write_le_bytes(&data);
    assert_eq!(f32::read_le_all(&bytes), data);
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
    assert!(Npy::<u16>::read(path).is_err());
}

fn round_trip<T: NpyType + Copy + PartialEq + std::fmt::Debug>(tag: &str, data: Vec<T>) {
    let path = format!("target/npy_generic_{tag}.npy");
    Npy {
        data: data.clone(),
        shape: vec![data.len()],
        fortran_order: false,
    }
    .write(path.as_str())
    .unwrap();
    let npy = Npy::<T>::read(path.as_str()).unwrap();
    assert_eq!(npy.data, data);
    assert_eq!(npy.shape, [data.len()]);
    assert!(!npy.fortran_order);
}

#[test]
fn round_trip_remaining_numeric_types() {
    round_trip::<i8>("i8", vec![-1, 2, -3]);
    round_trip::<i16>("i16", vec![-1, 2, -3]);
    round_trip::<i32>("i32", vec![-1, 2, -3]);
    round_trip::<u32>("u32", vec![1, 2, 3]);
    round_trip::<i64>("i64", vec![-1, 2, -3]);
    round_trip::<u64>("u64", vec![1, 2, 3]);
    round_trip::<f32>("f32", vec![1.5, -2.0, 3.25]);
}

fn read_error_paths<T: NpyType>(tag: &str, wrong_descr: &str) {
    let path = format!("target/npy_generic_{tag}_not_npy.npy");
    write(&path, b"this is not a numpy file").unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let path = format!("target/npy_generic_{tag}_short_prefix.npy");
    write(&path, &b"\x93NUMPY\x01\x00"[..]).unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let path = format!("target/npy_generic_{tag}_missing.npy");
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let data = vec![0u8; T::SIZE];
    let path = format!("target/npy_generic_{tag}_v3.npy");
    write(&path, npy_bytes(3, T::DESCR, 1, &data)).unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let path = format!("target/npy_generic_{tag}_short_header.npy");
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&100u16.to_le_bytes());
    bytes.extend_from_slice(b"{'descr'");
    write(&path, bytes).unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let path = format!("target/npy_generic_{tag}_badutf8.npy");
    write(&path, npy_raw(&[0xff, 0xfe, 0x00], &[])).unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let path = format!("target/npy_generic_{tag}_nodescr.npy");
    write(
        &path,
        npy_raw(b"{'fortran_order': False, 'shape': (1, ), }", &data),
    )
    .unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let path = format!("target/npy_generic_{tag}_dtype.npy");
    write(&path, npy_bytes(1, wrong_descr, 1, &data)).unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let path = format!("target/npy_generic_{tag}_badshape.npy");
    let header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': (x, ), }}",
        T::DESCR
    );
    write(&path, npy_raw(header.as_bytes(), &[])).unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());

    let path = format!("target/npy_generic_{tag}_truncdata.npy");
    write(&path, npy_bytes(1, T::DESCR, 10, &data)).unwrap();
    assert!(Npy::<T>::read(path.as_str()).is_err());
}

#[test]
fn read_error_paths_f64_and_u16() {
    read_error_paths::<f64>("f64", "|u1");
    read_error_paths::<u16>("u16", "|u1");
    read_error_paths::<u8>("u8_generic", "|i1");
}
