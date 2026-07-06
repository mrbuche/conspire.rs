use super::{Zip, ZipEntry};
use crate::io::{Npy, Write};
use std::{
    fs::write,
    io::{self, Error, Result},
};

#[test]
fn round_trip_heterogeneous_npz() {
    let path = "target/zip_heterogeneous.npz";
    let f64_data = vec![1.5_f64, -2.0, 3.25];
    let u8_data = vec![7u8, 8, 9, 10];
    let mut f64_bytes = Vec::new();
    Npy {
        data: f64_data.clone(),
        shape: vec![3],
        fortran_order: false,
    }
    .write_to(&mut f64_bytes)
    .unwrap();
    let mut u8_bytes = Vec::new();
    Npy {
        data: u8_data.clone(),
        shape: vec![2, 2],
        fortran_order: true,
    }
    .write_to(&mut u8_bytes)
    .unwrap();
    Zip {
        entries: vec![
            ZipEntry {
                name: "positions.npy".to_string(),
                data: f64_bytes,
            },
            ZipEntry {
                name: "labels.npy".to_string(),
                data: u8_bytes,
            },
        ],
    }
    .write(path)
    .unwrap();

    let zip = Zip::read(path).unwrap();
    let positions = Npy::<f64>::read_from(&mut zip.entry("positions.npy").unwrap()).unwrap();
    assert_eq!(positions.data, f64_data);
    assert_eq!(positions.shape, [3]);
    assert!(!positions.fortran_order);
    let labels = Npy::<u8>::read_from(&mut zip.entry("labels.npy").unwrap()).unwrap();
    assert_eq!(labels.data, u8_data);
    assert_eq!(labels.shape, [2, 2]);
    assert!(labels.fortran_order);
}

#[test]
fn round_trip_single_entry() {
    let path = "target/zip_single.npz";
    Zip {
        entries: vec![ZipEntry {
            name: "arr_0.npy".to_string(),
            data: b"hello zip".to_vec(),
        }],
    }
    .write(path)
    .unwrap();
    let zip = Zip::read(path).unwrap();
    assert_eq!(zip.entry("arr_0.npy"), Some(&b"hello zip"[..]));
    assert_eq!(zip.entry("missing"), None);
}

#[test]
fn round_trip_multiple_entries() {
    let path = "target/zip_multi.npz";
    Zip {
        entries: vec![
            ZipEntry {
                name: "a.npy".to_string(),
                data: b"aaaa".to_vec(),
            },
            ZipEntry {
                name: "b.npy".to_string(),
                data: b"bb".to_vec(),
            },
            ZipEntry {
                name: "empty.npy".to_string(),
                data: vec![],
            },
        ],
    }
    .write(path)
    .unwrap();
    let zip = Zip::read(path).unwrap();
    assert_eq!(zip.entry("a.npy"), Some(&b"aaaa"[..]));
    assert_eq!(zip.entry("b.npy"), Some(&b"bb"[..]));
    assert_eq!(zip.entry("empty.npy"), Some(&b""[..]));
}

#[test]
fn round_trip_empty_archive() {
    let path = "target/zip_empty.npz";
    Zip { entries: vec![] }.write(path).unwrap();
    let zip = Zip::read(path).unwrap();
    assert!(zip.entries.is_empty());
}

#[test]
fn not_zip_errors() {
    let path = "target/zip_not.npz";
    write(path, b"this is not a zip file").unwrap();
    assert!(Zip::read(path).is_err());
}

#[test]
fn truncated_eocd_errors() {
    let path = "target/zip_truncated_eocd.npz";
    write(path, b"PK\x05\x06").unwrap();
    assert!(Zip::read(path).is_err());
}

#[test]
fn read_missing_file_errors() {
    assert!(Zip::read("target/zip_does_not_exist.npz").is_err());
}

#[test]
fn read_directory_errors() {
    assert!(Zip::read("target").is_err());
}

#[test]
fn non_utf8_entry_name_errors() {
    let path = "target/zip_badutf8_name.npz";
    Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: b"aaaa".to_vec(),
        }],
    }
    .write(path)
    .unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    let central_start = bytes.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    bytes[central_start + 46] = 0xff;
    write(path, &bytes).unwrap();
    assert!(Zip::read(path).is_err());
}

#[test]
fn write_to_bad_path_errors() {
    let zip = Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: vec![1, 2, 3],
        }],
    };
    assert!(zip.write("target/nonexistent_dir/x.npz").is_err());
}

#[test]
fn central_directory_out_of_bounds_errors() {
    let path = "target/zip_central_oob.npz";
    Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: b"aaaa".to_vec(),
        }],
    }
    .write(path)
    .unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    let eocd = bytes.windows(4).rposition(|w| w == b"PK\x05\x06").unwrap();
    let huge = (bytes.len() as u32) * 10;
    bytes[eocd + 16..eocd + 20].copy_from_slice(&huge.to_le_bytes());
    write(path, &bytes).unwrap();
    assert!(Zip::read(path).is_err());
}

#[test]
fn malformed_central_directory_record_errors() {
    let path = "target/zip_central_malformed.npz";
    Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: b"aaaa".to_vec(),
        }],
    }
    .write(path)
    .unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    let central_start = bytes.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    bytes[central_start] = 0;
    write(path, &bytes).unwrap();
    assert!(Zip::read(path).is_err());
}

#[test]
fn truncated_file_name_errors() {
    let path = "target/zip_truncated_name.npz";
    Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: b"aaaa".to_vec(),
        }],
    }
    .write(path)
    .unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    let central_start = bytes.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    bytes[central_start + 28] = 0xff;
    bytes[central_start + 29] = 0xff;
    write(path, &bytes).unwrap();
    assert!(Zip::read(path).is_err());
}

#[test]
fn malformed_local_file_header_errors() {
    let path = "target/zip_local_malformed.npz";
    Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: b"aaaa".to_vec(),
        }],
    }
    .write(path)
    .unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    bytes[0] = 0;
    write(path, &bytes).unwrap();
    assert!(Zip::read(path).is_err());
}

#[test]
fn truncated_entry_data_errors() {
    let path = "target/zip_truncated_data.npz";
    Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: b"aaaa".to_vec(),
        }],
    }
    .write(path)
    .unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    let central_start = bytes.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    let bigger_size = 1_000u32;
    bytes[central_start + 24..central_start + 28].copy_from_slice(&bigger_size.to_le_bytes());
    write(path, &bytes).unwrap();
    assert!(Zip::read(path).is_err());
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
    let zip = Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: vec![1, 2, 3],
        }],
    };
    assert!(zip.write_to(&mut FailOnFlush).is_err());
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
    let zip = Zip {
        entries: vec![
            ZipEntry {
                name: "a.npy".to_string(),
                data: vec![1, 2, 3],
            },
            ZipEntry {
                name: "b.npy".to_string(),
                data: vec![4, 5],
            },
        ],
    };
    for allowed in 0..35 {
        assert!(
            zip.write_to(&mut FailOnNthWrite { allowed, count: 0 })
                .is_err()
        );
    }
    assert!(
        zip.write_to(&mut FailOnNthWrite {
            allowed: 35,
            count: 0
        })
        .is_ok()
    );
}

#[test]
fn unsupported_compression_method_errors() {
    let path = "target/zip_deflate.npz";
    Zip {
        entries: vec![ZipEntry {
            name: "a.npy".to_string(),
            data: b"aaaa".to_vec(),
        }],
    }
    .write(path)
    .unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    bytes[8] = 8;
    let central_start = bytes.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    bytes[central_start + 10] = 8;
    write(path, &bytes).unwrap();
    assert!(Zip::read(path).is_err());
}
