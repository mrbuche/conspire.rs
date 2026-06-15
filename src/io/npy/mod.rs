#[cfg(test)]
mod test;

use crate::io::Write;
use std::{
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write as _},
    path::Path,
};

pub trait NpyType: Copy {
    const DESCR: &'static str;
    const SIZE: usize;
    fn write_le(self, buffer: &mut Vec<u8>);
    fn read_le(bytes: &[u8]) -> Self;
}

macro_rules! npy_type {
    ($type:ty, $descr:literal) => {
        impl NpyType for $type {
            const DESCR: &'static str = $descr;
            const SIZE: usize = std::mem::size_of::<$type>();
            fn write_le(self, buffer: &mut Vec<u8>) {
                buffer.extend_from_slice(&self.to_le_bytes());
            }
            fn read_le(bytes: &[u8]) -> Self {
                <$type>::from_le_bytes(bytes.try_into().unwrap())
            }
        }
    };
}
npy_type!(u8, "|u1");
npy_type!(i8, "|i1");
npy_type!(u16, "<u2");
npy_type!(i16, "<i2");
npy_type!(u32, "<u4");
npy_type!(i32, "<i4");
npy_type!(u64, "<u8");
npy_type!(i64, "<i8");
npy_type!(f32, "<f4");
npy_type!(f64, "<f8");

pub struct Npy<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub fortran_order: bool,
}

impl<T, P> Write<P> for Npy<T>
where
    T: NpyType,
    P: AsRef<Path>,
{
    type Error = Error;
    fn write(&self, path: P) -> Result<()> {
        let order = if self.fortran_order { "True" } else { "False" };
        let dims: String = self.shape.iter().map(|d| format!("{d}, ")).collect();
        let mut header = format!(
            "{{'descr': '{}', 'fortran_order': {order}, 'shape': ({dims}), }}",
            T::DESCR
        );
        let pad = (64 - (10 + header.len() + 1) % 64) % 64;
        header.push_str(&" ".repeat(pad));
        header.push('\n');
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(b"\x93NUMPY")?;
        file.write_all(&[1, 0])?;
        file.write_all(&(header.len() as u16).to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        let mut buffer = Vec::with_capacity(self.data.len() * T::SIZE);
        for &value in &self.data {
            value.write_le(&mut buffer);
        }
        file.write_all(&buffer)?;
        Ok(())
    }
}

impl<T: NpyType> Npy<T> {
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return Err(invalid("not a .npy file".into()));
        }
        let (header_length, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 => (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            ),
            other => return Err(invalid(format!("unsupported .npy version {other}"))),
        };
        let header = std::str::from_utf8(&bytes[start..start + header_length])
            .map_err(|_| invalid("non-UTF-8 .npy header".into()))?;
        let descr = quoted(header, "'descr':").ok_or_else(|| invalid("no descr".into()))?;
        if descr.starts_with('>') || descr.get(1..) != T::DESCR.get(1..) {
            return Err(invalid(format!(
                "dtype {descr} does not match {}",
                T::DESCR
            )));
        }
        let fortran_order = header.contains("'fortran_order': True");
        let shape = shape(header)?;
        let count: usize = shape.iter().product();
        let data = &bytes[start + header_length..];
        if data.len() < count * T::SIZE {
            return Err(invalid("truncated .npy data".into()));
        }
        Ok(Npy {
            data: data.chunks(T::SIZE).take(count).map(T::read_le).collect(),
            shape,
            fortran_order,
        })
    }
}

fn quoted<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let at = header.find(key)? + key.len();
    let open = header[at..].find('\'')? + at + 1;
    let close = header[open..].find('\'')? + open;
    Some(&header[open..close])
}

fn shape(header: &str) -> Result<Vec<usize>> {
    let at = header
        .find("'shape':")
        .ok_or_else(|| invalid("no shape".into()))?;
    let open = header[at..]
        .find('(')
        .ok_or_else(|| invalid("malformed shape".into()))?
        + at
        + 1;
    let close = header[open..]
        .find(')')
        .ok_or_else(|| invalid("malformed shape".into()))?
        + open;
    header[open..close]
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| {
            dim.parse()
                .map_err(|_| invalid(format!("bad shape entry {dim}")))
        })
        .collect()
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}
