#[cfg(test)]
mod test;

use crate::io::Write;
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Error, ErrorKind, Read, Result},
    mem::size_of,
    path::Path,
    str::from_utf8,
};

pub trait NpyType: Copy {
    const DESCR: &'static str;
    const SIZE: usize;
    fn write_le(self, buffer: &mut Vec<u8>);
    fn read_le(bytes: &[u8]) -> Self;
    fn read_from<R: Read>(file: &mut R, count: usize) -> Result<Vec<Self>> {
        if cfg!(target_endian = "little") {
            let mut data = Vec::<Self>::with_capacity(count);
            unsafe {
                let bytes = std::slice::from_raw_parts_mut(
                    data.as_mut_ptr() as *mut u8,
                    count * Self::SIZE,
                );
                file.read_exact(bytes)
                    .map_err(|_| invalid("truncated .npy data".into()))?;
                data.set_len(count);
            }
            Ok(data)
        } else {
            let mut bytes = vec![0u8; count * Self::SIZE];
            file.read_exact(&mut bytes)
                .map_err(|_| invalid("truncated .npy data".into()))?;
            Ok(bytes.chunks_exact(Self::SIZE).map(Self::read_le).collect())
        }
    }
    fn write_le_all<W: io::Write>(data: &[Self], file: &mut W) -> Result<()> {
        if cfg!(target_endian = "little") {
            let bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            };
            file.write_all(bytes)
        } else {
            let mut buffer = Vec::with_capacity(std::mem::size_of_val(data));
            for &value in data {
                value.write_le(&mut buffer);
            }
            file.write_all(&buffer)
        }
    }
}

macro_rules! npy_type {
    ($type:ty, $descr:literal) => {
        impl NpyType for $type {
            const DESCR: &'static str = $descr;
            const SIZE: usize = size_of::<$type>();
            fn write_le(self, buffer: &mut Vec<u8>) {
                buffer.extend_from_slice(&self.to_le_bytes());
            }
            fn read_le(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
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
        self.write_to(&mut BufWriter::new(File::create(path)?))
    }
}

impl<T: NpyType> Npy<T> {
    fn write_to<W: io::Write>(&self, file: &mut W) -> Result<()> {
        let order = if self.fortran_order { "True" } else { "False" };
        let dims: String = self.shape.iter().map(|d| format!("{d}, ")).collect();
        let mut header = format!(
            "{{'descr': '{}', 'fortran_order': {order}, 'shape': ({dims}), }}",
            T::DESCR
        );
        let pad = (64 - (10 + header.len() + 1) % 64) % 64;
        header.push_str(&" ".repeat(pad));
        header.push('\n');
        file.write_all(b"\x93NUMPY")?;
        file.write_all(&[1, 0])?;
        file.write_all(&(header.len() as u16).to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        T::write_le_all(&self.data, file)?;
        file.flush()
    }
}

impl<T: NpyType> Npy<T> {
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let mut prefix = [0u8; 10];
        file.read_exact(&mut prefix)
            .map_err(|_| invalid("not a .npy file".into()))?;
        if &prefix[..6] != b"\x93NUMPY" {
            return Err(invalid("not a .npy file".into()));
        }
        let header_length = match prefix[6] {
            1 => u16::from_le_bytes([prefix[8], prefix[9]]) as usize,
            2 => {
                let mut rest = [0u8; 2];
                file.read_exact(&mut rest)?;
                u32::from_le_bytes([prefix[8], prefix[9], rest[0], rest[1]]) as usize
            }
            other => return Err(invalid(format!("unsupported .npy version {other}"))),
        };
        let mut header_bytes = vec![0u8; header_length];
        file.read_exact(&mut header_bytes)?;
        let header =
            from_utf8(&header_bytes).map_err(|_| invalid("non-UTF-8 .npy header".into()))?;
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
        let data = T::read_from(&mut file, count)?;
        Ok(Npy {
            data,
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
