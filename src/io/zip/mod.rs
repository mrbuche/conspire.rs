#[cfg(test)]
mod test;

use crate::io::Write;
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Error, ErrorKind, Read, Result},
    path::Path,
    str::from_utf8,
};

pub struct ZipEntry {
    pub name: String,
    pub data: Vec<u8>,
}

pub struct Zip {
    pub entries: Vec<ZipEntry>,
}

impl<P: AsRef<Path>> Write<P> for Zip {
    type Error = Error;
    fn write(&self, path: P) -> Result<()> {
        self.write_to(&mut BufWriter::new(File::create(path)?))
    }
}

impl Zip {
    fn write_to<W: io::Write>(&self, file: &mut W) -> Result<()> {
        let mut offset: u32 = 0;
        let mut central = Vec::new();
        for entry in &self.entries {
            let crc = crc32(&entry.data);
            let name = entry.name.as_bytes();
            let size = entry.data.len() as u32;
            file.write_all(b"PK\x03\x04")?;
            file.write_all(&20u16.to_le_bytes())?;
            file.write_all(&0u16.to_le_bytes())?;
            file.write_all(&0u16.to_le_bytes())?;
            file.write_all(&0u16.to_le_bytes())?;
            file.write_all(&0x21u16.to_le_bytes())?;
            file.write_all(&crc.to_le_bytes())?;
            file.write_all(&size.to_le_bytes())?;
            file.write_all(&size.to_le_bytes())?;
            file.write_all(&(name.len() as u16).to_le_bytes())?;
            file.write_all(&0u16.to_le_bytes())?;
            file.write_all(name)?;
            file.write_all(&entry.data)?;
            central.extend_from_slice(b"PK\x01\x02");
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0x21u16.to_le_bytes());
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&size.to_le_bytes());
            central.extend_from_slice(&size.to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name);
            offset += 30 + name.len() as u32 + size;
        }
        let central_size = central.len() as u32;
        let central_offset = offset;
        file.write_all(&central)?;
        file.write_all(b"PK\x05\x06")?;
        file.write_all(&0u16.to_le_bytes())?;
        file.write_all(&0u16.to_le_bytes())?;
        file.write_all(&(self.entries.len() as u16).to_le_bytes())?;
        file.write_all(&(self.entries.len() as u16).to_le_bytes())?;
        file.write_all(&central_size.to_le_bytes())?;
        file.write_all(&central_offset.to_le_bytes())?;
        file.write_all(&0u16.to_le_bytes())?;
        file.flush()
    }
}

impl Zip {
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Self::read_from(&bytes)
    }

    fn read_from(bytes: &[u8]) -> Result<Self> {
        let eocd_sig = [0x50, 0x4b, 0x05, 0x06];
        let pos = bytes
            .windows(4)
            .rposition(|w| w == eocd_sig)
            .ok_or_else(|| invalid("not a zip file (no end of central directory)".into()))?;
        if pos + 22 > bytes.len() {
            return Err(invalid("truncated end of central directory".into()));
        }
        let eocd = &bytes[pos..pos + 22];
        let total_entries = u16::from_le_bytes([eocd[10], eocd[11]]) as usize;
        let central_size = u32::from_le_bytes([eocd[12], eocd[13], eocd[14], eocd[15]]) as usize;
        let central_offset = u32::from_le_bytes([eocd[16], eocd[17], eocd[18], eocd[19]]) as usize;
        if central_offset > bytes.len() || central_offset + central_size > bytes.len() {
            return Err(invalid("central directory out of bounds".into()));
        }
        let mut entries = Vec::with_capacity(total_entries);
        let mut cursor = central_offset;
        for _ in 0..total_entries {
            if cursor + 46 > bytes.len() || &bytes[cursor..cursor + 4] != b"PK\x01\x02" {
                return Err(invalid("malformed central directory record".into()));
            }
            let record = &bytes[cursor..cursor + 46];
            let method = u16::from_le_bytes([record[10], record[11]]);
            let size =
                u32::from_le_bytes([record[24], record[25], record[26], record[27]]) as usize;
            let name_len = u16::from_le_bytes([record[28], record[29]]) as usize;
            let extra_len = u16::from_le_bytes([record[30], record[31]]) as usize;
            let comment_len = u16::from_le_bytes([record[32], record[33]]) as usize;
            let local_offset =
                u32::from_le_bytes([record[42], record[43], record[44], record[45]]) as usize;
            let name_start = cursor + 46;
            let name_end = name_start + name_len;
            if name_end > bytes.len() {
                return Err(invalid("truncated file name".into()));
            }
            let name = from_utf8(&bytes[name_start..name_end])
                .map_err(|_| invalid("non-UTF-8 entry name".into()))?
                .to_string();
            if method != 0 {
                return Err(invalid(format!(
                    "unsupported compression method {method} for {name} (only stored entries are supported)"
                )));
            }
            let data = read_local_entry(bytes, local_offset, size, &name)?;
            entries.push(ZipEntry { name, data });
            cursor = name_end + extra_len + comment_len;
        }
        Ok(Zip { entries })
    }

    pub fn entry(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.data.as_slice())
    }
}

fn read_local_entry(bytes: &[u8], offset: usize, size: usize, name: &str) -> Result<Vec<u8>> {
    if offset + 30 > bytes.len() || &bytes[offset..offset + 4] != b"PK\x03\x04" {
        return Err(invalid(format!("malformed local file header for {name}")));
    }
    let name_len = u16::from_le_bytes([bytes[offset + 26], bytes[offset + 27]]) as usize;
    let extra_len = u16::from_le_bytes([bytes[offset + 28], bytes[offset + 29]]) as usize;
    let data_start = offset + 30 + name_len + extra_len;
    let data_end = data_start + size;
    if data_end > bytes.len() {
        return Err(invalid(format!("truncated entry data for {name}")));
    }
    Ok(bytes[data_start..data_end].to_vec())
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}
