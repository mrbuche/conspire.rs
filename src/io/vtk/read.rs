use super::{invalid, unsupported};
use crate::io::deflate::zlib_decode;
use std::io::Result;

#[derive(Clone, Copy)]
pub struct DataArray<'a> {
    pub name: Option<&'a str>,
    pub data_type: &'a str,
    pub format: &'a str,
    pub tuples: usize,
    pub text: &'a str,
}

#[derive(Clone, Copy)]
pub struct Encoding {
    pub header_bytes: usize,
    pub compressed: bool,
}

pub fn encoding(header: &str) -> Result<Encoding> {
    let compressor = attribute(header, "compressor");
    if matches!(compressor, Some(other) if other != "vtkZLibDataCompressor") {
        return Err(unsupported(
            "only the vtkZLibDataCompressor compressor is supported",
        ));
    }
    Ok(Encoding {
        header_bytes: match attribute(header, "header_type") {
            Some("UInt32") | None => 4,
            Some("UInt64") => 8,
            Some(other) => return Err(invalid(format!("unsupported header_type {other}"))),
        },
        compressed: compressor.is_some(),
    })
}

pub fn tag<'a>(text: &'a str, open: &str) -> Result<&'a str> {
    let start = text
        .find(open)
        .ok_or_else(|| invalid(format!("missing {open}")))?;
    let end = text[start..]
        .find('>')
        .ok_or_else(|| invalid(format!("unterminated {open}")))?;
    Ok(&text[start..start + end])
}

pub fn region<'a>(text: &'a str, name: &str) -> Result<&'a str> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let start = text
        .find(&open)
        .ok_or_else(|| invalid(format!("missing <{name}>")))?;
    let end = text
        .find(&close)
        .ok_or_else(|| invalid(format!("missing </{name}>")))?;
    Ok(&text[start..end])
}

pub fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=\"");
    let start = tag.find(&key)? + key.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

pub fn data_arrays<'a>(region: &'a str) -> Result<Vec<DataArray<'a>>> {
    let mut rest = region;
    let mut arrays = Vec::new();
    while let Some(open) = rest.find("<DataArray") {
        let attributes_end = rest[open..]
            .find('>')
            .ok_or_else(|| invalid("unterminated DataArray".into()))?
            + open;
        let attributes = &rest[open..attributes_end];
        let close = rest[attributes_end..]
            .find("</DataArray>")
            .ok_or_else(|| invalid("unclosed DataArray".into()))?
            + attributes_end;
        arrays.push(DataArray {
            name: attribute(attributes, "Name"),
            data_type: attribute(attributes, "type")
                .ok_or_else(|| invalid("DataArray without type".into()))?,
            format: attribute(attributes, "format").unwrap_or("ascii"),
            tuples: attribute(attributes, "NumberOfTuples")
                .and_then(|t| t.parse().ok())
                .unwrap_or(0),
            text: rest[attributes_end + 1..close].trim(),
        });
        rest = &rest[close..];
    }
    Ok(arrays)
}

pub fn data_array<'a>(region: &'a str, name: Option<&str>) -> Result<DataArray<'a>> {
    find_data_array(&data_arrays(region)?, name)
}

pub fn find_data_array<'a>(arrays: &[DataArray<'a>], name: Option<&str>) -> Result<DataArray<'a>> {
    arrays
        .iter()
        .find(|array| name.is_none() || array.name == name)
        .copied()
        .ok_or_else(|| invalid(format!("missing DataArray {}", name.unwrap_or(""))))
}

pub fn floats(array: &DataArray, encoding: &Encoding) -> Result<Vec<f64>> {
    if array.format == "ascii" {
        return array.text.split_whitespace().map(parse).collect();
    }
    let bytes = decode(array, encoding)?;
    Ok(match array.data_type {
        "Float64" => bytes.chunks(8).map(le_f64).collect(),
        "Float32" => bytes.chunks(4).map(le_f32).collect(),
        other => return Err(invalid(format!("unsupported point type {other}"))),
    })
}

pub fn integers(array: &DataArray, encoding: &Encoding) -> Result<Vec<i64>> {
    if array.format == "ascii" {
        return array.text.split_whitespace().map(parse).collect();
    }
    let bytes = decode(array, encoding)?;
    Ok(match array.data_type {
        "Int64" | "UInt64" => bytes.chunks(8).map(le_i64).collect(),
        "Int32" | "UInt32" => bytes.chunks(4).map(le_i32).collect(),
        "Int8" | "UInt8" => bytes.iter().map(|&b| b as i64).collect(),
        other => return Err(invalid(format!("unsupported cell-data type {other}"))),
    })
}

pub fn bits(array: &DataArray, encoding: &Encoding) -> Result<Vec<u8>> {
    if array.format == "ascii" {
        return array.text.split_whitespace().map(parse).collect();
    }
    let bytes = decode(array, encoding)?;
    Ok((0..array.tuples)
        .map(|i| bytes[i / 8] >> (7 - i % 8) & 1)
        .collect())
}

pub fn decode(array: &DataArray, encoding: &Encoding) -> Result<Vec<u8>> {
    if array.format != "binary" {
        return Err(unsupported(
            "only ascii and inline binary DataArrays are supported",
        ));
    }
    let bytes = unbase64(array.text);
    if encoding.compressed {
        decode_compressed_blocks(&bytes, encoding.header_bytes)
    } else {
        if bytes.len() < encoding.header_bytes {
            return Err(invalid("binary DataArray shorter than its header".into()));
        }
        Ok(bytes[encoding.header_bytes..].to_vec())
    }
}

fn read_uint(bytes: &[u8], offset: usize, size: usize) -> Result<usize> {
    let slice = bytes
        .get(offset..offset + size)
        .ok_or_else(|| invalid("compressed DataArray header is truncated".into()))?;
    Ok(match size {
        4 => u32::from_le_bytes(slice.try_into().unwrap()) as usize,
        8 => u64::from_le_bytes(slice.try_into().unwrap()) as usize,
        _ => unreachable!("header integer size is always 4 or 8"),
    })
}

fn decode_compressed_blocks(bytes: &[u8], header_bytes: usize) -> Result<Vec<u8>> {
    let num_blocks = read_uint(bytes, 0, header_bytes)?;
    let sizes_start = 3 * header_bytes;
    let mut compressed_sizes = Vec::with_capacity(num_blocks);
    for block in 0..num_blocks {
        compressed_sizes.push(read_uint(
            bytes,
            sizes_start + block * header_bytes,
            header_bytes,
        )?);
    }
    let mut offset = sizes_start + num_blocks * header_bytes;
    let mut out = Vec::new();
    for size in compressed_sizes {
        let block = bytes
            .get(offset..offset + size)
            .ok_or_else(|| invalid("compressed DataArray block is truncated".into()))?;
        out.extend(zlib_decode(block)?);
        offset += size;
    }
    Ok(out)
}

fn le_f64(b: &[u8]) -> f64 {
    f64::from_le_bytes(b.try_into().unwrap())
}
fn le_f32(b: &[u8]) -> f64 {
    f32::from_le_bytes(b.try_into().unwrap()) as f64
}
fn le_i64(b: &[u8]) -> i64 {
    i64::from_le_bytes(b.try_into().unwrap())
}
fn le_i32(b: &[u8]) -> i64 {
    i32::from_le_bytes(b.try_into().unwrap()) as i64
}

pub fn parse<T: std::str::FromStr>(token: &str) -> Result<T> {
    token
        .parse()
        .map_err(|_| invalid(format!("could not parse '{token}'")))
}

const INVALID: u8 = 0xFF;

const DECODE_TABLE: [u8; 256] = {
    let mut table = [INVALID; 256];
    let mut i = 0;
    while i < 26 {
        table[b'A' as usize + i] = i as u8;
        table[b'a' as usize + i] = 26 + i as u8;
        i += 1;
    }
    i = 0;
    while i < 10 {
        table[b'0' as usize + i] = 52 + i as u8;
        i += 1;
    }
    table[b'+' as usize] = 62;
    table[b'/' as usize] = 63;
    table
};

pub fn unbase64(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for &byte in text.as_bytes() {
        let value = DECODE_TABLE[byte as usize];
        if value == INVALID {
            continue;
        }
        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
        }
    }
    out
}
