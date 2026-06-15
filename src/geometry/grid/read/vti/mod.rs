use crate::{geometry::grid::Grid, io::NpyType};
use std::{
    array::from_fn,
    fs::read_to_string,
    io::{Error, ErrorKind, Result},
    path::Path,
};

pub(super) fn read<const D: usize, T, P>(path: P) -> Result<Grid<D, T>>
where
    T: NpyType,
    P: AsRef<Path>,
{
    if D != 2 && D != 3 {
        return Err(unsupported("VTI supports only pixels or voxels"));
    }
    let text = read_to_string(path)?;
    let file = tag(&text, "<VTKFile")?;
    if attribute(file, "compressor").is_some() {
        return Err(unsupported("compressed VTI is not supported"));
    }
    if attribute(file, "type") != Some("ImageData") {
        return Err(invalid("VTI is not ImageData".into()));
    }
    if matches!(attribute(file, "byte_order"), Some(order) if order != "LittleEndian") {
        return Err(unsupported("big-endian VTI is not supported"));
    }
    let header_bytes = match attribute(file, "header_type") {
        Some("UInt32") | None => 4,
        Some("UInt64") => 8,
        Some(other) => return Err(invalid(format!("unsupported header_type {other}"))),
    };
    let image = tag(&text, "<ImageData")?;
    let bounds: Vec<i64> = attribute(image, "WholeExtent")
        .ok_or_else(|| invalid("no WholeExtent".into()))?
        .split_whitespace()
        .map(|n| n.parse().map_err(|_| invalid("bad WholeExtent".into())))
        .collect::<Result<_>>()?;
    if bounds.len() != 6 {
        return Err(invalid("WholeExtent must have 6 entries".into()));
    }
    let cells: [usize; 3] = from_fn(|axis| (bounds[2 * axis + 1] - bounds[2 * axis]) as usize);
    let total: usize = cells.iter().product();
    let nel: [usize; D] = from_fn(|axis| cells[axis]);
    if nel.iter().product::<usize>() != total {
        return Err(invalid(format!(
            "VTI extent {cells:?} does not reduce to {D} dimensions"
        )));
    }
    let array = data_array(&text)?;
    if array.data_type != vtk_type(T::DESCR) {
        return Err(invalid(format!(
            "DataArray type {} does not match the requested scalar",
            array.data_type
        )));
    }
    if array.format != "binary" {
        return Err(unsupported(
            "only inline binary VTI DataArrays are supported",
        ));
    }
    let mut bytes = unbase64(array.text);
    if bytes.len() < header_bytes {
        return Err(invalid("binary DataArray shorter than its header".into()));
    }
    let data: Vec<T> = bytes
        .split_off(header_bytes)
        .chunks(T::SIZE)
        .take(total)
        .map(T::read_le)
        .collect();
    Ok(Grid::new(data, nel))
}

fn vtk_type(descr: &str) -> &'static str {
    match descr {
        "|u1" => "UInt8",
        "|i1" => "Int8",
        "<u2" => "UInt16",
        "<i2" => "Int16",
        "<u4" => "UInt32",
        "<i4" => "Int32",
        "<u8" => "UInt64",
        "<i8" => "Int64",
        "<f4" => "Float32",
        "<f8" => "Float64",
        _ => "UInt8",
    }
}

struct Array<'a> {
    data_type: &'a str,
    format: &'a str,
    text: &'a str,
}

fn data_array(text: &str) -> Result<Array<'_>> {
    let open = text
        .find("<DataArray")
        .ok_or_else(|| invalid("no DataArray".into()))?;
    let gt = text[open..]
        .find('>')
        .ok_or_else(|| invalid("unterminated DataArray".into()))?
        + open;
    let attributes = &text[open..gt];
    let close = text[gt..]
        .find("</DataArray>")
        .ok_or_else(|| invalid("unclosed DataArray".into()))?
        + gt;
    Ok(Array {
        data_type: attribute(attributes, "type").ok_or_else(|| invalid("DataArray type".into()))?,
        format: attribute(attributes, "format").unwrap_or("ascii"),
        text: text[gt + 1..close].trim(),
    })
}

fn tag<'a>(text: &'a str, open: &str) -> Result<&'a str> {
    let start = text
        .find(open)
        .ok_or_else(|| invalid(format!("missing {open}")))?;
    let end = text[start..]
        .find('>')
        .ok_or_else(|| invalid(format!("unterminated {open}")))?;
    Ok(&text[start..start + end])
}

fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=\"");
    let start = tag.find(&key)? + key.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

fn unbase64(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => (c - b'A') as u32,
        b'a'..=b'z' => (c - b'a' + 26) as u32,
        b'0'..=b'9' => (c - b'0' + 52) as u32,
        b'+' => 62,
        b'/' => 63,
        _ => 0,
    };
    let chars: Vec<u8> = text
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'=')
        .collect();
    let mut out = Vec::with_capacity(chars.len() / 4 * 3);
    for chunk in chars.chunks(4) {
        let mut block = 0u32;
        for &c in chunk {
            block = (block << 6) | value(c);
        }
        block <<= 6 * (4 - chunk.len());
        for shift in (0..chunk.len().saturating_sub(1)).map(|i| 16 - 8 * i) {
            out.push((block >> shift) as u8);
        }
    }
    out
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

fn unsupported(message: &str) -> Error {
    Error::new(ErrorKind::Unsupported, message.to_string())
}
