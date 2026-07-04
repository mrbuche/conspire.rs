#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::Scalar,
};
use std::{
    fs::read_to_string,
    io::{Error, ErrorKind, Result},
    path::Path,
};

pub trait ReadVtu<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_vtu(input: P) -> Result<Self>;
}

impl<const D: usize, P> ReadVtu<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn read_vtu(input: P) -> Result<Self> {
        if D != 2 && D != 3 {
            return Err(unsupported("VTU supports only 2D or 3D meshes"));
        }
        let text = match read_to_string(input) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::InvalidData => {
                return Err(unsupported(
                    "raw appended VTU is not supported (use ascii or binary)",
                ));
            }
            Err(error) => return Err(error),
        };
        let header = tag(&text, "<VTKFile")?;
        if attribute(header, "compressor").is_some() {
            return Err(unsupported("compressed VTU is not supported"));
        }
        if attribute(header, "type") != Some("UnstructuredGrid") {
            return Err(invalid("VTU is not an UnstructuredGrid".into()));
        }
        if matches!(attribute(header, "byte_order"), Some(order) if order != "LittleEndian") {
            return Err(unsupported("big-endian VTU is not supported"));
        }
        let header_bytes = match attribute(header, "header_type") {
            Some("UInt32") | None => 4,
            Some("UInt64") => 8,
            Some(other) => return Err(invalid(format!("unsupported header_type {other}"))),
        };

        let points_region = region(&text, "Points")?;
        let points = floats(&data_array(points_region, None)?, header_bytes)?;
        let components = attribute(tag(points_region, "<DataArray")?, "NumberOfComponents")
            .and_then(|n| n.parse().ok())
            .unwrap_or(3);

        let cells_region = region(&text, "Cells")?;
        let connectivity = integers(
            &data_array(cells_region, Some("connectivity"))?,
            header_bytes,
        )?;
        let offsets = integers(&data_array(cells_region, Some("offsets"))?, header_bytes)?;
        let types = integers(&data_array(cells_region, Some("types"))?, header_bytes)?;

        let coordinates: Coordinates<D> = points
            .chunks(components)
            .map(|point| std::array::from_fn(|i| point[i]).into())
            .collect();
        let mut mesh = Mesh::<D>::from((blocks(&connectivity, &offsets, &types)?, coordinates));
        if let Ok(point_data) = region(&text, "PointData") {
            let mut node_sets = Vec::new();
            let mut set = 1;
            while point_data.contains(&format!("Name=\"NodeSet{set}\"")) {
                let flags = integers(
                    &data_array(point_data, Some(&format!("NodeSet{set}")))?,
                    header_bytes,
                )?;
                node_sets.push(
                    flags
                        .iter()
                        .enumerate()
                        .filter_map(|(node, &flag)| (flag != 0).then_some(node))
                        .collect(),
                );
                set += 1;
            }
            if !node_sets.is_empty() {
                mesh.set_node_sets(node_sets.into());
            }
        }
        Ok(mesh)
    }
}

#[derive(Clone, Copy)]
struct DataArray<'a> {
    data_type: &'a str,
    format: &'a str,
    text: &'a str,
}

fn blocks(connectivity: &[i64], offsets: &[i64], types: &[i64]) -> Result<Vec<Connectivity>> {
    let mut cells: Vec<(i64, &[i64])> = Vec::with_capacity(types.len());
    let mut start = 0;
    for (cell, &end) in offsets.iter().enumerate() {
        cells.push((types[cell], &connectivity[start..end as usize]));
        start = end as usize;
    }
    let mut blocks = Vec::new();
    let mut from = 0;
    while from < cells.len() {
        let mut to = from;
        while to < cells.len() && cells[to].0 == cells[from].0 {
            to += 1;
        }
        blocks.push(block(cells[from].0, &cells[from..to])?);
        from = to;
    }
    Ok(blocks)
}

fn block(cell_type: i64, cells: &[(i64, &[i64])]) -> Result<Connectivity> {
    Ok(match cell_type {
        5 => Connectivity::Triangular(arrays::<3>(cells)?.into()),
        9 => Connectivity::Quadrilateral(arrays::<4>(cells)?.into()),
        10 => Connectivity::Tetrahedral(arrays::<4>(cells)?.into()),
        12 => Connectivity::Hexahedral(arrays::<8>(cells)?.into()),
        13 => Connectivity::Wedge(arrays::<6>(cells)?.into()),
        14 => Connectivity::Pyramidal(arrays::<5>(cells)?.into()),
        other => return Err(invalid(format!("unsupported VTK cell type: {other}"))),
    })
}

fn arrays<const N: usize>(cells: &[(i64, &[i64])]) -> Result<Vec<[usize; N]>> {
    cells
        .iter()
        .map(|(_, nodes)| {
            if nodes.len() != N {
                return Err(invalid(format!(
                    "cell has {} nodes, expected {N}",
                    nodes.len()
                )));
            }
            Ok(std::array::from_fn(|i| nodes[i] as usize))
        })
        .collect()
}

fn floats(array: &DataArray, header_bytes: usize) -> Result<Vec<Scalar>> {
    if array.format == "ascii" {
        return array.text.split_whitespace().map(parse).collect();
    }
    let bytes = decode(array, header_bytes)?;
    Ok(match array.data_type {
        "Float64" => bytes.chunks(8).map(le_f64).collect(),
        "Float32" => bytes.chunks(4).map(le_f32).collect(),
        other => return Err(invalid(format!("unsupported point type {other}"))),
    })
}

fn integers(array: &DataArray, header_bytes: usize) -> Result<Vec<i64>> {
    if array.format == "ascii" {
        return array.text.split_whitespace().map(parse).collect();
    }
    let bytes = decode(array, header_bytes)?;
    Ok(match array.data_type {
        "Int64" | "UInt64" => bytes.chunks(8).map(le_i64).collect(),
        "Int32" | "UInt32" => bytes.chunks(4).map(le_i32).collect(),
        "Int8" | "UInt8" => bytes.iter().map(|&b| b as i64).collect(),
        other => return Err(invalid(format!("unsupported cell-data type {other}"))),
    })
}

fn decode(array: &DataArray, header_bytes: usize) -> Result<Vec<u8>> {
    if array.format != "binary" {
        return Err(unsupported(
            "only ascii and inline binary DataArrays are supported",
        ));
    }
    let mut bytes = unbase64(array.text);
    if bytes.len() < header_bytes {
        return Err(invalid("binary DataArray shorter than its header".into()));
    }
    Ok(bytes.split_off(header_bytes))
}

fn le_f64(b: &[u8]) -> Scalar {
    f64::from_le_bytes(b.try_into().unwrap())
}
fn le_f32(b: &[u8]) -> Scalar {
    f32::from_le_bytes(b.try_into().unwrap()) as Scalar
}
fn le_i64(b: &[u8]) -> i64 {
    i64::from_le_bytes(b.try_into().unwrap())
}
fn le_i32(b: &[u8]) -> i64 {
    i32::from_le_bytes(b.try_into().unwrap()) as i64
}

fn parse<T: std::str::FromStr>(token: &str) -> Result<T> {
    token
        .parse()
        .map_err(|_| invalid(format!("could not parse '{token}' in VTU")))
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

fn tag<'a>(text: &'a str, open: &str) -> Result<&'a str> {
    let start = text
        .find(open)
        .ok_or_else(|| invalid(format!("missing {open} in VTU")))?;
    let end = text[start..]
        .find('>')
        .ok_or_else(|| invalid(format!("unterminated {open} in VTU")))?;
    Ok(&text[start..start + end])
}

fn region<'a>(text: &'a str, name: &str) -> Result<&'a str> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let start = text
        .find(&open)
        .ok_or_else(|| invalid(format!("missing <{name}> in VTU")))?;
    let end = text
        .find(&close)
        .ok_or_else(|| invalid(format!("missing </{name}> in VTU")))?;
    Ok(&text[start..end])
}

fn data_array<'a>(region: &'a str, name: Option<&str>) -> Result<DataArray<'a>> {
    let mut rest = region;
    loop {
        let open = rest
            .find("<DataArray")
            .ok_or_else(|| invalid("missing DataArray in VTU".into()))?;
        let attributes_end = rest[open..]
            .find('>')
            .ok_or_else(|| invalid("unterminated DataArray in VTU".into()))?
            + open;
        let attributes = &rest[open..attributes_end];
        if name.is_none() || attribute(attributes, "Name") == name {
            let close = rest[attributes_end..]
                .find("</DataArray>")
                .ok_or_else(|| invalid("unclosed DataArray in VTU".into()))?
                + attributes_end;
            return Ok(DataArray {
                data_type: attribute(attributes, "type")
                    .ok_or_else(|| invalid("DataArray without type".into()))?,
                format: attribute(attributes, "format").unwrap_or("ascii"),
                text: rest[attributes_end + 1..close].trim(),
            });
        }
        rest = &rest[attributes_end..];
    }
}

fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=\"");
    let start = tag.find(&key)? + key.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

fn unsupported(message: &str) -> Error {
    Error::new(ErrorKind::Unsupported, message.to_string())
}
