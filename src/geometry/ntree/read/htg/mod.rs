#[cfg(test)]
mod test;

use crate::{
    geometry::ntree::{
        Balancing, Orthotree, Pairing, Rescaling,
        node::{Kind, Node, split::Split},
    },
    math::Scalar,
};
use std::{
    array::from_fn,
    collections::VecDeque,
    fs::read_to_string,
    io::{Error, ErrorKind, Result},
    ops::Add,
    path::Path,
};

pub trait ReadHtg<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_htg(input: P) -> Result<Self>;
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, P> ReadHtg<P>
    for Orthotree<D, L, M, N, T, U>
where
    P: AsRef<Path>,
    T: Add<Output = T> + Copy + Split + Into<usize> + TryFrom<usize>,
    U: Copy + From<usize> + Into<usize>,
{
    fn read_htg(input: P) -> Result<Self> {
        let text = read_to_string(input)?;
        let header = tag(&text, "<VTKFile")?;
        if attribute(header, "compressor").is_some() {
            return Err(unsupported("compressed HTG is not supported"));
        }
        if attribute(header, "type") != Some("HyperTreeGrid") {
            return Err(invalid("not a HyperTreeGrid".into()));
        }
        if matches!(attribute(header, "byte_order"), Some(order) if order != "LittleEndian") {
            return Err(unsupported("big-endian HTG is not supported"));
        }
        let header_bytes = match attribute(header, "header_type") {
            Some("UInt32") | None => 4,
            Some("UInt64") => 8,
            Some(other) => return Err(invalid(format!("unsupported header_type {other}"))),
        };
        let grid = tag(&text, "<HyperTreeGrid")?;
        if matches!(attribute(grid, "BranchFactor"), Some(factor) if factor != "2") {
            return Err(unsupported("only BranchFactor 2 is supported"));
        }
        let dimensions =
            attribute(grid, "Dimensions").ok_or_else(|| invalid("no Dimensions".into()))?;
        let split_axes = dimensions.split_whitespace().filter(|d| *d != "1").count();
        if split_axes != D {
            return Err(invalid(format!(
                "HTG has {split_axes} refined axes but Orthotree was asked for D={D}"
            )));
        }
        let axes = ["XCoordinates", "YCoordinates", "ZCoordinates"];
        let mut lo = [0.0; D];
        let mut hi = [0.0; D];
        for (axis, &name) in axes.iter().take(D).enumerate() {
            let coordinates = floats(&array(&text, name)?, header_bytes)?;
            lo[axis] = coordinates[0];
            hi[axis] = coordinates[1];
        }
        let levels: usize = attribute(tag(&text, "<Tree ")?, "NumberOfLevels")
            .and_then(|n| n.parse().ok())
            .ok_or_else(|| invalid("no NumberOfLevels".into()))?;
        let descriptor = bits(&array(&text, "Descriptor")?, header_bytes)?;
        let root_length = 1usize << (levels - 1);
        let cell = (hi[0] - lo[0]) / root_length as Scalar;
        let rescale = Rescaling {
            center: from_fn(|axis| 0.5 * (lo[axis] + hi[axis])),
            cell,
            half: 0.5 * root_length as Scalar,
        };
        let zero = number(0)?;
        let mut tree = Orthotree {
            balanced: Balancing::None,
            nodes: vec![Node {
                corner: from_fn(|_| zero),
                length: number(root_length)?,
                facets: [None; M],
                kind: Kind::Leaf,
                value: None,
            }],
            paired: Pairing::None,
            rescale,
        };
        let mut queue: VecDeque<U> = VecDeque::from([U::from(0)]);
        let mut bit = 0;
        while let Some(node) = queue.pop_front() {
            if bit >= descriptor.len() {
                break;
            }
            let refined = descriptor[bit] == 1;
            bit += 1;
            if refined {
                tree.subdivide(node).map_err(|e| invalid(e.into()))?;
                queue.extend(tree.nodes[node.into()].orthants().unwrap().iter().copied());
            }
        }
        Ok(tree)
    }
}

fn number<T: TryFrom<usize>>(value: usize) -> Result<T> {
    T::try_from(value).map_err(|_| invalid("tree coordinate does not fit in T".into()))
}

struct Array<'a> {
    data_type: &'a str,
    format: &'a str,
    tuples: usize,
    text: &'a str,
}

fn array<'a>(text: &'a str, name: &str) -> Result<Array<'a>> {
    let at = text
        .find(&format!("Name=\"{name}\""))
        .ok_or_else(|| invalid(format!("missing DataArray {name}")))?;
    let open = text[..at]
        .rfind("<DataArray")
        .ok_or_else(|| invalid(format!("malformed DataArray {name}")))?;
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
        data_type: attribute(attributes, "type")
            .ok_or_else(|| invalid("DataArray without type".into()))?,
        format: attribute(attributes, "format").unwrap_or("ascii"),
        tuples: attribute(attributes, "NumberOfTuples")
            .and_then(|t| t.parse().ok())
            .unwrap_or(0),
        text: text[gt + 1..close].trim(),
    })
}

fn floats(array: &Array, header_bytes: usize) -> Result<Vec<Scalar>> {
    if array.format == "ascii" {
        return array.text.split_whitespace().map(parse).collect();
    }
    let bytes = decode(array, header_bytes)?;
    Ok(match array.data_type {
        "Float64" => bytes
            .chunks(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        "Float32" => bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as Scalar)
            .collect(),
        other => return Err(invalid(format!("unsupported coordinate type {other}"))),
    })
}

fn bits(array: &Array, header_bytes: usize) -> Result<Vec<u8>> {
    if array.format == "ascii" {
        return array.text.split_whitespace().map(parse).collect();
    }
    let bytes = decode(array, header_bytes)?;
    Ok((0..array.tuples)
        .map(|i| bytes[i / 8] >> (7 - i % 8) & 1)
        .collect())
}

fn decode(array: &Array, header_bytes: usize) -> Result<Vec<u8>> {
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

fn parse<T: std::str::FromStr>(token: &str) -> Result<T> {
    token
        .parse()
        .map_err(|_| invalid(format!("could not parse '{token}' in HTG")))
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

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

fn unsupported(message: &str) -> Error {
    Error::new(ErrorKind::Unsupported, message.to_string())
}
