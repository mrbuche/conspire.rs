#[cfg(test)]
mod test;

use super::{Filter, Layout, Sizes, cstr, header::object_header, u16, u32, u64, uint, undefined};
use crate::io::netcdf::format::{
    AttValue, Attribute, DimSpec, NC_DOUBLE, NC_FLOAT, NC_INT, Storage, VarSpec,
};

pub(super) fn parse(
    b: &[u8],
    sizes: &Sizes,
    name: &str,
    header: usize,
    dims: &mut Vec<DimSpec>,
    vars: &mut Vec<VarSpec>,
) {
    let mut shape = Vec::new();
    let mut dt = None;
    let mut layout = None;
    let mut fill = None;
    let mut filters = Vec::new();
    let mut atts = Vec::new();
    let mut dimension_scale = false;
    let mut not_a_variable = false;
    for m in object_header(b, sizes, header) {
        let d = &b[m.start..m.end];
        match m.typ {
            0x0001 => shape = dataspace(d),
            0x0003 => dt = Some(datatype(d)),
            0x0005 => fill = fill_value(d),
            0x0008 => layout = Some(data_layout(sizes, d)),
            0x000B => filters = filter_pipeline(d),
            0x000C => {
                if let Some(att) = attribute(d) {
                    match att.name.as_str() {
                        "CLASS" if matches!(&att.value, AttValue::Text(t) if t == "DIMENSION_SCALE") => {
                            dimension_scale = true
                        }
                        "NAME"
                            if matches!(&att.value, AttValue::Text(t)
                                if t.starts_with("This is a netCDF dimension but not a netCDF variable")) =>
                        {
                            not_a_variable = true
                        }
                        _ => {}
                    }
                    atts.push(att);
                }
            }
            0x0015 => {
                let flags = b[m.start + 1];
                let q = m.start + 2 + if flags & 0x01 != 0 { 2 } else { 0 };
                assert!(
                    undefined(b, q, sizes.offset),
                    "dense attribute storage is unsupported (variable {name})"
                );
            }
            _ => {}
        }
    }

    if dimension_scale {
        dims.push(DimSpec {
            name: name.to_string(),
            len: shape.first().copied().unwrap_or(0),
        });
        if not_a_variable {
            return;
        }
    }

    let dt = dt.expect("dataset object header without a datatype message");
    let Some(xtype) = xtype(&dt) else { return };
    let layout = match layout.unwrap_or(RawLayout::Fill) {
        RawLayout::Fill => Layout::Fill,
        RawLayout::Contiguous { addr, size } => Layout::Contiguous { addr, size },
        RawLayout::Chunked { btree_addr, chunk } => Layout::Chunked {
            btree_addr,
            offset_size: sizes.offset,
            chunk,
        },
    };
    vars.push(VarSpec {
        name: name.to_string(),
        xtype,
        dimids: Vec::new(),
        atts,
        begin: 0,
        vsize: 0,
        storage: Storage::Hdf5 {
            little_endian: dt.little_endian,
            shape,
            layout,
            filters,
            fill,
        },
    });
}

pub(super) struct Datatype {
    pub class: u8,
    pub size: usize,
    pub little_endian: bool,
}

pub(super) fn datatype(d: &[u8]) -> Datatype {
    let class = d[0] & 0x0f;
    Datatype {
        class,
        size: u32(d, 4) as usize,
        little_endian: !matches!(class, 0 | 1) || d[1] & 0x01 == 0,
    }
}

fn xtype(dt: &Datatype) -> Option<i32> {
    Some(match (dt.class, dt.size) {
        (1, 8) => NC_DOUBLE,
        (1, 4) => NC_FLOAT,
        (0, 4) => NC_INT,
        _ => return None,
    })
}

pub(super) fn dataspace(d: &[u8]) -> Vec<u64> {
    assert_eq!(d[0], 2, "unsupported dataspace version {}", d[0]);
    let rank = d[1] as usize;
    (0..rank).map(|i| u64(d, 4 + 8 * i)).collect()
}

pub(super) enum RawLayout {
    Fill,
    Contiguous { addr: usize, size: usize },
    Chunked { btree_addr: usize, chunk: Vec<u64> },
}

pub(super) fn data_layout(sizes: &Sizes, d: &[u8]) -> RawLayout {
    assert_eq!(d[0], 3, "unsupported data layout message version {}", d[0]);
    match d[1] {
        1 => {
            if undefined(d, 2, sizes.offset) {
                return RawLayout::Fill;
            }
            RawLayout::Contiguous {
                addr: uint(d, 2, sizes.offset) as usize,
                size: uint(d, 2 + sizes.offset, sizes.length) as usize,
            }
        }
        2 => {
            let dimensionality = d[2] as usize;
            if undefined(d, 3, sizes.offset) {
                return RawLayout::Fill;
            }
            let btree_addr = uint(d, 3, sizes.offset) as usize;
            let base = 3 + sizes.offset;
            let vals: Vec<u32> = (0..dimensionality).map(|i| u32(d, base + 4 * i)).collect();
            RawLayout::Chunked {
                btree_addr,
                chunk: vals[..dimensionality - 1]
                    .iter()
                    .map(|&v| v as u64)
                    .collect(),
            }
        }
        c => panic!("unsupported data layout class {c}"),
    }
}

// Fill value message. v1/v2 carry `defined` at byte 3 and size+value from
// byte 4; v3 packs version+flags into bytes 0-1 (bit 5 = value present) and
// size+value from byte 2. Returns the raw fill bytes, or None when no value is
// stored (callers then fall back to zero-fill).
fn fill_value(d: &[u8]) -> Option<Vec<u8>> {
    assert!(
        matches!(d[0], 1..=3),
        "unsupported fill value message version {}",
        d[0]
    );
    let (defined, at) = if d[0] == 3 {
        (d[1] & 0x20 != 0, 2)
    } else {
        (d[3] != 0, 4)
    };
    defined.then(|| {
        let size = u32(d, at) as usize;
        d[at + 4..at + 4 + size].to_vec()
    })
}

pub(super) fn filter_pipeline(d: &[u8]) -> Vec<Filter> {
    assert_eq!(d[0], 2, "unsupported filter pipeline version {}", d[0]);
    let count = d[1] as usize;
    let mut p = 2;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let id = u16(d, p);
        let client = u16(d, p + 4) as usize;
        p += 6 + 4 * client;
        out.push(match id {
            1 => Filter::Deflate,
            2 => Filter::Shuffle,
            other => panic!("unsupported HDF5 filter {other}"),
        });
    }
    out
}

pub(super) fn attribute(d: &[u8]) -> Option<Attribute> {
    assert_eq!(d[0], 3, "unsupported attribute message version {}", d[0]);
    let name_size = u16(d, 2) as usize;
    let dt_size = u16(d, 4) as usize;
    let ds_size = u16(d, 6) as usize;
    let mut p = 9;
    let name = cstr(&d[p..p + name_size]);
    p += name_size;
    let dt = datatype(&d[p..p + dt_size]);
    p += dt_size + ds_size;
    (dt.class == 3).then(|| Attribute {
        name,
        value: AttValue::Text(cstr(&d[p..(p + dt.size).min(d.len())])),
    })
}
