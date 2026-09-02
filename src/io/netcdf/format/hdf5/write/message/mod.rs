use crate::io::netcdf::format::{NC_DOUBLE, NC_FLOAT, NC_INT};

pub(super) const MSG_DATASPACE: u8 = 0x01;
pub(super) const MSG_DATATYPE: u8 = 0x03;
pub(super) const MSG_FILL: u8 = 0x05;
pub(super) const MSG_LINK: u8 = 0x06;
pub(super) const MSG_LAYOUT: u8 = 0x08;
pub(super) const MSG_GROUP_INFO: u8 = 0x0A;
pub(super) const MSG_FILTER: u8 = 0x0B;
pub(super) const MSG_ATTRIBUTE: u8 = 0x0C;
pub(super) const MSG_LINK_INFO: u8 = 0x02;

const UNDEF: [u8; 8] = [0xFF; 8];

fn extend_u16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn extend_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn extend_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}

pub(super) fn link_info() -> Vec<u8> {
    let mut v = vec![0, 0];
    v.extend_from_slice(&UNDEF);
    v.extend_from_slice(&UNDEF);
    v
}

pub(super) fn group_info() -> Vec<u8> {
    vec![0, 0]
}

pub(super) fn link(name: &str, object_header: u64) -> Vec<u8> {
    assert!(
        name.len() < 256,
        "link name too long for a 1-byte length field"
    );
    let mut v = vec![1, 0, name.len() as u8];
    v.extend_from_slice(name.as_bytes());
    extend_u64(&mut v, object_header);
    v
}

pub(super) fn dataspace(dims: &[u64]) -> Vec<u8> {
    if dims.is_empty() {
        return vec![2, 0, 0, 0];
    }
    let mut v = vec![2, dims.len() as u8, 0x01, 1];
    for &d in dims {
        extend_u64(&mut v, d);
    }
    for &d in dims {
        extend_u64(&mut v, d);
    }
    v
}

pub(super) fn datatype(xtype: i32) -> Vec<u8> {
    match xtype {
        NC_INT => vec![0x10, 0x08, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0],
        NC_FLOAT => vec![
            0x11, 0x20, 0x1f, 0, 4, 0, 0, 0, 0, 0, 32, 0, 23, 8, 0, 23, 127, 0, 0, 0,
        ],
        NC_DOUBLE => vec![
            0x11, 0x20, 0x3f, 0, 8, 0, 0, 0, 0, 0, 64, 0, 52, 11, 0, 52, 0xFF, 3, 0, 0,
        ],
        other => panic!("cannot write netCDF external type {other} to HDF5"),
    }
}

pub(super) fn datatype_string(size: u32) -> Vec<u8> {
    let mut v = vec![0x13, 0, 0, 0];
    extend_u32(&mut v, size);
    v
}

pub(super) fn fill_default() -> Vec<u8> {
    vec![3, 0x0a]
}

pub(super) fn fill_value(xtype: i32) -> Vec<u8> {
    let bytes: Vec<u8> = match xtype {
        NC_INT => (-2147483647i32).to_le_bytes().to_vec(),
        NC_FLOAT => f32::from_bits(0x7cf0_0000).to_le_bytes().to_vec(),
        NC_DOUBLE => f64::from_bits(0x479e_0000_0000_0000).to_le_bytes().to_vec(),
        other => panic!("cannot write netCDF external type {other} to HDF5"),
    };
    let mut v = vec![3, 0x2a];
    extend_u32(&mut v, bytes.len() as u32);
    v.extend_from_slice(&bytes);
    v
}

pub(super) fn layout_contiguous(addr: u64, size: u64) -> Vec<u8> {
    let mut v = vec![3, 1];
    extend_u64(&mut v, addr);
    extend_u64(&mut v, size);
    v
}

pub(super) fn layout_unallocated(size: u64) -> Vec<u8> {
    layout_contiguous(u64::MAX, size)
}

pub(super) fn layout_chunked(btree: u64, chunk: &[u64], elem: u32) -> Vec<u8> {
    let mut v = vec![3, 2, (chunk.len() + 1) as u8];
    extend_u64(&mut v, btree);
    for &c in chunk {
        extend_u32(&mut v, c as u32);
    }
    extend_u32(&mut v, elem);
    v
}

pub(super) fn filter_pipeline(elem_size: u32, level: u32) -> Vec<u8> {
    let mut v = vec![2, 2];
    for (id, client) in [(2u16, elem_size), (1, level)] {
        extend_u16(&mut v, id);
        extend_u16(&mut v, 0);
        extend_u16(&mut v, 1);
        extend_u32(&mut v, client);
    }
    v
}

fn attribute(name: &str, datatype: &[u8], dataspace: &[u8], data: &[u8]) -> Vec<u8> {
    let mut v = vec![3, 0];
    extend_u16(&mut v, name.len() as u16 + 1);
    extend_u16(&mut v, datatype.len() as u16);
    extend_u16(&mut v, dataspace.len() as u16);
    v.push(0);
    v.extend_from_slice(name.as_bytes());
    v.push(0);
    v.extend_from_slice(datatype);
    v.extend_from_slice(dataspace);
    v.extend_from_slice(data);
    v
}

pub(super) fn attr_text(name: &str, value: &str) -> Vec<u8> {
    let mut data = value.as_bytes().to_vec();
    data.push(0);
    attribute(
        name,
        &datatype_string(data.len() as u32),
        &dataspace(&[]),
        &data,
    )
}

pub(super) fn attr_i32(name: &str, values: &[i32]) -> Vec<u8> {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let space = if values.len() == 1 {
        dataspace(&[])
    } else {
        dataspace(&[values.len() as u64])
    };
    attribute(name, &datatype(NC_INT), &space, &data)
}

pub(super) fn attr_f32(name: &str, values: &[f32]) -> Vec<u8> {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let space = if values.len() == 1 {
        dataspace(&[])
    } else {
        dataspace(&[values.len() as u64])
    };
    attribute(name, &datatype(NC_FLOAT), &space, &data)
}

pub(super) fn attr_dimension_scale() -> Vec<u8> {
    let value = b"DIMENSION_SCALE\0";
    attribute(
        "CLASS",
        &datatype_string(value.len() as u32),
        &dataspace(&[]),
        value,
    )
}

pub(super) fn attr_pure_dimension_name(size: u64) -> Vec<u8> {
    let text = format!("This is a netCDF dimension but not a netCDF variable.{size:>11}");
    let mut data = text.into_bytes();
    data.push(0);
    attribute(
        "NAME",
        &datatype_string(data.len() as u32),
        &dataspace(&[]),
        &data,
    )
}
