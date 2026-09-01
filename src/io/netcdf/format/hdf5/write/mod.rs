mod checksum;
mod chunk;
mod message;
#[cfg(test)]
mod test;

use crate::io::netcdf::format::{
    AttValue, Attribute, DimSpec, NC_DOUBLE, NC_FLOAT, NC_INT, VarSpec,
};
use checksum::jenkins;
use message::*;

const DEFLATE_LEVEL: u32 = 4;

fn elem_size(xtype: i32) -> usize {
    match xtype {
        NC_INT | NC_FLOAT => 4,
        NC_DOUBLE => 8,
        other => panic!("cannot write netCDF external type {other} to HDF5"),
    }
}

enum Storage {
    Contiguous(Vec<u8>),
    Chunked(chunk::Plan),
}

struct VarPlan {
    dims: Vec<u64>,
    storage: Storage,
}

fn plan_var(var: &VarSpec, dim_lens: &[u64], data: &[u8], threads: usize) -> VarPlan {
    let dims: Vec<u64> = var.dimids.iter().map(|&d| dim_lens[d]).collect();
    let storage = if dims.is_empty() {
        Storage::Contiguous(data.to_vec())
    } else {
        Storage::Chunked(chunk::plan(&dims, data, elem_size(var.xtype), threads))
    };
    VarPlan { dims, storage }
}

fn attribute_message(att: &Attribute) -> Vec<u8> {
    match &att.value {
        AttValue::Text(s) => attr_text(&att.name, s),
        AttValue::Int(v) => attr_i32(&att.name, v),
        AttValue::Float(v) => attr_f32(&att.name, v),
    }
}

fn dim_scale_messages(index: usize, len: u64) -> Vec<(u8, Vec<u8>)> {
    vec![
        (MSG_DATASPACE, dataspace(&[len])),
        (MSG_DATATYPE, datatype(NC_INT)),
        (MSG_FILL, fill_default()),
        (MSG_LAYOUT, layout_unallocated(len * 4)),
        (MSG_ATTRIBUTE, attr_dimension_scale()),
        (MSG_ATTRIBUTE, attr_pure_dimension_name(len)),
        (MSG_ATTRIBUTE, attr_i32("_Netcdf4Dimid", &[index as i32])),
    ]
}

fn var_messages(
    var: &VarSpec,
    plan: &VarPlan,
    btree_addr: u64,
    data_addr: u64,
) -> Vec<(u8, Vec<u8>)> {
    let mut msgs = vec![
        (MSG_DATASPACE, dataspace(&plan.dims)),
        (MSG_DATATYPE, datatype(var.xtype)),
        (MSG_FILL, fill_value(var.xtype)),
    ];
    match &plan.storage {
        Storage::Contiguous(bytes) => {
            msgs.push((MSG_LAYOUT, layout_contiguous(data_addr, bytes.len() as u64)))
        }
        Storage::Chunked(p) => {
            msgs.push((
                MSG_LAYOUT,
                layout_chunked(btree_addr, &p.chunk_dims, p.elem),
            ));
            msgs.push((
                MSG_FILTER,
                filter_pipeline(elem_size(var.xtype) as u32, DEFLATE_LEVEL),
            ));
        }
    }
    if !var.dimids.is_empty() {
        let dimids: Vec<i32> = var.dimids.iter().map(|&d| d as i32).collect();
        msgs.push((MSG_ATTRIBUTE, attr_i32("_Netcdf4Coordinates", &dimids)));
    }
    for att in &var.atts {
        msgs.push((MSG_ATTRIBUTE, attribute_message(att)));
    }
    msgs
}

fn object_header(messages: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let chunk0: usize = messages.iter().map(|(_, b)| 4 + b.len()).sum();
    let mut v = b"OHDR".to_vec();
    v.push(2);
    v.push(0x02);
    v.extend_from_slice(&(chunk0 as u32).to_le_bytes());
    for (typ, body) in messages {
        v.push(*typ);
        v.extend_from_slice(&(body.len() as u16).to_le_bytes());
        v.push(0);
        v.extend_from_slice(body);
    }
    let sum = jenkins(&v);
    v.extend_from_slice(&sum.to_le_bytes());
    v
}

fn object_header_len(messages: &[(u8, Vec<u8>)]) -> usize {
    10 + messages.iter().map(|(_, b)| 4 + b.len()).sum::<usize>() + 4
}

fn superblock(root_oh: u64, eof: u64) -> Vec<u8> {
    let mut v = vec![
        0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n', 2, 8, 8, 0,
    ];
    v.extend_from_slice(&0u64.to_le_bytes());
    v.extend_from_slice(&u64::MAX.to_le_bytes());
    v.extend_from_slice(&eof.to_le_bytes());
    v.extend_from_slice(&root_oh.to_le_bytes());
    let sum = jenkins(&v);
    v.extend_from_slice(&sum.to_le_bytes());
    v
}

pub(in crate::io::netcdf) fn write(
    dims: &[DimSpec],
    global: &[Attribute],
    vars: &[VarSpec],
    data: &[Vec<u8>],
    threads: usize,
) -> Vec<u8> {
    let dim_lens: Vec<u64> = dims.iter().map(|d| d.len).collect();
    let plans: Vec<VarPlan> = vars
        .iter()
        .zip(data)
        .map(|(v, d)| plan_var(v, &dim_lens, d, threads))
        .collect();

    let dim_scale_msgs: Vec<Vec<(u8, Vec<u8>)>> = (0..dims.len())
        .map(|i| dim_scale_messages(i, dim_lens[i]))
        .collect();

    let placeholder_var_msgs: Vec<Vec<(u8, Vec<u8>)>> = vars
        .iter()
        .zip(&plans)
        .map(|(v, p)| var_messages(v, p, 0, 0))
        .collect();

    const SUPERBLOCK: u64 = 48;
    let mut cursor = SUPERBLOCK;
    let root_oh = cursor;

    let root_placeholder = root_messages(
        dims,
        vars,
        global,
        &vec![0u64; dims.len()],
        &vec![0u64; vars.len()],
    );
    cursor += object_header_len(&root_placeholder) as u64;

    let mut dim_oh = Vec::with_capacity(dims.len());
    for msgs in &dim_scale_msgs {
        dim_oh.push(cursor);
        cursor += object_header_len(msgs) as u64;
    }
    let mut var_oh = Vec::with_capacity(vars.len());
    for msgs in &placeholder_var_msgs {
        var_oh.push(cursor);
        cursor += object_header_len(msgs) as u64;
    }

    let mut var_btree = vec![0u64; vars.len()];
    let mut var_data = vec![0u64; vars.len()];
    let mut var_chunk_addrs: Vec<Vec<u64>> = vec![Vec::new(); vars.len()];
    for (i, plan) in plans.iter().enumerate() {
        match &plan.storage {
            Storage::Contiguous(bytes) => {
                var_data[i] = cursor;
                cursor += bytes.len() as u64;
            }
            Storage::Chunked(p) => {
                var_btree[i] = cursor;
                cursor += chunk::btree_node(p, &vec![0u64; p.blobs.len()]).len() as u64;
                for blob in &p.blobs {
                    var_chunk_addrs[i].push(cursor);
                    cursor += blob.bytes.len() as u64;
                }
            }
        }
    }
    let eof = cursor;

    let mut out = superblock(root_oh, eof);
    out.extend(object_header(&root_messages(
        dims, vars, global, &dim_oh, &var_oh,
    )));
    for msgs in &dim_scale_msgs {
        out.extend(object_header(msgs));
    }
    for (i, (var, plan)) in vars.iter().zip(&plans).enumerate() {
        out.extend(object_header(&var_messages(
            var,
            plan,
            var_btree[i],
            var_data[i],
        )));
    }
    for (i, plan) in plans.iter().enumerate() {
        match &plan.storage {
            Storage::Contiguous(bytes) => out.extend_from_slice(bytes),
            Storage::Chunked(p) => {
                out.extend(chunk::btree_node(p, &var_chunk_addrs[i]));
                for blob in &p.blobs {
                    out.extend_from_slice(&blob.bytes);
                }
            }
        }
    }
    debug_assert_eq!(out.len() as u64, eof);
    out
}

fn root_messages(
    dims: &[DimSpec],
    vars: &[VarSpec],
    global: &[Attribute],
    dim_oh: &[u64],
    var_oh: &[u64],
) -> Vec<(u8, Vec<u8>)> {
    let mut msgs = vec![(MSG_LINK_INFO, link_info()), (MSG_GROUP_INFO, group_info())];
    for (d, &addr) in dims.iter().zip(dim_oh) {
        msgs.push((MSG_LINK, link(&d.name, addr)));
    }
    for (v, &addr) in vars.iter().zip(var_oh) {
        msgs.push((MSG_LINK, link(&v.name, addr)));
    }
    for att in global {
        msgs.push((MSG_ATTRIBUTE, attribute_message(att)));
    }
    msgs
}
