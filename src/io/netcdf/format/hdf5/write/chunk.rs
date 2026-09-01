use crate::io::zlib_encode;

pub(super) struct Plan {
    pub chunk_dims: Vec<u64>,
    pub elem: u32,
    pub blobs: Vec<Blob>,
}

pub(super) struct Blob {
    pub offsets: Vec<u64>,
    pub mask: u32,
    pub bytes: Vec<u8>,
}

fn shuffle(data: &[u8], elem: usize) -> Vec<u8> {
    let n = data.len() / elem;
    let mut out = vec![0u8; data.len()];
    for j in 0..elem {
        for k in 0..n {
            out[j * n + k] = data[k * elem + j];
        }
    }
    out[n * elem..].copy_from_slice(&data[n * elem..]);
    out
}

fn compress(raw: &[u8], elem: usize) -> (u32, Vec<u8>) {
    let shuffled = shuffle(raw, elem);
    let deflated = zlib_encode(&shuffled);
    if deflated.len() < raw.len() {
        (0, deflated)
    } else {
        (0b11, raw.to_vec())
    }
}

/// HDF5's default internal-node K for raw-data chunk B-trees (`HDF5_BTREE_CHUNK_IK_DEF`).
const BTREE_K: usize = 32;
const TARGET_CHUNK_BYTES: u64 = 1 << 20;

/// Split a dataset into chunks along its slowest-varying axis so each chunk is
/// roughly [`TARGET_CHUNK_BYTES`], capped so the B-tree stays a single level.
pub(super) fn plan(dims: &[u64], data: &[u8], elem: usize) -> Plan {
    let rank = dims.len();
    let row: u64 = dims[1..].iter().product::<u64>() * elem as u64;
    let mut slab = dims[0].max(1);
    if let Some(per_target) = TARGET_CHUNK_BYTES.checked_div(row) {
        slab = slab.min(per_target.max(1));
        let min_slab = dims[0].div_ceil(2 * BTREE_K as u64).max(1);
        slab = slab.max(min_slab);
    }
    let mut chunk_dims = dims.to_vec();
    chunk_dims[0] = slab;
    let full = (slab * row) as usize;

    let mut blobs = Vec::new();
    let mut start = 0;
    while start < dims[0] {
        let extent = (dims[0] - start).min(slab);
        let mut raw = data[(start * row) as usize..((start + extent) * row) as usize].to_vec();
        raw.resize(full, 0);
        let (mask, bytes) = compress(&raw, elem);
        let mut offsets = vec![0u64; rank];
        offsets[0] = start;
        blobs.push(Blob {
            offsets,
            mask,
            bytes,
        });
        start += slab;
    }
    Plan {
        chunk_dims,
        elem: elem as u32,
        blobs,
    }
}

pub(super) fn btree_key_size(rank: usize) -> usize {
    8 + 8 * (rank + 1)
}

/// A v1 B-tree node occupies a fixed slab on disk (`2*K` slots plus a trailing
/// key), regardless of how many entries are actually used; libhdf5 reads the
/// whole slab.
pub(super) fn btree_node_size(rank: usize) -> usize {
    let key = btree_key_size(rank);
    24 + 2 * BTREE_K * (key + 8) + key
}

pub(super) fn btree_node(plan: &Plan, chunk_addrs: &[u64]) -> Vec<u8> {
    let rank = plan.chunk_dims.len();
    let mut v = b"TREE".to_vec();
    v.push(1);
    v.push(0);
    v.extend_from_slice(&(plan.blobs.len() as u16).to_le_bytes());
    v.extend_from_slice(&[0xFF; 8]);
    v.extend_from_slice(&[0xFF; 8]);
    for (blob, &addr) in plan.blobs.iter().zip(chunk_addrs) {
        v.extend_from_slice(&(blob.bytes.len() as u32).to_le_bytes());
        v.extend_from_slice(&blob.mask.to_le_bytes());
        for &o in &blob.offsets {
            v.extend_from_slice(&o.to_le_bytes());
        }
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(&addr.to_le_bytes());
    }
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    for &c in &plan.chunk_dims {
        v.extend_from_slice(&c.to_le_bytes());
    }
    v.extend_from_slice(&(plan.elem as u64).to_le_bytes());
    v.resize(btree_node_size(rank), 0);
    v
}
