mod chunk;
mod dataset;
mod group;
mod header;
#[cfg(test)]
mod test;
mod write;

use super::Parsed;

pub(in crate::io::netcdf) use chunk::read_data;
pub(in crate::io::netcdf) use write::write;

pub(super) const OHDR: &[u8; 4] = b"OHDR";
pub(super) const TREE: &[u8; 4] = b"TREE";
pub(super) const FRHP: &[u8; 4] = b"FRHP";
pub(super) const FHDB: &[u8; 4] = b"FHDB";
pub(super) const FHIB: &[u8; 4] = b"FHIB";

pub(super) struct Sizes {
    pub offset: usize,
    pub length: usize,
}

pub(in crate::io::netcdf) enum Layout {
    Fill,
    Contiguous {
        addr: usize,
        size: usize,
    },
    Chunked {
        btree_addr: usize,
        offset_size: usize,
        chunk: Vec<u64>,
    },
}

pub(in crate::io::netcdf) enum Filter {
    Deflate,
    Shuffle,
}

// HDF5 shuffle filter: `encode` groups byte j of every element together (planar);
// decode is the exact inverse. Trailing bytes that don't fill an element pass
// through untouched.
pub(super) fn shuffle(data: &[u8], elem: usize, encode: bool) -> Vec<u8> {
    let n = data.len() / elem;
    let mut out = vec![0u8; data.len()];
    for j in 0..elem {
        for k in 0..n {
            let (planar, interleaved) = (j * n + k, k * elem + j);
            if encode {
                out[planar] = data[interleaved];
            } else {
                out[interleaved] = data[planar];
            }
        }
    }
    let tail = n * elem;
    out[tail..].copy_from_slice(&data[tail..]);
    out
}

pub(super) fn u16(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([b[p], b[p + 1]])
}

pub(super) fn u32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

pub(super) fn u64(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
}

pub(super) fn uint(b: &[u8], p: usize, width: usize) -> u64 {
    let mut v = 0;
    for i in 0..width {
        v |= (b[p + i] as u64) << (8 * i);
    }
    v
}

pub(super) fn undefined(b: &[u8], p: usize, width: usize) -> bool {
    (0..width).all(|i| b[p + i] == 0xFF)
}

pub(super) fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

pub(in crate::io::netcdf) fn superblock_offset(b: &[u8]) -> Option<usize> {
    const SIG: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];
    let mut o = 0;
    while o + 8 <= b.len() {
        if b[o..o + 8] == SIG {
            return Some(o);
        }
        o = if o == 0 { 512 } else { o << 1 };
    }
    None
}

pub(in crate::io::netcdf) fn parse(b: &[u8], base: usize) -> Parsed {
    let (sizes, root) = superblock(b, base);
    let mut dims = Vec::new();
    let mut vars = Vec::new();
    for (name, oh) in group::links(b, &sizes, root) {
        dataset::parse(b, &sizes, &name, oh, &mut dims, &mut vars);
    }
    Parsed {
        dims,
        gatts: Vec::new(),
        vars,
    }
}

fn superblock(b: &[u8], base: usize) -> (Sizes, usize) {
    let p = base + 8;
    assert!(
        matches!(b[p], 2 | 3),
        "unsupported HDF5 superblock version {}",
        b[p]
    );
    let sizes = Sizes {
        offset: b[p + 1] as usize,
        length: b[p + 2] as usize,
    };
    let root = uint(b, p + 4 + 3 * sizes.offset, sizes.offset) as usize;
    (sizes, root)
}
