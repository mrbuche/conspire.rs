#[cfg(test)]
mod test;

use super::{OHDR, Sizes, u16, uint};

pub(super) struct Message {
    pub typ: u16,
    pub start: usize,
    pub end: usize,
}

pub(super) fn object_header(b: &[u8], sizes: &Sizes, addr: usize) -> Vec<Message> {
    assert_eq!(
        &b[addr..addr + 4],
        OHDR,
        "expected a version 2 object header"
    );
    let flags = b[addr + 5];
    let creation_order = flags & 0x04 != 0;
    let mut p = addr + 6;
    if flags & 0x20 != 0 {
        p += 16;
    }
    if flags & 0x10 != 0 {
        p += 4;
    }
    let size_bytes = 1usize << (flags & 0x03);
    let chunk0 = uint(b, p, size_bytes) as usize;
    p += size_bytes;

    let header_len = 4 + if creation_order { 2 } else { 0 };
    let mut out = Vec::new();
    let mut blocks = vec![(p, p + chunk0)];
    let mut i = 0;
    while i < blocks.len() {
        let (mut p, end) = blocks[i];
        i += 1;
        while p + header_len <= end {
            let typ = b[p] as u16;
            let size = u16(b, p + 1) as usize;
            let data = p + header_len;
            p = data + size;
            match typ {
                0x0000 => {}
                0x0010 => {
                    let ca = uint(b, data, sizes.offset) as usize;
                    let cl = uint(b, data + sizes.offset, sizes.length) as usize;
                    blocks.push((ca + 4, ca + cl - 4));
                }
                _ => out.push(Message {
                    typ,
                    start: data,
                    end: data + size,
                }),
            }
        }
    }
    out
}
