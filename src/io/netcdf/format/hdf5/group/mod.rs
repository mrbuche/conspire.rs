use super::{FHDB, FHIB, FRHP, Sizes, header::object_header, u16, uint, undefined};

pub(super) fn links(b: &[u8], sizes: &Sizes, group: usize) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut dense_heap = None;
    for m in object_header(b, sizes, group) {
        let d = &b[m.start..m.end];
        match m.typ {
            0x0002 => {
                let mut q = 2;
                if d[1] & 0x01 != 0 {
                    q += 8;
                }
                if !undefined(d, q, sizes.offset) {
                    dense_heap = Some(uint(d, q, sizes.offset) as usize);
                }
            }
            0x0006 => {
                let (name, oh, _) = link_record(b, sizes, m.start);
                out.push((name, oh));
            }
            _ => {}
        }
    }
    if let Some(heap) = dense_heap {
        dense_links(b, sizes, heap, &mut out);
    }
    out
}

fn link_record(b: &[u8], sizes: &Sizes, p: usize) -> (String, usize, usize) {
    assert_eq!(b[p], 1, "unsupported link message version {}", b[p]);
    let flags = b[p + 1];
    let mut q = p + 2;
    if flags & 0x04 != 0 {
        q += 8;
    }
    let name_bytes = 1usize << (flags & 0x03);
    let name_len = uint(b, q, name_bytes) as usize;
    q += name_bytes;
    let name = String::from_utf8_lossy(&b[q..q + name_len]).into_owned();
    q += name_len;
    let oh = uint(b, q, sizes.offset) as usize;
    (name, oh, q + sizes.offset)
}

fn dense_links(b: &[u8], sizes: &Sizes, heap: usize, out: &mut Vec<(String, usize)>) {
    assert_eq!(&b[heap..heap + 4], FRHP, "bad fractal heap signature");
    let io_filter_len = u16(b, heap + 7);
    let flags = b[heap + 9];
    let mut p = heap + 10;
    p += 4;
    p += 8;
    p += sizes.offset;
    p += sizes.length;
    p += sizes.offset;
    p += sizes.length;
    p += sizes.length;
    p += sizes.length;
    let managed = uint(b, p, sizes.length) as usize;
    p += sizes.length;
    p += sizes.length;
    p += sizes.length;
    p += sizes.length;
    p += sizes.length;
    let table_width = u16(b, p) as usize;
    p += 2;
    p += sizes.length;
    p += sizes.length;
    let max_heap_bits = u16(b, p) as usize;
    p += 2;
    p += 2;
    let root = uint(b, p, sizes.offset) as usize;
    p += sizes.offset;
    let root_rows = u16(b, p) as usize;

    assert_eq!(io_filter_len, 0, "filtered fractal heaps are unsupported");
    let checksummed = flags & 0x02 != 0;
    let block_offset_bytes = max_heap_bits.div_ceil(8);

    if root_rows == 0 {
        let block = root;
        heap_direct_block(
            b,
            sizes,
            block,
            block_offset_bytes,
            checksummed,
            managed,
            out,
        );
        return;
    }
    assert_eq!(&b[root..root + 4], FHIB, "bad fractal heap indirect block");
    let mut q = root + 4 + 1 + sizes.offset + block_offset_bytes;
    for _ in 0..root_rows * table_width {
        if out.len() >= managed {
            break;
        }
        if !undefined(b, q, sizes.offset) {
            let block = uint(b, q, sizes.offset) as usize;
            heap_direct_block(
                b,
                sizes,
                block,
                block_offset_bytes,
                checksummed,
                managed,
                out,
            );
        }
        q += sizes.offset;
    }
}

fn heap_direct_block(
    b: &[u8],
    sizes: &Sizes,
    addr: usize,
    block_offset_bytes: usize,
    checksummed: bool,
    managed: usize,
    out: &mut Vec<(String, usize)>,
) {
    assert_eq!(&b[addr..addr + 4], FHDB, "bad fractal heap direct block");
    let mut q = addr + 4 + 1 + sizes.offset + block_offset_bytes;
    if checksummed {
        q += 4;
    }
    while out.len() < managed && b.get(q) == Some(&1) {
        let (name, oh, next) = link_record(b, sizes, q);
        out.push((name, oh));
        q = next;
    }
}
