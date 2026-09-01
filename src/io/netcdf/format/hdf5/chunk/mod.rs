#[cfg(test)]
mod test;

use super::{Filter, Layout, TREE, u16, u32, u64, uint};

#[allow(clippy::too_many_arguments)]
pub(in crate::io::netcdf) fn read_data(
    file: &[u8],
    layout: &Layout,
    filters: &[Filter],
    shape: &[u64],
    start: &[usize],
    count: &[usize],
    elem: usize,
) -> Vec<u8> {
    let out_elems = count.iter().product::<usize>();
    match layout {
        Layout::Fill => vec![0; out_elems * elem],
        Layout::Contiguous { addr, size } => {
            gather(&file[*addr..*addr + size], shape, start, count, elem)
        }
        Layout::Chunked {
            btree_addr,
            offset_size,
            chunk,
        } => {
            let mut out = vec![0; out_elems * elem];
            btree(
                file,
                *btree_addr,
                *offset_size,
                shape,
                start,
                count,
                chunk,
                elem,
                filters,
                &mut out,
            );
            out
        }
    }
}

/// Copy the hyperslab `start .. start + count` out of a row-major blob that
/// holds the whole dataset (`shape`).
fn gather(blob: &[u8], shape: &[u64], start: &[usize], count: &[usize], elem: usize) -> Vec<u8> {
    let rank = shape.len();
    let mut out = vec![0u8; count.iter().product::<usize>() * elem];
    if rank == 0 {
        out.copy_from_slice(&blob[..elem]);
        return out;
    }
    if count.contains(&0) {
        return out;
    }
    let mut stride = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1] as usize;
    }
    let run = count[rank - 1];
    let outer = count[..rank - 1].iter().product::<usize>().max(1);
    for lin in 0..outer {
        let mut rem = lin;
        let mut src = start[rank - 1] * stride[rank - 1];
        for i in (0..rank - 1).rev() {
            let k = rem % count[i];
            rem /= count[i];
            src += (start[i] + k) * stride[i];
        }
        let dst = lin * run;
        out[dst * elem..(dst + run) * elem].copy_from_slice(&blob[src * elem..(src + run) * elem]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn btree(
    file: &[u8],
    addr: usize,
    offset_size: usize,
    shape: &[u64],
    start: &[usize],
    count: &[usize],
    chunk: &[u64],
    elem: usize,
    filters: &[Filter],
    out: &mut [u8],
) {
    assert_eq!(&file[addr..addr + 4], TREE, "bad chunk B-tree node");
    assert_eq!(file[addr + 4], 1, "expected a raw-data chunk B-tree node");
    let level = file[addr + 5];
    let entries = u16(file, addr + 6) as usize;
    let rank = shape.len();
    let key_size = 8 + 8 * (rank + 1);
    let mut p = addr + 8 + 2 * offset_size;
    for _ in 0..entries {
        let chunk_bytes = u32(file, p) as usize;
        let filter_mask = u32(file, p + 4);
        let offsets: Vec<u64> = (0..rank).map(|i| u64(file, p + 8 + 8 * i)).collect();
        p += key_size;
        let child = uint(file, p, offset_size) as usize;
        p += offset_size;
        if level > 0 {
            btree(
                file,
                child,
                offset_size,
                shape,
                start,
                count,
                chunk,
                elem,
                filters,
                out,
            );
        } else if overlaps(&offsets, chunk, start, count) {
            let raw = &file[child..child + chunk_bytes];
            let data = unfilter(raw, filters, filter_mask, elem);
            scatter(out, shape, start, count, chunk, &offsets, elem, &data);
        }
    }
}

fn overlaps(chunk_off: &[u64], chunk: &[u64], start: &[usize], count: &[usize]) -> bool {
    (0..chunk_off.len()).all(|i| {
        let c0 = chunk_off[i] as usize;
        c0 < start[i] + count[i] && start[i] < c0 + chunk[i] as usize
    })
}

fn unfilter(raw: &[u8], filters: &[Filter], mask: u32, elem: usize) -> Vec<u8> {
    let mut data = raw.to_vec();
    for (i, filter) in filters.iter().enumerate().rev() {
        if mask & (1 << i) != 0 {
            continue;
        }
        data = match filter {
            Filter::Deflate => crate::io::zlib_decode(&data).expect("HDF5 deflate chunk inflate"),
            Filter::Shuffle => unshuffle(&data, elem),
        };
    }
    data
}

fn unshuffle(data: &[u8], elem: usize) -> Vec<u8> {
    let n = data.len() / elem;
    let mut out = vec![0; data.len()];
    for j in 0..elem {
        for k in 0..n {
            out[k * elem + j] = data[j * n + k];
        }
    }
    out[n * elem..].copy_from_slice(&data[n * elem..]);
    out
}

/// Copy the part of one decoded chunk (`src`, row-major over `chunk`, at element
/// offset `chunk_off`) that falls inside the requested box (`start .. start +
/// count`) into `out` (row-major over `count`), clipping at the dataset edge.
#[allow(clippy::too_many_arguments)]
fn scatter(
    out: &mut [u8],
    shape: &[u64],
    start: &[usize],
    count: &[usize],
    chunk: &[u64],
    chunk_off: &[u64],
    elem: usize,
    src: &[u8],
) {
    let rank = shape.len();
    if rank == 0 {
        out.copy_from_slice(&src[..elem]);
        return;
    }
    let mut lo = vec![0usize; rank];
    let mut hi = vec![0usize; rank];
    for i in 0..rank {
        lo[i] = (chunk_off[i] as usize).max(start[i]);
        hi[i] = ((chunk_off[i] + chunk[i]) as usize)
            .min(start[i] + count[i])
            .min(shape[i] as usize);
        if lo[i] >= hi[i] {
            return;
        }
    }
    let mut src_stride = vec![1usize; rank];
    let mut dst_stride = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        src_stride[i] = src_stride[i + 1] * chunk[i + 1] as usize;
        dst_stride[i] = dst_stride[i + 1] * count[i + 1];
    }
    let run = hi[rank - 1] - lo[rank - 1];
    let outer = (0..rank - 1)
        .map(|i| hi[i] - lo[i])
        .product::<usize>()
        .max(1);
    for lin in 0..outer {
        let mut rem = lin;
        let mut s = (lo[rank - 1] - chunk_off[rank - 1] as usize) * src_stride[rank - 1];
        let mut d = (lo[rank - 1] - start[rank - 1]) * dst_stride[rank - 1];
        for i in (0..rank - 1).rev() {
            let w = hi[i] - lo[i];
            let k = rem % w;
            rem /= w;
            s += (lo[i] + k - chunk_off[i] as usize) * src_stride[i];
            d += (lo[i] + k - start[i]) * dst_stride[i];
        }
        out[d * elem..(d + run) * elem].copy_from_slice(&src[s * elem..(s + run) * elem]);
    }
}
