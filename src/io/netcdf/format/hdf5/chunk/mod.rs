#[cfg(test)]
mod test;

use super::{Filter, Layout, TREE, u16, u32, u64, uint};

pub(in crate::io::netcdf) fn read_data(
    file: &[u8],
    layout: &Layout,
    filters: &[Filter],
    want: usize,
) -> Vec<u8> {
    let mut out = match layout {
        Layout::Fill => vec![0; want],
        Layout::Contiguous { addr, size } => file[*addr..*addr + want.min(*size)].to_vec(),
        Layout::Chunked {
            btree_addr,
            offset_size,
            dims,
            chunk,
            elem_size,
        } => {
            let total = dims.iter().product::<u64>().max(1) as usize * elem_size;
            let mut out = vec![0; total];
            btree(
                file,
                *btree_addr,
                *offset_size,
                dims,
                chunk,
                *elem_size,
                filters,
                &mut out,
            );
            out
        }
    };
    out.resize(want, 0);
    out
}

#[allow(clippy::too_many_arguments)]
fn btree(
    file: &[u8],
    addr: usize,
    offset_size: usize,
    dims: &[u64],
    chunk: &[u64],
    elem: usize,
    filters: &[Filter],
    out: &mut [u8],
) {
    assert_eq!(&file[addr..addr + 4], TREE, "bad chunk B-tree node");
    assert_eq!(file[addr + 4], 1, "expected a raw-data chunk B-tree node");
    let level = file[addr + 5];
    let entries = u16(file, addr + 6) as usize;
    let rank = dims.len();
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
            btree(file, child, offset_size, dims, chunk, elem, filters, out);
        } else {
            let raw = &file[child..child + chunk_bytes];
            let data = unfilter(raw, filters, filter_mask, elem);
            scatter(out, dims, chunk, elem, &offsets, &data);
        }
    }
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

fn scatter(out: &mut [u8], dims: &[u64], chunk: &[u64], elem: usize, offsets: &[u64], src: &[u8]) {
    let rank = dims.len();
    if rank == 0 {
        out[..elem].copy_from_slice(&src[..elem]);
        return;
    }
    let ext: Vec<u64> = (0..rank)
        .map(|i| chunk[i].min(dims[i].saturating_sub(offsets[i])))
        .collect();
    if ext.contains(&0) {
        return;
    }
    let mut src_stride = vec![0usize; rank];
    let mut dst_stride = vec![0usize; rank];
    src_stride[rank - 1] = elem;
    dst_stride[rank - 1] = elem;
    for i in (0..rank - 1).rev() {
        src_stride[i] = src_stride[i + 1] * chunk[i + 1] as usize;
        dst_stride[i] = dst_stride[i + 1] * dims[i + 1] as usize;
    }
    let row = ext[rank - 1] as usize * elem;
    let outer = ext[..rank - 1].iter().product::<u64>().max(1);
    for lin in 0..outer {
        let mut rem = lin;
        let mut s = 0;
        let mut d = 0;
        for i in (0..rank - 1).rev() {
            let k = rem % ext[i];
            rem /= ext[i];
            s += k as usize * src_stride[i];
            d += (offsets[i] + k) as usize * dst_stride[i];
        }
        d += offsets[rank - 1] as usize * dst_stride[rank - 1];
        if d + row <= out.len() && s + row <= src.len() {
            out[d..d + row].copy_from_slice(&src[s..s + row]);
        }
    }
}
