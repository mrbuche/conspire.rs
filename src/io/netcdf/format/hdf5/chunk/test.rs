use super::{Filter, Layout, read_data, scatter, unfilter, unshuffle};

#[test]
fn fill_and_contiguous_layouts() {
    assert_eq!(read_data(&[], &Layout::Fill, &[], 8), [0; 8]);
    let file = [0u8, 0, 1, 2, 3, 4, 5, 6, 7, 8];
    let layout = Layout::Contiguous { addr: 2, size: 6 };
    assert_eq!(read_data(&file, &layout, &[], 8), [1, 2, 3, 4, 5, 6, 0, 0]);
}

#[test]
fn scatter_scalar_chunk() {
    let mut out = [0u8; 4];
    scatter(&mut out, &[], &[], 4, &[], &[9, 9, 9, 9, 1, 1]);
    assert_eq!(out, [9, 9, 9, 9]);
}

#[test]
fn scatter_skips_chunk_past_the_edge() {
    let mut out = [7u8; 8];
    scatter(&mut out, &[4], &[2], 1, &[6], &[1, 2]);
    assert_eq!(out, [7; 8]);
}

#[test]
fn unfilter_honours_the_skip_mask() {
    let raw = [1u8, 2, 3, 4];
    assert_eq!(unfilter(&raw, &[Filter::Deflate], 0b1, 4), raw);
}

#[test]
fn unshuffle_inverts_shuffle() {
    let shuffled = [0u8, 4, 1, 5, 2, 6, 3, 7];
    assert_eq!(unshuffle(&shuffled, 4), [0, 1, 2, 3, 4, 5, 6, 7]);
}
