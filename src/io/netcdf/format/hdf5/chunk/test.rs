use super::{Filter, Layout, gather, read_data, scatter, unfilter, unshuffle};

#[test]
fn fill_layout_reads_zeros() {
    assert_eq!(
        read_data(&[], &Layout::Fill, &[], &[2], &[0], &[2], 4),
        [0; 8]
    );
}

#[test]
fn contiguous_full_and_hyperslab() {
    let file = [0u8, 0, 10, 0, 20, 0, 30, 0, 40, 0, 50, 0];
    let layout = Layout::Contiguous { addr: 2, size: 10 };
    assert_eq!(
        read_data(&file, &layout, &[], &[5], &[0], &[5], 2),
        [10, 0, 20, 0, 30, 0, 40, 0, 50, 0]
    );
    assert_eq!(
        read_data(&file, &layout, &[], &[5], &[1], &[3], 2),
        [20, 0, 30, 0, 40, 0]
    );
}

#[test]
fn gather_two_dimensional_window() {
    // 3x4 row-major, 1 byte per element: rows 0,1,2 * cols 0..3.
    let blob: Vec<u8> = (0..12).collect();
    let out = gather(&blob, &[3, 4], &[1, 1], &[2, 2], 1);
    assert_eq!(out, [5, 6, 9, 10]);
}

#[test]
fn gather_scalar() {
    assert_eq!(gather(&[9, 8, 7, 6], &[], &[], &[], 4), [9, 8, 7, 6]);
}

#[test]
fn gather_zero_count() {
    assert_eq!(gather(&[0u8; 8], &[4], &[1], &[0], 2), Vec::<u8>::new());
}

#[test]
fn scatter_scalar_chunk() {
    let mut out = [0u8; 4];
    scatter(&mut out, &[], &[], &[], &[], &[], 4, &[9, 9, 9, 9]);
    assert_eq!(out, [9, 9, 9, 9]);
}

#[test]
fn scatter_chunk_outside_the_box() {
    let mut out = [7u8; 4];
    // dataset shape [8], box [0,2), chunk of size 2 at offset 4: no overlap.
    scatter(&mut out, &[8], &[0], &[2], &[2], &[4], 1, &[1, 2]);
    assert_eq!(out, [7; 4]);
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
