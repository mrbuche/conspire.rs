use super::Cell;

#[test]
fn a_length_is_refused_only_when_the_cell_cannot_hold_it() {
    assert_eq!(u8::length(255), Some(255u8));
    assert_eq!(u8::length(256), None);
    assert_eq!(u16::length(65535), Some(65535u16));
    assert_eq!(u16::length(65536), None);
    assert_eq!(u32::length(65536), Some(65536u32));
    assert_eq!(u32::length(1 << 31), Some(1u32 << 31));
    assert_eq!(u64::length(1 << 40), Some(1u64 << 40));
    assert_eq!(usize::length(usize::MAX), Some(usize::MAX));
}

#[test]
fn halving_a_cell_walks_it_down_to_nothing() {
    assert_eq!(u32::length(8).unwrap().split(), 4);
    assert_eq!(u16::ONE.split(), u16::ZERO);
}

#[test]
fn a_cell_measures_itself_the_same_either_way() {
    let length = u32::length(1024).unwrap();
    assert_eq!(length.cells(), 1024);
    assert_eq!(length.scalar(), 1024.0);
}
