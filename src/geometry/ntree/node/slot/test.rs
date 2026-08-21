use super::Slot;

#[test]
fn a_slot_is_refused_only_when_the_index_cannot_hold_it() {
    assert_eq!(u16::at(65535), Some(65535u16));
    assert_eq!(u16::at(65536), None);
    assert_eq!(u32::at(1 << 31), Some(1u32 << 31));
    assert_eq!(usize::at(usize::MAX), Some(usize::MAX));
}

#[test]
fn a_slot_counts_itself_back() {
    assert_eq!(u32::at(1024).unwrap().slot(), 1024);
}
