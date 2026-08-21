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

#[test]
fn a_nonzero_slot_hides_its_offset() {
    use std::num::NonZeroU32;
    assert_eq!(NonZeroU32::at(0).unwrap().slot(), 0);
    assert_eq!(NonZeroU32::at(4294967294).unwrap().slot(), 4294967294);
    assert_eq!(NonZeroU32::at(4294967295), None);
}

#[test]
fn a_nonzero_slot_leaves_room_in_an_option() {
    use std::num::NonZeroU32;
    assert_eq!(size_of::<Option<NonZeroU32>>(), size_of::<NonZeroU32>());
    assert!(size_of::<Option<NonZeroU32>>() < size_of::<Option<u32>>());
}
