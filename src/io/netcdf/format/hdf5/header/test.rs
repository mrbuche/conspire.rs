use super::{Sizes, object_header};

const SIZES: Sizes = Sizes {
    offset: 8,
    length: 8,
};

#[test]
fn skips_optional_time_and_phase_change_fields() {
    let mut b = b"OHDR".to_vec();
    b.push(2);
    b.push(0x30);
    b.extend([0u8; 16]);
    b.extend([0u8; 4]);
    b.push(0);
    assert!(object_header(&b, &SIZES, 0).is_empty());
}
