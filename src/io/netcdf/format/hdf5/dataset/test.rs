use super::{Sizes, attribute, data_layout, dataspace, datatype, fill_value, filter_pipeline};

const SIZES: Sizes = Sizes {
    offset: 8,
    length: 8,
};

#[test]
fn datatype_endianness() {
    let le = datatype(&[0x10, 0x08, 0, 0, 4, 0, 0, 0]);
    assert!(le.little_endian && le.class == 0 && le.size == 4);
    let be = datatype(&[0x10, 0x01, 0, 0, 4, 0, 0, 0]);
    assert!(!be.little_endian);
}

#[test]
fn dataspace_scalar_and_simple() {
    assert_eq!(dataspace(&[2, 0, 0, 0]), Vec::<u64>::new());
    let mut d = vec![2, 1, 0, 0];
    d.extend(7u64.to_le_bytes());
    assert_eq!(dataspace(&d), [7]);
}

#[test]
#[should_panic(expected = "unsupported dataspace version")]
fn dataspace_rejects_v1() {
    dataspace(&[1, 0, 0, 0]);
}

#[test]
#[should_panic(expected = "unsupported data layout message version")]
fn data_layout_rejects_old_version() {
    data_layout(&SIZES, &[2, 1, 0]);
}

#[test]
#[should_panic(expected = "unsupported data layout class")]
fn data_layout_rejects_compact_class() {
    data_layout(&SIZES, &[3, 0, 0, 0]);
}

#[test]
fn data_layout_unallocated_chunked_is_fill() {
    let mut d = vec![3, 2, 2];
    d.extend([0xFF; 8]);
    d.extend([4, 0, 0, 0, 8, 0, 0, 0]);
    assert!(matches!(data_layout(&SIZES, &d), super::RawLayout::Fill));
}

#[test]
fn fill_value_versions() {
    assert_eq!(
        fill_value(&[3, 0x2a, 4, 0, 0, 0, 9, 9, 9, 9]),
        Some(vec![9; 4])
    );
    assert_eq!(fill_value(&[3, 0x00, 0, 0, 0, 0]), None);
    assert_eq!(
        fill_value(&[2, 2, 2, 1, 4, 0, 0, 0, 7, 7, 7, 7]),
        Some(vec![7; 4])
    );
    assert_eq!(fill_value(&[2, 2, 2, 0]), None);
}

#[test]
#[should_panic(expected = "unsupported fill value message version")]
fn fill_value_rejects_unknown_version() {
    fill_value(&[9, 0]);
}

#[test]
#[should_panic(expected = "unsupported filter pipeline version")]
fn filter_pipeline_rejects_v1() {
    filter_pipeline(&[1, 1, 0, 0, 0, 0, 0, 0]);
}

#[test]
#[should_panic(expected = "unsupported HDF5 filter")]
fn filter_pipeline_rejects_unknown_filter() {
    filter_pipeline(&[2, 1, 99, 0, 0, 0, 0, 0]);
}

#[test]
#[should_panic(expected = "unsupported attribute message version")]
fn attribute_rejects_old_version() {
    attribute(&[1, 0, 0, 0, 0, 0, 0, 0]);
}
