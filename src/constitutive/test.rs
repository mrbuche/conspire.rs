use super::Scalar;

#[test]
fn size() {
    assert_eq!(core::mem::size_of::<&[Scalar]>(), 16)
}
