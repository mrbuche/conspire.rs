use super::Scalar;

#[test]
fn size() {
    assert_eq!(std::mem::size_of::<&[Scalar]>(), 16)
}
