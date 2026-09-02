use super::jenkins;

#[test]
fn jenkins_reference_vectors() {
    assert_eq!(jenkins(b""), 0xdead_beef);
    assert_eq!(jenkins(b"Four score and seven years ago"), 0x1777_0551);
}
