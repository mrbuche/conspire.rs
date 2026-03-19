mod random_u8 {
    use crate::random_u8;
    #[test]
    fn u8_max() {
        random_u8(u8::MAX);
    }
    #[test]
    fn one() {
        assert!(random_u8(1) < 2)
    }
    #[test]
    fn zero() {
        assert_eq!(random_u8(0), 0)
    }
}
