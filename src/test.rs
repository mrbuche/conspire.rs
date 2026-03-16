mod random_u8 {
    use crate::{random_u8, random_uniform};
    #[test]
    fn u8_max() {
        random_u8(u8::MAX);
    }
    #[test]
    fn one() {
        assert!(random_u8(1) < 2);
        assert!(random_uniform() < 1.0)
    }
    #[test]
    fn zero() {
        assert_eq!(random_u8(0), 0);
        assert!(random_uniform() >= 0.0)
    }
}
