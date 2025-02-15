use super::*;
use crate::math::test::{assert_eq_within_tols, TestError};

const LENGTH: usize = 10_000;

mod inverse_langevin {
    use super::*;
    #[test]
    #[should_panic]
    fn above_one() {
        inverse_langevin(1.3);
    }
    #[test]
    #[should_panic]
    fn one() {
        inverse_langevin(1.0);
    }
    #[test]
    fn range() -> Result<(), TestError> {
        let mut x = -1.0;
        let dx = 2.0 / ((LENGTH + 1) as f64);
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            assert_eq_within_tols(&langevin(inverse_langevin(x)), &x)
        })
    }
    #[test]
    fn zero() {
        assert_eq!(inverse_langevin(0.0), 0.0)
    }
}

mod lambert_w {
    use super::*;
    #[test]
    fn one() {
        assert_eq!(lambert_w(1.0_f64.exp()), 1.0)
    }
    #[test]
    fn range() -> Result<(), TestError> {
        let mut x = -(-1.0_f64).exp(); // test other branch later
        let mut w = 0.0;
        let dx = (6.0 - x) / (LENGTH as f64);
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            w = lambert_w(x);
            assert_eq_within_tols(&(w * w.exp()), &x)
        })
    }
    #[test]
    fn zero() {
        assert_eq!(lambert_w(0.0), 0.0)
    }
}
