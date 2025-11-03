use super::{E, inverse_langevin, lambert_w, langevin};
use crate::math::test::{TestError, assert_eq, assert_eq_within_tols};

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
    fn zero() -> Result<(), TestError> {
        assert_eq(&inverse_langevin(0.0), &0.0)
    }
}

mod lambert_w {
    use super::*;
    #[test]
    fn end() -> Result<(), TestError> {
        assert_eq(&lambert_w(-1.0 / E), &-1.0)
    }
    #[test]
    fn euler() -> Result<(), TestError> {
        assert_eq(&lambert_w(E), &1.0)
    }
    #[test]
    #[should_panic]
    fn panic() {
        let _ = lambert_w(-10.0);
    }
    #[test]
    fn range() -> Result<(), TestError> {
        let mut x = -1.0 / E;
        let dx = (6.0 - x) / ((LENGTH + 1) as f64);
        let mut w = 0.0;
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            w = lambert_w(x);
            assert_eq_within_tols(&(w * w.exp()), &x)
        })
    }
    #[test]
    fn zero() -> Result<(), TestError> {
        assert_eq(&lambert_w(0.0), &0.0)
    }
}

mod langevin {
    use super::*;
    #[test]
    fn zero() -> Result<(), TestError> {
        assert_eq(&langevin(0.0), &0.0)
    }
}
