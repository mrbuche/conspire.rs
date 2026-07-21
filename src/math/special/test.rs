use super::{
    E, erf, erfc, inverse_langevin, lambert_w, langevin, rosenbrock, rosenbrock_derivative,
};
use crate::math::assert::Assert;
use crate::math::{Vector, assert::AssertionError};

const LENGTH: usize = 10_000;

mod erf {
    use super::*;
    #[test]
    fn zero() -> Result<(), AssertionError> {
        Assert::eq(erf(0.0), &0.0)
    }
    #[test]
    fn odd_symmetry() -> Result<(), AssertionError> {
        let mut x = 0.0;
        let dx = 10.0 / ((LENGTH + 1) as f64);
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            Assert::default().eq_within_tols(erf(-x), &-erf(x))
        })
    }
    #[test]
    fn saturates_positive() -> Result<(), AssertionError> {
        Assert::eq(erf(1e3), &1.0)
    }
    #[test]
    fn saturates_negative() -> Result<(), AssertionError> {
        Assert::eq(erf(-1e3), &-1.0)
    }
    #[test]
    fn no_overflow_region() -> Result<(), AssertionError> {
        let mut x = -1e3;
        let dx = (1e3 - 27.0) / ((LENGTH + 1) as f64);
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            Assert::eq(erf(x), &-1.0)
        })
    }
}

mod erfc {
    use super::*;
    #[test]
    fn zero() -> Result<(), AssertionError> {
        Assert::eq(erfc(0.0), &1.0)
    }
    #[test]
    fn relation_to_erf() -> Result<(), AssertionError> {
        let mut x = -10.0;
        let dx = 20.0 / ((LENGTH + 1) as f64);
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            Assert::default().eq_within_tols(erfc(x), &(1.0 - erf(x)))
        })
    }
    #[test]
    fn saturates_positive() -> Result<(), AssertionError> {
        Assert::eq(erfc(1e3), &0.0)
    }
    #[test]
    fn saturates_negative() -> Result<(), AssertionError> {
        Assert::eq(erfc(-1e3), &2.0)
    }
    #[test]
    fn no_overflow_region() -> Result<(), AssertionError> {
        let mut x = -1e3;
        let dx = (1e3 - 27.0) / ((LENGTH + 1) as f64);
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            Assert::eq(erfc(x), &2.0)
        })
    }
}

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
    fn range() -> Result<(), AssertionError> {
        let mut x = -1.0;
        let dx = 2.0 / ((LENGTH + 1) as f64);
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            Assert::default().eq_within_tols(langevin(inverse_langevin(x)), &x)
        })
    }
    #[test]
    fn zero() -> Result<(), AssertionError> {
        Assert::zero(&inverse_langevin(0.0))
    }
}

mod lambert_w {
    use super::*;
    #[test]
    fn end() -> Result<(), AssertionError> {
        Assert::eq(lambert_w(-1.0 / E), &-1.0)
    }
    #[test]
    fn euler() -> Result<(), AssertionError> {
        Assert::eq(lambert_w(E), &1.0)
    }
    #[test]
    #[should_panic]
    fn panic() {
        let _ = lambert_w(-10.0);
    }
    #[test]
    fn range() -> Result<(), AssertionError> {
        let mut x = -1.0 / E;
        let dx = (6.0 - x) / ((LENGTH + 1) as f64);
        let mut w = 0.0;
        (0..LENGTH).try_for_each(|_| {
            x += dx;
            w = lambert_w(x);
            Assert::default().eq_within_tols(w * w.exp(), &x)
        })
    }
    #[test]
    fn zero() -> Result<(), AssertionError> {
        Assert::zero(&lambert_w(0.0))
    }
}

mod langevin {
    use super::*;
    #[test]
    fn zero() -> Result<(), AssertionError> {
        Assert::zero(&langevin(0.0))
    }
}

mod rosenbrock {
    use super::*;
    #[test]
    fn zero() -> Result<(), AssertionError> {
        Assert::zero(&rosenbrock(&Vector::from([1.0; 3]), 1.0, 1.0))
    }
    mod derivative {
        use super::*;
        #[test]
        fn zero() -> Result<(), AssertionError> {
            Assert::eq(
                rosenbrock_derivative(&Vector::from([1.0; 3]), 1.0, 1.0),
                &Vector::from([0.0; 3]),
            )
        }
    }
}
