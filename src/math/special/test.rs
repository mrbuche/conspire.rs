use super::{inverse_langevin, langevin};
use crate::math::test::{assert_eq_within_tols, TestError};

const LENGTH: usize = 10_000;

mod langevin {
    use super::*;
    #[test]
    fn zero() {
        assert_eq!(langevin(0.0), 0.0)
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
