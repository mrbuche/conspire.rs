use super::*;
use crate::math::test::{assert_eq_within_tols, TestError};

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
        let mut gamma = -1.0;
        let length = 9999;
        let d_gamma = 2.0 / ((length + 1) as f64);
        (0..length).try_for_each(|_| {
            gamma += d_gamma;
            assert_eq_within_tols(&langevin(inverse_langevin(gamma)), &gamma)
        })
    }
    #[test]
    fn zero() {
        assert_eq!(inverse_langevin(0.0), 0.0)
    }
}
