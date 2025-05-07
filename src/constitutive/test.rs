use super::{ConstitutiveError, Scalar};
use crate::math::test::TestError;
use std::convert::From;

#[test]
fn size() {
    assert_eq!(std::mem::size_of::<&[Scalar]>(), 16)
}

impl From<ConstitutiveError> for TestError {
    fn from(error: ConstitutiveError) -> TestError {
        TestError {
            message: error.to_string(),
        }
    }
}
