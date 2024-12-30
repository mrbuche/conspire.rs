use super::{ConstitutiveError, Parameters};
use crate::math::test::TestError;
use std::convert::From;

#[test]
fn size() {
    assert_eq!(std::mem::size_of::<Parameters>(), 16)
}

impl From<ConstitutiveError> for TestError {
    fn from(error: ConstitutiveError) -> TestError {
        TestError {
            message: format!("{}", error),
        }
    }
}
