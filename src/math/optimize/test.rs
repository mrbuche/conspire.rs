use super::{super::test::TestError, OptimizeError};

impl From<OptimizeError> for TestError {
    fn from(error: OptimizeError) -> Self {
        TestError {
            message: format!("{:?}", error),
        }
    }
}
