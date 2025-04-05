use super::{super::test::TestError, OptimizeError};

impl From<OptimizeError> for TestError {
    fn from(error: OptimizeError) -> Self {
        TestError {
            message: format!("{:?}", error),
        }
    }
}

#[test]
fn debug() {
    let _ = format!(
        "{:?}",
        OptimizeError::MaximumStepsReached(1, "foo".to_string())
    );
    let _ = format!(
        "{:?}",
        OptimizeError::NotMinimum("foo".to_string(), "bar".to_string())
    );
}

#[test]
fn display() {
    let _ = format!(
        "{}",
        OptimizeError::MaximumStepsReached(1, "foo".to_string())
    );
    let _ = format!(
        "{}",
        OptimizeError::NotMinimum("foo".to_string(), "bar".to_string())
    );
}

#[test]
fn into_test_error() {
    let optimize_error = OptimizeError::MaximumStepsReached(1, "foo".to_string());
    let _: TestError = optimize_error.into();
}
