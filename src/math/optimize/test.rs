use super::{super::test::TestError, OptimizeError};

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
