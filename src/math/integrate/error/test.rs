use super::IntegrationError;

#[test]
fn debug() {
    let _ = format!("{:?}", IntegrationError::InitialTimeNotLessThanFinalTime);
    let _ = format!("{:?}", IntegrationError::LengthTimeLessThanTwo);
}

#[test]
fn display() {
    let _ = format!("{}", IntegrationError::InitialTimeNotLessThanFinalTime);
    let _ = format!("{}", IntegrationError::LengthTimeLessThanTwo);
}
