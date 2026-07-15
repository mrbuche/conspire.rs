#[cfg(test)]
mod test;

use crate::math::{TensorError, defeat_message};
use std::{
    fmt::{self, Debug, Formatter},
    io::Error as ErrorIO,
};

/// An error produced by a failed assertion, comparing two values.
pub struct AssertionError {
    pub message: String,
}

impl Debug for AssertionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\n\x1b[0;2;31m{}\x1b[0m\n",
            self.message,
            defeat_message()
        )
    }
}

impl From<String> for AssertionError {
    fn from(error: String) -> Self {
        Self { message: error }
    }
}

impl From<&str> for AssertionError {
    fn from(error: &str) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<TensorError> for AssertionError {
    fn from(error: TensorError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<ErrorIO> for AssertionError {
    fn from(error: ErrorIO) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
