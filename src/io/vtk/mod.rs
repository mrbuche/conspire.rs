pub mod read;
#[cfg(test)]
mod test;
pub mod write;

use std::io::{Error, ErrorKind};

pub fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

pub fn unsupported(message: &str) -> Error {
    Error::new(ErrorKind::Unsupported, message.to_string())
}
