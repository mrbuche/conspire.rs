//! Styling and flavor text for error and result messages.
//!
//! Color is applied only at the formatting boundary, and only when output is
//! headed for a terminal (stderr is a TTY and `NO_COLOR` is unset). When it is
//! not, every token is empty, so error payloads stay free of escape codes when
//! piped, logged, or serialized.

#[cfg(test)]
mod test;

use super::random::random_u8;
use std::{
    env::var_os,
    io::{IsTerminal, stderr},
};

const HEADLINE: &str = "\x1b[1;91m";
const FRAME: &str = "\x1b[0;91m";
const FOOTER: &str = "\x1b[0;2;31m";
const RESET: &str = "\x1b[0m";

/// The styling tokens to splice into an error message.
pub(crate) struct Style {
    /// Prefix for the primary error message (bold red).
    pub headline: &'static str,
    /// Prefix for provenance/context frames (red).
    pub frame: &'static str,
    /// Prefix for the closing flavor message (dim red).
    pub footer: &'static str,
    /// Resets all styling.
    pub reset: &'static str,
}

impl Style {
    /// Returns colored tokens for a terminal, or empty tokens otherwise.
    pub fn detect() -> Self {
        Self::detect_inner(var_os("NO_COLOR").is_none(), stderr().is_terminal())
    }
    fn detect_inner(no_color_unset: bool, is_terminal: bool) -> Self {
        if no_color_unset && is_terminal {
            Self {
                headline: HEADLINE,
                frame: FRAME,
                footer: FOOTER,
                reset: RESET,
            }
        } else {
            Self {
                headline: "",
                frame: "",
                footer: "",
                reset: "",
            }
        }
    }
}

/// An error whose message is built from [`Style`] tokens.
///
/// Implement this, then invoke [`styled_error!`] to derive terminal-aware
/// `Debug` and `Display` from it.
pub(crate) trait StyledError {
    /// Builds the error message, splicing in the given styling tokens.
    fn message(&self, style: &Style) -> String;
}

/// Implements `Debug` and `Display` for a [`StyledError`].
///
/// Color is resolved once per format via [`Style::detect`]. `Debug` (the panic
/// path) appends a flavor footer; `Display` does not.
macro_rules! styled_error {
    ($ty:ty) => {
        impl std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let style = $crate::math::Style::detect();
                write!(
                    f,
                    "\n{}\n{}{}{}\n",
                    $crate::math::StyledError::message(self, &style),
                    style.footer,
                    $crate::math::defeat_message(),
                    style.reset
                )
            }
        }
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let style = $crate::math::Style::detect();
                write!(
                    f,
                    "{}{}",
                    $crate::math::StyledError::message(self, &style),
                    style.reset
                )
            }
        }
    };
}
pub(crate) use styled_error;

pub(crate) fn defeat_message<'a>() -> &'a str {
    defeat_message_inner(random_u8(14))
}

fn defeat_message_inner<'a>(n: u8) -> &'a str {
    match n {
        0 => "Game over.",
        1 => "I am Error.",
        2 => "Insert coin to continue.",
        3 => "Now let's all agree to never be creative again.",
        4 => "Oh dear, you are dead!",
        5 => "Press F to pay respects.",
        6 => "Surprise! You're dead!",
        7 => "Task failed successfully.",
        8 => "This is not your grave, but you are welcome in it.",
        9 => "To be continued...",
        10 => "What a horrible night to have a curse.",
        11 => "You cannot give up just yet.",
        12 => "You have died of dysentery.",
        13 => "You lost the game.",
        14.. => "You've met with a terrible fate, haven't you?",
    }
}
