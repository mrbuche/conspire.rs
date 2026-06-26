//! Styling and flavor text for error and result messages.
//!
//! Color is applied only at the formatting boundary, and only when output is
//! headed for a terminal (stderr is a TTY and `NO_COLOR` is unset). When it is
//! not, every token is empty, so error payloads stay free of escape codes when
//! piped, logged, or serialized.

use super::random_u8;
use std::io::IsTerminal;

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
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn detect() -> Self {
        if std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal() {
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

#[allow(dead_code)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) fn defeat_message<'a>() -> &'a str {
    match random_u8(14) {
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

#[allow(dead_code)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) fn victory_message<'a>() -> &'a str {
    match random_u8(7) {
        0 => "A winner is you!",
        1 => "Bird up!",
        2 => "Congraturation, this story is happy end!",
        3 => "Flawless victory.",
        4 => "Hey, that's pretty good!",
        5 => "Nice work, bone daddy.",
        6 => "That's Numberwang!",
        7.. => "That was totes yeet, yo!",
    }
}
