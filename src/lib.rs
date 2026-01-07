#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![doc = include_str!("../README.md")]

#[cfg(feature = "constitutive")]
pub mod constitutive;

#[cfg(feature = "fem")]
#[path = "domain/fem/mod.rs"]
pub mod fem;

#[cfg(feature = "math")]
pub mod math;

#[cfg(feature = "mechanics")]
pub mod mechanics;

#[cfg(feature = "vem")]
#[path = "domain/vem/mod.rs"]
pub mod vem;

#[cfg(test)]
mod test;

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

/// Absolute tolerance.
pub const ABS_TOL: f64 = 1e-12;

/// Relative tolerance.
pub const REL_TOL: f64 = 1e-12;

#[cfg(test)]
/// A perturbation.
pub const EPSILON: f64 = 1e-6;

#[allow(dead_code)]
#[cfg_attr(coverage_nightly, coverage(off))]
fn defeat_message<'a>() -> &'a str {
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
fn victory_message<'a>() -> &'a str {
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

fn random_u8(max: u8) -> u8 {
    if max == u8::MAX {
        return get_random();
    }
    // let range = (max as u16) + 1;
    // let threshold = ((256_u16 / range) * range) as u8;
    let mut attempts = 0;
    loop {
        let val = get_random();
        // if val < threshold {
        //     return val % (max + 1);
        // }
        attempts += 1;
        if attempts > 10 {
            return val % (max + 1);
        }
    }
}

fn get_random() -> u8 {
    static STATE: AtomicU64 = AtomicU64::new(0);
    let mut s = STATE.load(Ordering::Relaxed);
    if s == 0 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        s = 1 + now.as_nanos() as u64;
    }
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    STATE.store(s, Ordering::Relaxed);
    (s >> 56) as u8
}
