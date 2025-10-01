#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![doc = include_str!("../README.md")]

#[cfg(feature = "constitutive")]
pub mod constitutive;

#[cfg(feature = "fem")]
pub mod fem;

#[cfg(feature = "math")]
pub mod math;

#[cfg(feature = "mechanics")]
pub mod mechanics;

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
        // "You, the master of unlocking!"
    }
}

/// Generate a random u8 in range 0..=max using rejection sampling to avoid modulo bias
fn random_u8(max: u8) -> u8 {
    // For perfectly uniform distribution, we need to reject values that would cause modulo bias
    let mut attempts = 0;
    loop {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        // Use multiple entropy sources mixed together
        let nanos = now.subsec_nanos();
        let secs = now.as_secs();
        let addr = &nanos as *const _ as usize;

        // Mix the values with different shifts to spread the entropy
        let mixed = ((nanos as u64) << 32 | (secs as u64)) ^ (addr as u64);
        let mut x = mixed;

        // Quick PRNG shuffle (xorshift)
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;

        // Convert to u8, avoiding modulo bias by rejecting values in the biased range
        let val = x as u8;
        let cutoff = (256_u16 - (256_u16 % (max as u16 + 1))) as u8;

        if val < cutoff {
            return val % (max + 1);
        }

        // Avoid infinite loops in case of system time issues
        attempts += 1;
        if attempts > 100 {
            return (mixed as u8) % (max + 1); // Fallback with some bias
        }
    }
}

#[test]
fn fail() {
    println!("{}", defeat_message());
    assert!(false)
}
