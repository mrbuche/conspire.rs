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
fn get_defeat_message<'a>() -> &'a str {
    match get_random_number() {
        0 => "Game over.",
        1 => "I am Error.",
        2 => "Oh dear, you are dead!",
        3 => "Press F to pay respects.",
        4 => "Surprise! You're dead!",
        5 => "This is not your grave, but you are welcome in it.",
        6 => "What a horrible night to have a curse.",
        7 => "You cannot give up just yet.",
        8 => "You have died of dysentery.",
        9.. => "You've met with a terrible fate, haven't you?",
        // Now let's all agree to never be creative again.
        // You lost the game.
    }
}

#[allow(dead_code)]
fn get_victory_message<'a>() -> &'a str {
    match get_random_number() {
        0 => "Flawless victory.",
        1 => "Hey, that's pretty good!",
        2 => "Nice work, bone daddy.",
        3 => "That's Numberwang!",
        4.. => "Totes yeet, yo!",
    }
}

fn get_random_number() -> u8 {
    let now = format!("{:?}", std::time::SystemTime::now());
    let length = now.len();
    now[length - 3..length - 2].parse::<u8>().unwrap()
}
