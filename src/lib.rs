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

#[cfg(feature = "physics")]
pub mod physics;

#[cfg(feature = "vem")]
#[path = "domain/vem/mod.rs"]
pub mod vem;

#[cfg(test)]
mod test;

use std::{
    cell::Cell,
    f64::consts::TAU,
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

thread_local! {
    static STATE: Cell<u64> = const { Cell::new(0) };
}

fn seed() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let t = now.as_nanos() as u64;
    let x = 0u8;
    let addr = (&x as *const u8 as usize) as u64;
    let mut s = t ^ addr.wrapping_mul(0x9E3779B97F4A7C15);
    if s == 0 {
        s = 1;
    }
    s
}

fn next_u64() -> u64 {
    STATE.with(|st| {
        let mut s = st.get();
        if s == 0 {
            s = seed();
        }
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        st.set(s);
        s.wrapping_mul(0x2545F4914F6CDD1D)
    })
}

fn get_random() -> u8 {
    (next_u64() >> 56) as u8
}

fn random_u8(max: u8) -> u8 {
    if max == u8::MAX {
        return get_random();
    }
    let bound = (max as u16) + 1;
    let threshold = (256u16 / bound) * bound;
    loop {
        let v = get_random() as u16;
        if v < threshold {
            return (v % bound) as u8;
        }
    }
}

// fn random_u64() -> u64 {
//     next_u64()
// }

fn random_uniform() -> f64 {
    let x = next_u64() >> 11;
    (x as f64) * (1.0 / ((1u64 << 53) as f64))
}

thread_local! {
    static NORMAL_SPARE: Cell<Option<f64>> = const { Cell::new(None) };
}

fn random_normal_standard() -> f64 {
    NORMAL_SPARE.with(|spare| {
        if let Some(z) = spare.take() {
            return z;
        }
        let mut u1 = random_uniform();
        while u1 <= 0.0 {
            u1 = random_uniform();
        }
        let u2 = random_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (TAU * u2).sin_cos();
        let z0 = r * c;
        let z1 = r * s;
        spare.set(Some(z1));
        z0
    })
}

fn random_normal(mean: f64, std: f64) -> f64 {
    mean + std * random_normal_standard()
}

fn random_x2_normal(mean: f64, std: f64, num: f64) -> f64 {
    let x_max = (mean + num * std).max(0.0);
    if x_max == 0.0 {
        return 0.0;
    }
    loop {
        let x = random_normal(mean, std);
        if x < 0.0 || x > x_max {
            continue;
        }
        let u = random_uniform();
        let a = (x / x_max).powi(2);
        if u < a {
            return x;
        }
    }
}

fn random_exp1() -> f64 {
    let mut u = random_uniform();
    while u <= 0.0 {
        u = random_uniform();
    }
    -u.ln()
}

fn random_gamma_k3_scale1() -> f64 {
    random_exp1() + random_exp1() + random_exp1()
}

fn foo(mean: f64, std: f64) -> f64 {
    let m = mean / std;
    let z_star = if m >= -1.0 { m + 1.0 } else { 0.0 };
    let h_min = 0.5 * (z_star - m).powi(2) - z_star;
    loop {
        let z = random_gamma_k3_scale1();
        let h = 0.5 * (z - m).powi(2) - z;
        let acceptance_probability = (-(h - h_min)).exp();
        if random_uniform() < acceptance_probability {
            return std * z;
        }
    }
}
