use std::{
    cell::Cell,
    f64::consts::TAU,
    time::{SystemTime, UNIX_EPOCH},
};

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

pub fn random_u8(max: u8) -> u8 {
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

pub fn random_u64() -> u64 {
    next_u64()
}

pub fn random_uniform() -> f64 {
    let x = next_u64() >> 11;
    (x as f64) * (1.0 / ((1u64 << 53) as f64))
}

thread_local! {
    static NORMAL_SPARE: Cell<Option<f64>> = const { Cell::new(None) };
}

pub fn random_normal_standard() -> f64 {
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

pub fn random_normal(mean: f64, std: f64) -> f64 {
    mean + std * random_normal_standard()
}

// fn random_exp1() -> f64 {
//     let mut u = random_uniform();
//     while u <= 0.0 {
//         u = random_uniform();
//     }
//     -u.ln()
// }

// fn random_gamma_k3_scale1() -> f64 {
//     random_exp1() + random_exp1() + random_exp1()
// }

// pub fn random_x2_normal(mean: f64, std: f64) -> f64 {
//     let m = mean / std;
//     let z_star = if m >= -1.0 { m + 1.0 } else { 0.0 };
//     let h_min = 0.5 * (z_star - m).powi(2) - z_star;
//     loop {
//         let z = random_gamma_k3_scale1();
//         let h = 0.5 * (z - m).powi(2) - z;
//         let acceptance_probability = (-(h - h_min)).exp();
//         if random_uniform() < acceptance_probability {
//             return std * z;
//         }
//     }
// }

use crate::math::special::erf;

fn stretch_cdf_unnormalized(lambda: f64, kappa: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }

    let s = (kappa / 2.0).sqrt();
    let u = s * (lambda - 1.0);
    let u0 = -s;

    let a = (2.0 / kappa).sqrt();
    let b = 2.0 / kappa;

    fn primitive(u: f64, a: f64, b: f64) -> f64 {
        let erf_term = 0.5 * std::f64::consts::PI.sqrt() * (1.0 + 0.5 * b) * erf(u);
        let exp_term = -(a + 0.5 * b * u) * (-u * u).exp();
        erf_term + exp_term
    }

    a * (primitive(u, a, b) - primitive(u0, a, b))
}

fn stretch_pdf_unnormalized(lambda: f64, kappa: f64) -> f64 {
    if lambda <= 0.0 {
        0.0
    } else {
        lambda * lambda * (-0.5 * kappa * (lambda - 1.0).powi(2)).exp()
    }
}

pub fn random_x2_normal(mean: f64, std: f64) -> f64 {
    assert!(mean == 1.0, "current implementation assumes mean=1");
    let kappa = 1.0 / (std * std);

    let z_inf = stretch_cdf_unnormalized(f64::INFINITY, kappa); // replace with closed form limit
    let u = random_uniform();

    let target = u * z_inf;

    // bracket
    let mut lo = 0.0;
    let mut hi = 1.0 + 10.0 * std.max(1.0 / kappa.sqrt());
    while stretch_cdf_unnormalized(hi, kappa) < target {
        hi *= 2.0;
    }

    let mut x = 1.0;
    for _ in 0..20 {
        let fx = stretch_cdf_unnormalized(x, kappa) - target;
        let dfx = stretch_pdf_unnormalized(x, kappa);
        let mut x_new = if dfx > 0.0 {
            x - fx / dfx
        } else {
            0.5 * (lo + hi)
        };

        if !(lo < x_new && x_new < hi && x_new.is_finite()) {
            x_new = 0.5 * (lo + hi);
        }

        if stretch_cdf_unnormalized(x_new, kappa) < target {
            lo = x_new;
        } else {
            hi = x_new;
        }

        x = x_new;
    }

    x
}
