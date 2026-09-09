use super::{helmholtz_free_energy, inverse, inverse_derivative};
use crate::math::special::{inverse_langevin, langevin, sinhc};

const KAPPAS: [f64; 5] = [3.0, 5.0, 20.0, 50.0, 500.0];

#[test]
fn inverse_solves_the_relation() {
    for kappa in KAPPAS {
        for i in 1..=250 {
            let y = i as f64 * 0.1;
            let eta = inverse(y, kappa);
            let residual = langevin(eta) + eta / kappa - y;
            assert!(residual.abs() < 1e-9, "kappa={kappa} y={y}: {residual:e}");
        }
    }
}

#[test]
fn inverse_reduces_to_inverse_langevin_for_stiff_links() {
    for i in 1..10 {
        let y = i as f64 * 0.1;
        let got = inverse(y, 1e10);
        let reference = inverse_langevin(y);
        assert!(
            (got - reference).abs() < 1e-5 * reference,
            "y={y}: {got} vs {reference}"
        );
    }
}

#[test]
fn inverse_small_extension_slope() {
    // y = L(eta) + eta/kappa ~ eta (1/3 + 1/kappa)  =>  eta ~ y / (1/3 + 1/kappa)
    for kappa in KAPPAS {
        let y = 1e-4;
        let got = inverse(y, kappa);
        let expected = y / (1.0 / 3.0 + 1.0 / kappa);
        assert!((got / expected - 1.0).abs() < 1e-6, "kappa={kappa}");
    }
}

#[test]
fn inverse_derivative_matches_finite_difference() {
    let h = 1e-6;
    for kappa in KAPPAS {
        for i in 1..40 {
            let y = i as f64 * 0.15;
            let fd = (inverse(y + h, kappa) - inverse(y - h, kappa)) / (2.0 * h);
            let got = inverse_derivative(y, kappa);
            assert!(
                (got - fd).abs() < 1e-6 * (1.0 + got.abs()),
                "kappa={kappa} y={y}"
            );
        }
    }
}

#[test]
fn free_energy_derivative_is_the_force() {
    let h = 1e-6;
    for kappa in KAPPAS {
        for i in 1..40 {
            let y = i as f64 * 0.15;
            let fd = (helmholtz_free_energy(y + h, kappa) - helmholtz_free_energy(y - h, kappa))
                / (2.0 * h);
            let eta = inverse(y, kappa);
            assert!((fd - eta).abs() < 1e-6 * (1.0 + eta), "kappa={kappa} y={y}");
        }
    }
}

#[test]
fn free_energy_vanishes_at_zero_extension() {
    for kappa in KAPPAS {
        assert!(helmholtz_free_energy(0.0, kappa).abs() < 1e-12);
    }
}

#[test]
fn free_energy_reduces_to_freely_jointed_chain_for_stiff_links() {
    for i in 1..9 {
        let y = i as f64 * 0.1;
        let got = helmholtz_free_energy(y, 1e10);
        let eta = inverse_langevin(y);
        let reference = y * eta - sinhc(eta).ln();
        assert!(
            (got - reference).abs() < 1e-4 * (1.0 + reference.abs()),
            "y={y}: {got} vs {reference}"
        );
    }
}
