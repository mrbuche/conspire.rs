use super::{gamma, gauss_hermite, gauss_laguerre, gauss_legendre};
use std::f64::consts::PI;

fn integrate(nodes: &[f64], weights: &[f64], f: impl Fn(f64) -> f64) -> f64 {
    nodes.iter().zip(weights).map(|(&x, &w)| w * f(x)).sum()
}

mod legendre {
    use super::*;

    #[test]
    fn exact_for_polynomials_up_to_degree_2n_minus_1() {
        for n in [1, 2, 5, 12, 32] {
            let (nodes, weights) = gauss_legendre(n);
            for k in 0..2 * n {
                let got = integrate(&nodes, &weights, |x| x.powi(k as i32));
                let exact = if k % 2 == 1 {
                    0.0
                } else {
                    2.0 / (k as f64 + 1.0)
                };
                assert!(
                    (got - exact).abs() < 1e-11 + 1e-11 * exact.abs(),
                    "n={n} k={k}: {got} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn weights_sum_to_interval_length() {
        let (_, weights) = gauss_legendre(20);
        assert!((weights.iter().sum::<f64>() - 2.0).abs() < 1e-13);
    }

    #[test]
    fn nodes_are_symmetric_and_inside_the_interval() {
        let (nodes, _) = gauss_legendre(15);
        for (&lo, &hi) in nodes.iter().zip(nodes.iter().rev()) {
            assert!((lo + hi).abs() < 1e-12);
        }
        assert!(nodes.iter().all(|&x| x.abs() < 1.0));
    }

    #[test]
    fn transcendental_integral() {
        let (nodes, weights) = gauss_legendre(24);
        let got = integrate(&nodes, &weights, |x| x.exp());
        assert!((got - (1.0_f64.exp() - (-1.0_f64).exp())).abs() < 1e-13);
    }
}

mod laguerre {
    use super::*;

    #[test]
    fn exact_for_polynomials_up_to_degree_2n_minus_1() {
        // Degree capped at 18: beyond it both the weighted sum (terms ~ x_max^k)
        // and the reference gamma lose digits to f64. The construction is fully
        // exercised at moderate degree; the smooth radial-kernel tests below
        // cover the orders the model actually uses.
        for alpha in [0.0, 0.5, 1.0, 2.5] {
            for n in [1, 2, 6, 16, 32] {
                let (nodes, weights) = gauss_laguerre(n, alpha);
                for k in 0..(2 * n).min(19) {
                    let got = integrate(&nodes, &weights, |x| x.powi(k as i32));
                    let exact = gamma(k as f64 + alpha + 1.0);
                    assert!(
                        (got - exact).abs() < 1e-11 * exact.abs(),
                        "alpha={alpha} n={n} k={k}: {got} vs {exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn weights_sum_to_the_zeroth_moment() {
        for alpha in [0.0, 0.5, 1.0] {
            let (_, weights) = gauss_laguerre(24, alpha);
            assert!((weights.iter().sum::<f64>() - gamma(alpha + 1.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn nodes_are_positive() {
        let (nodes, _) = gauss_laguerre(32, 1.0);
        assert!(nodes.iter().all(|&x| x > 0.0));
        assert!(nodes.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn transcendental_integral() {
        // int_0^inf x e^{-x} cos(x) dx = 0
        let (nodes, weights) = gauss_laguerre(48, 1.0);
        let got = integrate(&nodes, &weights, |x| x.cos());
        assert!(got.abs() < 1e-10, "{got}");
    }

    #[test]
    fn radial_kernel_shape() {
        // The Buche-Silberstein radial kernel: with x = w lam^2, generalized
        // Gauss-Laguerre (alpha = 1) evaluates
        //   G(w) = int_0^inf f(lam) lam^3 e^{-w lam^2} dlam
        //        = 1/(2 w^2) sum_i weights_i * f(sqrt(nodes_i / w)).
        // For f = 1 the exact value is 1/(2 w^2).
        let (nodes, weights) = gauss_laguerre(32, 1.0);
        for w in [0.05, 1.0, 40.0] {
            let g = integrate(&nodes, &weights, |_| 1.0) / (2.0 * w * w);
            let exact = 1.0 / (2.0 * w * w);
            assert!(
                (g - exact).abs() < 1e-10 * (1.0 + exact),
                "w={w}: {g} vs {exact}"
            );
        }
    }
}

mod hermite {
    use super::*;

    #[test]
    fn exact_for_polynomials_up_to_degree_2n_minus_1() {
        for n in [1, 2, 5, 12, 32] {
            let (nodes, weights) = gauss_hermite(n);
            // degree capped at 18, see the Laguerre note
            for k in 0..(2 * n).min(19) {
                let got = integrate(&nodes, &weights, |x| x.powi(k as i32));
                let exact = if k % 2 == 1 {
                    0.0
                } else {
                    gamma((k as f64 + 1.0) / 2.0)
                };
                // scale by the natural moment magnitude (odd moments vanish by
                // symmetry but the summed terms are still ~ x_max^k)
                let scale = gamma((k as f64 + 1.0) / 2.0);
                assert!(
                    (got - exact).abs() < 1e-10 * scale,
                    "n={n} k={k}: {got} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn weights_sum_to_sqrt_pi() {
        let (_, weights) = gauss_hermite(20);
        assert!((weights.iter().sum::<f64>() - PI.sqrt()).abs() < 1e-13);
    }

    #[test]
    fn nodes_are_symmetric() {
        let (nodes, _) = gauss_hermite(16);
        for (&lo, &hi) in nodes.iter().zip(nodes.iter().rev()) {
            assert!((lo + hi).abs() < 1e-11);
        }
    }
}

mod gamma_function {
    use super::*;

    #[test]
    fn known_values() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-13);
        assert!((gamma(2.0) - 1.0).abs() < 1e-13);
        assert!((gamma(5.0) - 24.0).abs() < 1e-11);
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-13);
        assert!((gamma(1.5) - PI.sqrt() / 2.0).abs() < 1e-13);
    }
}
