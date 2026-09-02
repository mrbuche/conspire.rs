//! Always-on cross-checks of [`BrepOracle`] against the closed-form
//! [`Primitive`](crate::geometry::csg::Primitive) oracle for the same solid,
//! plus representation-independent signed-distance invariants. Run as part of
//! the periodic `cad` review: a regression here is a real oracle bug, not a
//! flaky tolerance.

use crate::geometry::{
    Coordinate,
    cad::brep::{
        Brep,
        test::{ball, capped_cylinder, cone, hemisphere_solid, torus, unit_cube},
    },
    solid::{Solid, SolidOracle},
};
use std::array::from_fn;

const D: usize = 3;

fn corners(lo: [f64; D], hi: [f64; D], pad_fraction: f64) -> ([f64; D], [f64; D]) {
    let pad: [f64; D] = from_fn(|k| pad_fraction * (hi[k] - lo[k]).abs().max(1.0e-9));
    (from_fn(|k| lo[k] - pad[k]), from_fn(|k| hi[k] + pad[k]))
}

fn grid_point(lo: [f64; D], hi: [f64; D], idx: [usize; D], n: usize) -> Coordinate<D> {
    Coordinate::from(from_fn::<f64, D, _>(|k| {
        lo[k] + (idx[k] as f64 + 0.5) / n as f64 * (hi[k] - lo[k])
    }))
}

fn diagonal(lo: [f64; D], hi: [f64; D]) -> f64 {
    (0..D).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>().sqrt()
}

/// Samples an `n^3` grid over the oracle's bounds padded by 40 %. Outside a
/// `band` of the surface the B-rep and primitive signed distances must agree in
/// sign always and in magnitude to `mag_tol`; both are fractions of the box
/// diagonal.
fn cross_check_primitive(brep: &Brep, n: usize) {
    let primitive = brep.primitive().expect("brep reduces to a primitive");
    let brep_oracle = brep.oracle().expect("brep oracle");
    let primitive_oracle = primitive.oracle().expect("primitive oracle");

    let (low, high) = brep_oracle.bounds();
    let lo: [f64; D] = from_fn(|k| low[k].value());
    let hi: [f64; D] = from_fn(|k| high[k].value());
    let diag = diagonal(lo, hi);
    let band = 1.0e-3 * diag;
    let mag_tol = 3.0e-3 * diag;
    let (lo, hi) = corners(lo, hi, 0.4);

    let mut checked = 0u32;
    let mut sign_fails: Vec<([f64; D], f64, f64)> = Vec::new();
    let mut worst_mag = (0.0_f64, [0.0; D], 0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let query = grid_point(lo, hi, [i, j, k], n);
                let sb = brep_oracle.signed_distance(&query);
                let sp = primitive_oracle.signed_distance(&query);
                if !sb.is_finite() || !sp.is_finite() {
                    continue;
                }
                checked += 1;
                let point = from_fn::<f64, D, _>(|c| query[c].value());
                if sp.abs() > band && sb.signum() != sp.signum() {
                    sign_fails.push((point, sb, sp));
                }
                let mag_err = (sb.abs() - sp.abs()).abs();
                if mag_err > worst_mag.0 {
                    worst_mag = (mag_err, point, sb, sp);
                }
            }
        }
    }

    assert!(
        checked > (n * n * n) as u32 / 4,
        "only {checked} in-bounds samples of {}",
        n * n * n
    );
    assert!(
        sign_fails.is_empty(),
        "{} sign disagreements, e.g. at {:?}: brep {:+.4e} vs primitive {:+.4e}",
        sign_fails.len(),
        sign_fails[0].0,
        sign_fails[0].1,
        sign_fails[0].2
    );
    assert!(
        worst_mag.0 <= mag_tol,
        "magnitude drift {:.3e} > tol {:.3e} (diag {:.3}) at {:?}: brep {:.4e} vs primitive {:.4e}",
        worst_mag.0,
        mag_tol,
        diag,
        worst_mag.1,
        worst_mag.2,
        worst_mag.3
    );
}

/// Where the signed distance is smooth (a central stencil with a negligible
/// second difference, so no medial axis or trim-edge crease inside it) and in
/// the shell `band < |sdf| < 0.15 * diag`, its central-difference gradient must
/// be a unit vector, and every projection foot must sit on the zero level set
/// with an outward normal. Holds for any [`Brep`], primitive or not.
fn check_sdf_invariants(brep: &Brep, n: usize) {
    assert!(
        brep.shells.iter().any(|shell| shell.closed),
        "signed-distance invariants need a closed solid, not an open patch"
    );
    let oracle = brep.oracle().expect("brep oracle");
    let (low, high) = oracle.bounds();
    let lo: [f64; D] = from_fn(|k| low[k].value());
    let hi: [f64; D] = from_fn(|k| high[k].value());
    let diag = diagonal(lo, hi);
    let step = 1.0e-4 * diag;
    let band = 2.0e-3 * diag;
    let shell = 0.15 * diag;
    let kink_tol = 1.0e-6 * diag;
    let (lo, hi) = corners(lo, hi, 0.2);

    let sdf = |p: [f64; D]| oracle.signed_distance(&Coordinate::from(p));
    let mut grad_checks = 0u32;
    let mut worst_grad = (0.0_f64, [0.0; D]);
    let mut worst_zero = (0.0_f64, [0.0; D]);
    let mut worst_normal = (0.0_f64, [0.0; D]);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let query = grid_point(lo, hi, [i, j, k], n);
                let point = from_fn::<f64, D, _>(|c| query[c].value());
                let here = sdf(point);
                if !here.is_finite() || here.abs() <= band || here.abs() >= shell {
                    continue;
                }

                let mut grad = [0.0; D];
                let mut smooth = true;
                for (axis, g) in grad.iter_mut().enumerate() {
                    let mut plus = point;
                    let mut minus = point;
                    plus[axis] += step;
                    minus[axis] -= step;
                    let (sp, sm) = (sdf(plus), sdf(minus));
                    if !sp.is_finite() || !sm.is_finite() || (sp + sm - 2.0 * here).abs() > kink_tol
                    {
                        smooth = false;
                        break;
                    }
                    *g = (sp - sm) / (2.0 * step);
                }
                if !smooth {
                    continue;
                }
                grad_checks += 1;

                let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                let err = (norm - 1.0).abs();
                if err > worst_grad.0 {
                    worst_grad = (err, point);
                }

                if let Some((foot, normal)) = oracle.project(&query) {
                    let foot = from_fn::<f64, D, _>(|c| foot[c].value());
                    let on_surface = sdf(foot).abs();
                    if on_surface > worst_zero.0 {
                        worst_zero = (on_surface, point);
                    }
                    let to_query: [f64; D] = from_fn(|c| point[c] - foot[c]);
                    let dot: f64 = (0..D).map(|c| to_query[c] * normal[c].value()).sum::<f64>();
                    // outward normal: away from the interior, so `dot` and the
                    // signed distance carry opposite signs.
                    let violation = (dot * here).max(0.0) / here.abs();
                    if violation > worst_normal.0 {
                        worst_normal = (violation, point);
                    }
                }
            }
        }
    }

    assert!(
        grad_checks > 20,
        "only {grad_checks} smooth near-surface samples"
    );
    assert!(
        worst_grad.0 < 0.05,
        "|grad sdf| off unit by {:.3e} at {:?}",
        worst_grad.0,
        worst_grad.1
    );
    assert!(
        worst_zero.0 < 1.0e-3 * diag,
        "projection foot off the zero set by {:.3e} at {:?}",
        worst_zero.0,
        worst_zero.1
    );
    assert!(
        worst_normal.0 < 5.0e-2,
        "projection normal not outward, violation {:.3e} at {:?}",
        worst_normal.0,
        worst_normal.1
    );
}

#[test]
fn oracle_matches_primitive_cube() {
    cross_check_primitive(&unit_cube(), 21);
}

#[test]
fn oracle_matches_primitive_capped_cylinder() {
    cross_check_primitive(&capped_cylinder(2.0, 5.0), 21);
}

#[test]
#[ignore = "FINDING cone-distance: conical-face distance is the radial gap, not \
            the perpendicular distance to the slant (over-reports by ~1/cos(semi_angle)); \
            near the top rim it is off by ~0.5. See src/geometry/cad/REVIEW.md."]
fn oracle_matches_primitive_cone() {
    cross_check_primitive(&cone(3.0, 1.0, 4.0), 21);
}

#[test]
fn oracle_matches_primitive_ball() {
    cross_check_primitive(&ball(2.5), 21);
}

#[test]
fn oracle_matches_primitive_torus() {
    cross_check_primitive(&torus(4.0, 1.0), 25);
}

#[test]
fn sdf_invariants_cube() {
    check_sdf_invariants(&unit_cube(), 17);
}

#[test]
fn sdf_invariants_capped_cylinder() {
    check_sdf_invariants(&capped_cylinder(2.0, 5.0), 17);
}

#[test]
#[ignore = "FINDING cone-distance: |grad sdf| ~ 1/cos(semi_angle) at the conical \
            wall, so the wall distance is the radial gap, not the true normal \
            distance. See src/geometry/cad/REVIEW.md."]
fn sdf_invariants_cone() {
    check_sdf_invariants(&cone(3.0, 1.0, 4.0), 17);
}

#[test]
fn sdf_invariants_ball() {
    check_sdf_invariants(&ball(2.5), 17);
}

#[test]
fn sdf_invariants_torus() {
    check_sdf_invariants(&torus(4.0, 1.0), 21);
}

#[test]
fn sdf_invariants_hemisphere() {
    check_sdf_invariants(&hemisphere_solid(2.0), 17);
}
