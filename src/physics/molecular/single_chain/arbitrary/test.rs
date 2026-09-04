use crate::{
    math::{Current, Tensor},
    mechanics::Vector,
    physics::molecular::{
        potential::{Cosine, Harmonic},
        single_chain::{
            ArbitraryDiscrete, ArbitraryDiscretePotential, Ensemble, ExtensibleFreelyRotatingChain,
            FreelyRotatingChain, MonteCarlo,
        },
    },
    units::{BOLTZMANN_CONSTANT, ROOM_TEMPERATURE},
};
use std::f64::consts::PI;

const THETA_B: f64 = 0.4363323129985824;
const T: f64 = ROOM_TEMPERATURE.value();

fn rigid_bend(n: u8, torsion: ArbitraryDiscretePotential<Cosine>) -> ArbitraryDiscrete {
    ArbitraryDiscrete {
        number_of_links: n,
        link_potential: ArbitraryDiscretePotential::Rigid(1.0),
        angular_potential: ArbitraryDiscretePotential::Rigid(THETA_B),
        torsional_potential: torsion,
        ensemble: Ensemble::Isometric(T),
    }
}

fn arr(v: &Vector<Current>) -> [f64; 3] {
    [v[0].value(), v[1].value(), v[2].value()]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    scale(a, 1.0 / dot(a, a).sqrt())
}

// Recover the dihedral phi about link `b1` exactly as frame_propagated_link_vectors
// constructs it, so the check does not depend on any external sign convention.
fn dihedral(b0: [f64; 3], b1: [f64; 3], b2: [f64; 3]) -> f64 {
    let axis = normalize(b1);
    let e1 = normalize(sub(b0, scale(axis, dot(b0, axis))));
    let e2 = cross(axis, e1);
    let perp = sub(b2, scale(axis, dot(b2, axis)));
    dot(perp, e2).atan2(dot(perp, e1))
}

fn mean_cos_dihedral(model: &ArbitraryDiscrete, samples: usize) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for _ in 0..samples {
        let links: Vec<[f64; 3]> = model
            .random_nondimensional_link_vectors(0.0)
            .iter()
            .map(arr)
            .collect();
        for w in links.windows(3) {
            sum += dihedral(w[0], w[1], w[2]).cos();
            count += 1;
        }
    }
    sum / count as f64
}

// ArbitraryDiscrete with a fixed bend angle and free torsion must reproduce the
// freely-rotating chain on the shared longitudinal-extension observable.
#[test]
fn frc_matches_reference() {
    let arbitrary = rigid_bend(8, ArbitraryDiscretePotential::Free);
    let reference = FreelyRotatingChain {
        link_angle: THETA_B,
        link_length: 1.0,
        number_of_links: 8,
        ensemble: Ensemble::Isometric(T),
    };
    for eta in [0.5, 2.0] {
        let a = arbitrary.nondimensional_longitudinal_extension(eta, 200_000, 8);
        let r = reference.nondimensional_longitudinal_extension(eta, 200_000, 8);
        assert!((a - r).abs() < 0.02, "eta={eta}: arbitrary={a}, frc={r}");
    }
}

// Extensible links plus fixed bend and free torsion must reproduce the EFRC.
#[test]
fn efrc_matches_reference() {
    let kappa = 25.0;
    let stiffness = kappa * BOLTZMANN_CONSTANT.value() * T;
    let arbitrary = ArbitraryDiscrete {
        number_of_links: 5,
        link_potential: ArbitraryDiscretePotential::Weak(Harmonic {
            rest_length: 1.0,
            stiffness,
        }),
        angular_potential: ArbitraryDiscretePotential::Rigid(THETA_B),
        torsional_potential: ArbitraryDiscretePotential::Free,
        ensemble: Ensemble::Isometric(T),
    };
    let reference = ExtensibleFreelyRotatingChain {
        link_angle: THETA_B,
        link_length: 1.0,
        link_stiffness: stiffness,
        number_of_links: 5,
        ensemble: Ensemble::Isometric(T),
    };
    for eta in [0.5, 2.0] {
        let a = arbitrary.nondimensional_longitudinal_extension(eta, 200_000, 8);
        let r = reference.nondimensional_longitudinal_extension(eta, 200_000, 8);
        assert!((a - r).abs() < 0.03, "eta={eta}: arbitrary={a}, efrc={r}");
    }
}

// A zero-stiffness cosine torsion is a uniform dihedral, i.e. free torsion.
#[test]
fn torsion_free_limit() {
    let free = rigid_bend(8, ArbitraryDiscretePotential::Free);
    let slack = rigid_bend(
        8,
        ArbitraryDiscretePotential::Weak(Cosine {
            rest_angle: 0.0,
            stiffness: 0.0,
        }),
    );
    let a = free.nondimensional_longitudinal_extension(1.0, 200_000, 8);
    let b = slack.nondimensional_longitudinal_extension(1.0, 200_000, 8);
    assert!((a - b).abs() < 0.02, "free={a}, slack cosine={b}");
}

// A stiff cosine torsion concentrates the dihedral at its rest angle: <cos phi>
// -> cos(phi_0). phi is the proper dihedral built from the preceding bond.
#[test]
fn torsion_tracks_rest_angle() {
    for phi_0 in [0.0, PI, 2.0 * PI / 3.0] {
        let model = rigid_bend(
            8,
            ArbitraryDiscretePotential::Weak(Cosine {
                rest_angle: phi_0,
                stiffness: 20.0 * BOLTZMANN_CONSTANT.value() * T,
            }),
        );
        let c = mean_cos_dihedral(&model, 40_000);
        assert!(
            (c - phi_0.cos()).abs() < 0.1,
            "phi_0={phi_0}: <cos phi>={c}"
        );
    }
}

// phi_0 = 0 is the anti/extended dihedral (the next bond recovers the heading of
// the bond before last); phi_0 = pi curls the chain. Free torsion sits between.
#[test]
fn anti_torsion_extends_chain() {
    let stiffness = 8.0 * BOLTZMANN_CONSTANT.value() * T;
    let anti = rigid_bend(
        8,
        ArbitraryDiscretePotential::Weak(Cosine {
            rest_angle: 0.0,
            stiffness,
        }),
    )
    .nondimensional_longitudinal_extension(1.0, 400_000, 8);
    let free = rigid_bend(8, ArbitraryDiscretePotential::Free)
        .nondimensional_longitudinal_extension(1.0, 400_000, 8);
    let syn = rigid_bend(
        8,
        ArbitraryDiscretePotential::Weak(Cosine {
            rest_angle: PI,
            stiffness,
        }),
    )
    .nondimensional_longitudinal_extension(1.0, 400_000, 8);
    assert!(
        anti > free && free > syn,
        "anti={anti}, free={free}, syn={syn}"
    );
}
