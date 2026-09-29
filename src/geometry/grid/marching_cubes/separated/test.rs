use crate::{
    geometry::{
        Coordinate,
        grid::{Gradient, Isosurface, MarchingCubes, Method, Voxels},
    },
    math::Tensor,
    units::Length,
};
use std::{array::from_fn, collections::HashMap};

pub(crate) fn sample(nel: [usize; 3], mut field: impl FnMut([usize; 3]) -> f64) -> Voxels<f64> {
    let mut data = Vec::with_capacity(nel.iter().product());
    for k in 0..nel[2] {
        for j in 0..nel[1] {
            for i in 0..nel[0] {
                data.push(field([i, j, k]))
            }
        }
    }
    Voxels::new(data, nel)
}

pub(crate) fn sphere(nel: [usize; 3], spacing: [f64; 3], radius: f64) -> Voxels<f64> {
    let centre: [f64; 3] = from_fn(|axis| 0.5 * (nel[axis] - 1) as f64 * spacing[axis]);
    sample(nel, |index| {
        let distance: f64 = (0..3)
            .map(|axis| (index[axis] as f64 * spacing[axis] - centre[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        radius - distance
    })
}

pub(crate) fn extractor(spacing: [f64; 3], method: Method) -> MarchingCubes {
    MarchingCubes {
        level: Some(0.0),
        spacing: Coordinate::from(spacing.map(Length::meters)),
        method,
        ..Default::default()
    }
}

fn points(surface: &Isosurface) -> Vec<[f64; 3]> {
    surface
        .vertices
        .iter()
        .map(|vertex| from_fn(|axis| vertex[axis].value()))
        .collect()
}

fn enclosed(surface: &Isosurface) -> f64 {
    let points = points(surface);
    -surface
        .faces
        .iter()
        .map(|&[a, b, c]| {
            let (p, q, r) = (points[a], points[b], points[c]);
            (p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0])
                + p[2] * (q[0] * r[1] - q[1] * r[0]))
                / 6.0
        })
        .sum::<f64>()
}

fn area(surface: &Isosurface) -> f64 {
    let points = points(surface);
    surface
        .faces
        .iter()
        .map(|&[a, b, c]| {
            let u: [f64; 3] = from_fn(|axis| points[b][axis] - points[a][axis]);
            let v: [f64; 3] = from_fn(|axis| points[c][axis] - points[a][axis]);
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            0.5 * cross
                .iter()
                .map(|component| component * component)
                .sum::<f64>()
                .sqrt()
        })
        .sum()
}

fn closed_and_oriented(surface: &Isosurface) {
    let mut edges = HashMap::<(usize, usize), usize>::new();
    for &[a, b, c] in &surface.faces {
        for edge in [(a, b), (b, c), (c, a)] {
            *edges.entry(edge).or_default() += 1
        }
    }
    for (&(a, b), &count) in &edges {
        assert_eq!(count, 1, "edge {a}->{b} met {count} times");
        assert_eq!(
            edges.get(&(b, a)),
            Some(&1),
            "edge {a}->{b} has no single reverse"
        );
    }
}

#[test]
fn the_surface_of_a_sphere_is_closed_and_oriented_like_lewiner() {
    let spacing = [0.1, 0.1, 0.1];
    let volume = sphere([24, 24, 24], spacing, 1.0);
    let separated = extractor(spacing, Method::Separated)
        .extract(&volume, None)
        .unwrap();
    let lewiner = extractor(spacing, Method::Lewiner)
        .extract(&volume, None)
        .unwrap();
    closed_and_oriented(&separated);
    let exact = 4.0 / 3.0 * std::f64::consts::PI;
    let (enclosed_separated, enclosed_lewiner) = (enclosed(&separated), enclosed(&lewiner));
    assert!(
        (enclosed_separated - exact).abs() < 0.05 * exact,
        "{enclosed_separated} against {exact}"
    );
    assert_eq!(
        enclosed_separated.signum(),
        enclosed_lewiner.signum(),
        "the windings disagree"
    );
    let (points_separated, points_lewiner) = (points(&separated), points(&lewiner));
    let alike = (0..points_separated.len())
        .filter(|&vertex| {
            let nearest = (0..points_lewiner.len())
                .min_by(|&a, &b| {
                    let distance = |other: usize| {
                        (0..3)
                            .map(|axis| {
                                (points_separated[vertex][axis] - points_lewiner[other][axis])
                                    .powi(2)
                            })
                            .sum::<f64>()
                    };
                    distance(a).total_cmp(&distance(b))
                })
                .unwrap();
            let dot: f64 = (0..3)
                .map(|axis| {
                    separated.normals[vertex][axis].value() * lewiner.normals[nearest][axis].value()
                })
                .sum();
            dot > 0.9
        })
        .count();
    assert!(
        alike as f64 > 0.95 * points_separated.len() as f64,
        "{alike} of {} normals point as Lewiner's do",
        points_separated.len()
    );
    let (area_separated, area_lewiner) = (area(&separated), area(&lewiner));
    assert!(
        (area_separated - area_lewiner).abs() < 0.05 * area_lewiner,
        "{area_separated} against {area_lewiner}"
    );
}

#[test]
fn spacing_that_differs_along_each_axis_scales_the_surface() {
    let spacing = [0.08, 0.12, 0.2];
    let volume = sphere([30, 20, 12], spacing, 0.9);
    let surface = extractor(spacing, Method::Separated)
        .extract(&volume, None)
        .unwrap();
    closed_and_oriented(&surface);
    let exact = 4.0 / 3.0 * std::f64::consts::PI * 0.9_f64.powi(3);
    let volume = enclosed(&surface);
    assert!(
        (volume - exact).abs() < 0.08 * exact,
        "{volume} against {exact}"
    );
}

#[test]
fn a_field_of_noise_gives_a_surface_without_tears() {
    let nel = [9, 9, 9];
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let volume = sample(nel, |[i, j, k]| {
        let edge = [i, j, k]
            .iter()
            .any(|&index| index == 0 || index == nel[0] - 1);
        if edge { -1.0 } else { noise() }
    });
    let surface = extractor([1.0; 3], Method::Separated)
        .extract(&volume, None)
        .unwrap();
    closed_and_oriented(&surface);
    assert!(enclosed(&surface) > 0.0);
}

#[test]
fn the_gradient_decides_which_side_is_the_object() {
    let spacing = [0.1; 3];
    let volume = sphere([24, 24, 24], spacing, 1.0);
    let ascent = MarchingCubes {
        gradient: Gradient::Ascent,
        ..extractor(spacing, Method::Separated)
    }
    .extract(&volume, None)
    .unwrap();
    let descent = extractor(spacing, Method::Separated)
        .extract(&volume, None)
        .unwrap();
    closed_and_oriented(&ascent);
    let (ascent, descent) = (enclosed(&ascent), enclosed(&descent));
    assert!(
        (ascent + descent).abs() < 1.0e-9 * descent.abs(),
        "{ascent} {descent}"
    );
}

#[test]
fn stepping_over_samples_is_refused() {
    let volume = sphere([12, 12, 12], [0.2; 3], 0.8);
    let march = MarchingCubes {
        step: 2,
        ..extractor([0.2; 3], Method::Separated)
    };
    assert!(march.extract(&volume, None).is_err());
}

#[test]
fn an_ambiguous_face_is_kept_apart() {
    use super::{CORNERS, cell};
    let mut inside = [false; 8];
    inside[0] = true;
    inside[2] = true;
    let pieces = cell(CORNERS, inside).unwrap();
    assert_eq!(pieces.len(), 2);
    pieces
        .iter()
        .for_each(|piece| assert_eq!(piece.cuts().count(), 1));
}
