use super::{Gradient, MarchingCubes, Method};
use crate::{
    geometry::{Coordinate, grid::Voxels},
    math::Tensor,
};
use std::{array::from_fn, str::Lines};

fn f32s(line: &str) -> Vec<f32> {
    line.split_whitespace()
        .map(|token| f32::from_bits(u32::from_str_radix(token, 16).unwrap()))
        .collect()
}

fn f64s(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|token| f64::from_bits(u64::from_str_radix(token, 16).unwrap()))
        .collect()
}

fn next<'a>(lines: &mut Lines<'a>) -> &'a str {
    lines.next().unwrap()
}

#[test]
fn matches_scikit_image() {
    let mut lines = include_str!("fixtures.txt").lines();
    let total: usize = next(&mut lines)
        .strip_prefix("cases ")
        .unwrap()
        .parse()
        .unwrap();
    for case in 0..total {
        let header: Vec<&str> = next(&mut lines).split_whitespace().collect();
        let nel: [usize; 3] = from_fn(|axis| header[1 + axis].parse().unwrap());
        let marching = MarchingCubes {
            level: (header[4] != "x").then(|| header[4].parse().unwrap()),
            spacing: Coordinate::const_from(from_fn(|axis| header[5 + axis].parse().unwrap())),
            gradient: if header[8] == "descent" {
                Gradient::Descent
            } else {
                Gradient::Ascent
            },
            step: header[9].parse().unwrap(),
            degenerate: header[10] == "1",
            method: if header[11] == "lewiner" {
                Method::Lewiner
            } else {
                Method::Lorensen
            },
        };
        let volume = Voxels::new_row_major(
            f32s(next(&mut lines)).into_iter().map(f64::from).collect(),
            nel,
        );
        let mask = (header[12] == "1").then(|| {
            let flags = next(&mut lines)
                .split_whitespace()
                .map(|token| token == "1")
                .collect();
            Voxels::new_row_major(flags, nel)
        });
        let counts: Vec<usize> = next(&mut lines)
            .split_whitespace()
            .skip(1)
            .map(|token| token.parse().unwrap())
            .collect();
        let vertices = if header[5..8].iter().all(|&token| token == "1.0") {
            f32s(next(&mut lines)).into_iter().map(f64::from).collect()
        } else {
            f64s(next(&mut lines))
        };
        let faces: Vec<usize> = next(&mut lines)
            .split_whitespace()
            .map(|token| token.parse().unwrap())
            .collect();
        let normals: Vec<f64> = f32s(next(&mut lines)).into_iter().map(f64::from).collect();
        let values: Vec<f64> = f32s(next(&mut lines)).into_iter().map(f64::from).collect();
        let surface = marching.extract(&volume, mask.as_ref()).unwrap();
        assert_eq!(
            surface.vertices.len(),
            counts[0],
            "case {case} vertex count"
        );
        assert_eq!(surface.faces.len(), counts[1], "case {case} face count");
        assert_eq!(surface.faces.concat(), faces, "case {case} faces");
        let close = |name: &str, found: Vec<f64>, expected: &[f64], tolerance: f64| {
            assert_eq!(found.len(), expected.len(), "case {case} {name}");
            found.iter().zip(expected).for_each(|(a, b)| {
                assert!(
                    (a - b).abs() <= tolerance * (1.0 + b.abs()),
                    "case {case} {name}: {a} vs {b}"
                )
            });
        };
        let vertex_values = surface
            .vertices
            .iter()
            .flat_map(|point| (0..3).map(|axis| point[axis].value()))
            .collect();
        let normal_values = surface
            .normals
            .iter()
            .flat_map(|normal| (0..3).map(|axis| normal[axis].value()))
            .collect();
        close("vertices", vertex_values, &vertices, 1e-6);
        close("normals", normal_values, &normals, 1e-5);
        close("values", surface.values, &values, 1e-6);
    }
}

fn ramp(nel: [usize; 3]) -> Voxels<f64> {
    let data = (0..nel.iter().product::<usize>())
        .map(|i| (i % nel[2]) as f64)
        .collect();
    Voxels::new_row_major(data, nel)
}

#[test]
fn rejects_too_small_a_volume() {
    let volume = Voxels::new_row_major(vec![0.0, 1.0], [1, 2, 1]);
    assert_eq!(
        MarchingCubes::default().extract(&volume, None),
        Err("Input array must be at least 2x2x2.")
    );
}

#[test]
fn rejects_zero_step() {
    let marching = MarchingCubes {
        step: 0,
        ..Default::default()
    };
    assert_eq!(
        marching.extract(&ramp([3, 3, 3]), None),
        Err("step_size must be at least one.")
    );
}

#[test]
fn rejects_mismatched_mask() {
    let mask = Voxels::new_row_major(vec![true; 8], [2, 2, 2]);
    assert_eq!(
        MarchingCubes::default().extract(&ramp([3, 3, 3]), Some(&mask)),
        Err("volume and mask must have the same shape.")
    );
}

#[test]
fn rejects_level_outside_the_data() {
    for level in [-0.5, 2.5] {
        let marching = MarchingCubes {
            level: Some(level),
            ..Default::default()
        };
        assert_eq!(
            marching.extract(&ramp([3, 3, 3]), None),
            Err("Surface level must be within volume data range.")
        );
    }
}

#[test]
fn reports_when_no_surface_is_found() {
    let volume = Voxels::new_row_major(vec![1.0; 27], [3, 3, 3]);
    let marching = MarchingCubes {
        level: Some(1.0),
        ..Default::default()
    };
    assert_eq!(
        marching.extract(&volume, None),
        Err("No surface found at the given iso value.")
    );
}
