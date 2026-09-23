use super::{Gradient, MarchingCubes, Method};
use crate::geometry::grid::Voxels;
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
            spacing: from_fn(|axis| header[5 + axis].parse().unwrap()),
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
        let volume = Voxels::new_row_major(f32s(next(&mut lines)), nel);
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
        let vertices = if marching.spacing == [1.0; 3] {
            f32s(next(&mut lines)).into_iter().map(f64::from).collect()
        } else {
            f64s(next(&mut lines))
        };
        let faces: Vec<usize> = next(&mut lines)
            .split_whitespace()
            .map(|token| token.parse().unwrap())
            .collect();
        let normals = f32s(next(&mut lines));
        let values = f32s(next(&mut lines));
        let surface = marching.extract(&volume, mask.as_ref()).unwrap();
        assert_eq!(
            surface.vertices.len(),
            counts[0],
            "case {case} vertex count"
        );
        assert_eq!(surface.faces.len(), counts[1], "case {case} face count");
        let bits = |a: &[f64]| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&surface.vertices.concat()),
            bits(&vertices),
            "case {case} vertices"
        );
        assert_eq!(surface.faces.concat(), faces, "case {case} faces");
        let bits = |a: &[f32]| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&surface.normals.concat()),
            bits(&normals),
            "case {case} normals"
        );
        assert_eq!(bits(&surface.values), bits(&values), "case {case} values");
    }
}

fn ramp(nel: [usize; 3]) -> Voxels<f32> {
    let data = (0..nel.iter().product::<usize>())
        .map(|i| (i % nel[2]) as f32)
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
