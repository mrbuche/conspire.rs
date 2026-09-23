#[cfg(test)]
mod test;

mod cell;
mod lut;
mod switch;
mod tables;

use self::{cell::Cell, tables::*};
use super::Voxels;
use std::array::from_fn;

const EDGES_X: [[usize; 2]; 12] = [
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 0],
    [1, 1],
    [1, 1],
    [0, 0],
];
const EDGES_Y: [[usize; 2]; 12] = [
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 0],
    [1, 1],
    [1, 1],
];
const EDGES_Z: [[usize; 2]; 12] = [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Lewiner,
    Lorensen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gradient {
    Descent,
    Ascent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MarchingCubes {
    pub level: Option<f64>,
    pub spacing: [f64; 3],
    pub gradient: Gradient,
    pub step: usize,
    pub degenerate: bool,
    pub method: Method,
}

impl Default for MarchingCubes {
    fn default() -> Self {
        Self {
            level: None,
            spacing: [1.0; 3],
            gradient: Gradient::Descent,
            step: 1,
            degenerate: true,
            method: Method::Lewiner,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Isosurface {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[usize; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub values: Vec<f32>,
}

impl MarchingCubes {
    pub fn extract(
        &self,
        volume: &Voxels<f32>,
        mask: Option<&Voxels<bool>>,
    ) -> Result<Isosurface, &'static str> {
        let nel = *volume.nel();
        if nel.iter().any(|&n| n < 2) {
            return Err("Input array must be at least 2x2x2.");
        }
        if self.step < 1 {
            return Err("step_size must be at least one.");
        }
        if let Some(mask) = mask
            && mask.nel() != volume.nel()
        {
            return Err("volume and mask must have the same shape.");
        }
        let data = volume.data();
        let minimum = data.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let level = match self.level {
            None => f64::from(0.5 * (minimum + maximum)),
            Some(level) => {
                if level < f64::from(minimum) || level > f64::from(maximum) {
                    return Err("Surface level must be within volume data range.");
                }
                level
            }
        };
        let [nz, ny, nx] = nel;
        let at = |z: usize, y: usize, x: usize| data[volume.flat([z, y, x])];
        let masked =
            |z: usize, y: usize, x: usize| mask.is_none_or(|m| m.data()[m.flat([z, y, x])]);
        let mut cell = Cell::new(nx, ny);
        let step = self.step;
        let classic = self.method == Method::Lorensen;
        let bound = |n: usize| n as isize - 2 * step as isize;
        let (bound_x, bound_y, bound_z) = (bound(nx), bound(ny), bound(nz));
        let stride = step as isize;
        let mut z = -stride;
        while z < bound_z {
            z += stride;
            let (zu, zs) = (z as usize, z as usize + step);
            cell.new_z_value();
            let mut y = -stride;
            while y < bound_y {
                y += stride;
                let (yu, ys) = (y as usize, y as usize + step);
                let mut x = -stride;
                while x < bound_x {
                    x += stride;
                    let (xu, xs) = (x as usize, x as usize + step);
                    if !masked(zs, ys, xs) {
                        continue;
                    }
                    cell.set_cube(
                        level,
                        [xu, yu, zu],
                        step,
                        [
                            at(zu, yu, xu),
                            at(zu, yu, xs),
                            at(zu, ys, xs),
                            at(zu, ys, xu),
                            at(zs, yu, xu),
                            at(zs, yu, xs),
                            at(zs, ys, xs),
                            at(zs, ys, xu),
                        ],
                    );
                    if classic {
                        let mut triangles = 0;
                        while CASESCLASSIC.get2(cell.index, 3 * triangles) != -1 {
                            triangles += 1;
                        }
                        if triangles > 0 {
                            cell.add_triangles(&CASESCLASSIC, cell.index, triangles);
                        }
                    } else {
                        let case = CASES.get2(cell.index, 0);
                        if case > 0 {
                            let config = CASES.get2(cell.index, 1) as usize;
                            switch::the_big_switch(&mut cell, case, config);
                        }
                    }
                }
            }
        }
        let (mut vertices, mut faces, normals, values) = cell.finish();
        if vertices.is_empty() {
            return Err("No surface found at the given iso value.");
        }
        vertices.iter_mut().for_each(|vertex| vertex.reverse());
        let normals: Vec<[f32; 3]> = normals
            .into_iter()
            .map(|mut normal| {
                normal.reverse();
                normal
            })
            .collect();
        if self.gradient == Gradient::Descent {
            faces.iter_mut().for_each(|face| face.reverse());
        }
        let scaled = self.spacing != [1.0; 3];
        let vertices: Vec<[f64; 3]> = vertices
            .iter()
            .map(|vertex| {
                from_fn(|axis| {
                    let coordinate = f64::from(vertex[axis]);
                    if scaled {
                        coordinate * self.spacing[axis]
                    } else {
                        coordinate
                    }
                })
            })
            .collect();
        if self.degenerate {
            Ok(Isosurface {
                vertices,
                faces,
                normals,
                values,
            })
        } else {
            Ok(remove_degenerate_faces(vertices, faces, normals, values))
        }
    }
}

fn remove_degenerate_faces(
    vertices: Vec<[f64; 3]>,
    faces: Vec<[usize; 3]>,
    normals: Vec<[f32; 3]>,
    values: Vec<f32>,
) -> Isosurface {
    let vertices: Vec<[f32; 3]> = vertices
        .iter()
        .map(|vertex| vertex.map(|coordinate| coordinate as f32))
        .collect();
    let mut map: Vec<usize> = (0..vertices.len()).collect();
    let mut keep = vec![true; faces.len()];
    for (j, face) in faces.iter().enumerate() {
        let [i1, i2, i3] = *face;
        for (a, b) in [(i1, i2), (i1, i3), (i2, i3)] {
            if vertices[a] == vertices[b] {
                let lowest = map[a].min(map[b]);
                map[a] = lowest;
                map[b] = lowest;
                keep[j] = false;
            }
        }
    }
    let mut renumber = vec![0usize; vertices.len()];
    let mut count = 0;
    let ok: Vec<bool> = map.iter().enumerate().map(|(i, &m)| m == i).collect();
    for (i, &flag) in ok.iter().enumerate() {
        if flag {
            renumber[i] = count;
            count += 1;
        }
    }
    Isosurface {
        vertices: vertices
            .iter()
            .zip(&ok)
            .filter(|(_, ok)| **ok)
            .map(|(vertex, _)| vertex.map(f64::from))
            .collect(),
        faces: faces
            .iter()
            .zip(&keep)
            .filter(|(_, keep)| **keep)
            .map(|(face, _)| face.map(|index| renumber[map[index]]))
            .collect(),
        normals: normals
            .into_iter()
            .zip(&ok)
            .filter(|(_, ok)| **ok)
            .map(|(normal, _)| normal)
            .collect(),
        values: values
            .into_iter()
            .zip(&ok)
            .filter(|(_, ok)| **ok)
            .map(|(value, _)| value)
            .collect(),
    }
}
