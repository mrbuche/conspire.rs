#[cfg(test)]
mod test;

use super::{EDGES_X, EDGES_Y, EDGES_Z, lut::Lut};
use std::array::from_fn;

const EPSILON: f64 = f64::EPSILON;

const CORNERS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

pub(super) struct Cell {
    pub(super) v: [f64; 8],
    pub(super) index: usize,
    origin: [usize; 3],
    step: usize,
    vv: [f64; 8],
    vg: [[f64; 3]; 8],
    vmax: f64,
    center: Option<([f64; 3], [f64; 3])>,
    nx: usize,
    layers: [Vec<i32>; 2],
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    values: Vec<f32>,
    faces: Vec<usize>,
}

impl Cell {
    pub(super) fn new(nx: usize, ny: usize) -> Self {
        Self {
            v: [0.0; 8],
            index: 0,
            origin: [0; 3],
            step: 1,
            vv: [0.0; 8],
            vg: [[0.0; 3]; 8],
            vmax: 0.0,
            center: None,
            nx,
            layers: [vec![-1; nx * ny * 4], vec![-1; nx * ny * 4]],
            vertices: Vec::new(),
            normals: Vec::new(),
            values: Vec::new(),
            faces: Vec::new(),
        }
    }
    #[expect(clippy::type_complexity)]
    pub(super) fn finish(self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>, Vec<[f32; 3]>, Vec<f32>) {
        let normals = self
            .normals
            .iter()
            .map(|normal| {
                let mut length = 0.0f64;
                for &component in normal {
                    let component = f64::from(component);
                    length += component * component;
                }
                if length > 0.0 {
                    length = 1.0 / length.powf(0.5);
                }
                normal.map(|component| (f64::from(component) * length) as f32)
            })
            .collect();
        let faces = self.faces.as_chunks::<3>().0.to_vec();
        (self.vertices, faces, normals, self.values)
    }
    pub(super) fn new_z_value(&mut self) {
        self.layers.swap(0, 1);
        self.layers[1].fill(-1);
    }
    pub(super) fn set_cube(
        &mut self,
        isovalue: f64,
        origin: [usize; 3],
        step: usize,
        corners: [f32; 8],
    ) {
        self.origin = origin;
        self.step = step;
        self.v = corners.map(|corner| f64::from(corner) - isovalue);
        self.index = self
            .v
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > 0.0)
            .map(|(i, _)| 1 << i)
            .sum();
        self.center = None;
    }
    pub(super) fn add_triangles<const N: usize>(
        &mut self,
        lut: &Lut<N>,
        index: usize,
        triangles: usize,
    ) {
        self.prepare();
        for i in 0..triangles {
            for j in 0..3 {
                let edge = lut.get2(index, i * 3 + j) as usize;
                self.add_face_from_edge(edge);
            }
        }
    }
    pub(super) fn add_triangles_2<const N: usize>(
        &mut self,
        lut: &Lut<N>,
        index: usize,
        index2: usize,
        triangles: usize,
    ) {
        self.prepare();
        for i in 0..triangles {
            for j in 0..3 {
                let edge = lut.get3(index, index2, i * 3 + j) as usize;
                self.add_face_from_edge(edge);
            }
        }
    }
    fn add_vertex(&mut self, x: f64, y: f64, z: f64) -> usize {
        self.vertices.push([x as f32, y as f32, z as f32]);
        self.normals.push([0.0; 3]);
        self.values.push(0.0);
        self.vertices.len() - 1
    }
    fn add_gradient(&mut self, vertex: usize, gradient: [f32; 3]) {
        (0..3).for_each(|k| self.normals[vertex][k] += gradient[k]);
    }
    fn add_gradient_from_index(&mut self, vertex: usize, i: usize, strength: f32) {
        let gradient = self.vg[i].map(|component| (component * f64::from(strength)) as f32);
        self.add_gradient(vertex, gradient);
    }
    fn add_face(&mut self, index: usize) {
        self.faces.push(index);
        if self.vmax > f64::from(self.values[index]) {
            self.values[index] = self.vmax as f32;
        }
    }
    fn add_face_from_edge(&mut self, edge: usize) {
        let (layer, slot) = self.face_layer_index(edge);
        let existing = self.layers[layer][slot];
        let step = self.step as f64;
        let [x, y, z] = self.origin.map(|coordinate| coordinate as f64);
        if edge == 12 {
            let (position, gradient) = self.center_vertex();
            let vertex = if existing >= 0 {
                existing as usize
            } else {
                let vertex = self.add_vertex(position[0], position[1], position[2]);
                self.layers[layer][slot] = vertex as i32;
                vertex
            };
            self.add_face(vertex);
            self.add_gradient(vertex, gradient.map(|component| component as f32));
        } else {
            let (dx, dy, dz) = (EDGES_X[edge], EDGES_Y[edge], EDGES_Z[edge]);
            let index1 = dz[0] * 4 + dy[0] * 2 + dx[0];
            let index2 = dz[1] * 4 + dy[1] * 2 + dx[1];
            let strength1 = 1.0 / (EPSILON + self.vv[index1].abs());
            let strength2 = 1.0 / (EPSILON + self.vv[index2].abs());
            let vertex = if existing >= 0 {
                existing as usize
            } else {
                let fx = dx[0] as f64 * strength1 + dx[1] as f64 * strength2;
                let fy = dy[0] as f64 * strength1 + dy[1] as f64 * strength2;
                let fz = dz[0] as f64 * strength1 + dz[1] as f64 * strength2;
                let ff = strength1 + strength2;
                let vertex =
                    self.add_vertex(x + step * fx / ff, y + step * fy / ff, z + step * fz / ff);
                self.layers[layer][slot] = vertex as i32;
                vertex
            };
            self.add_face(vertex);
            self.add_gradient_from_index(vertex, index1, strength1 as f32);
            self.add_gradient_from_index(vertex, index2, strength2 as f32);
        }
    }
    fn face_layer_index(&self, mut edge: usize) -> (usize, usize) {
        let [x, y, _] = self.origin;
        let mut i = self.nx * y + x;
        let mut j = 0;
        let layer;
        if edge < 8 {
            if edge < 4 {
                layer = 0;
            } else {
                edge -= 4;
                layer = 1;
            }
            match edge {
                1 => {
                    i += self.step;
                    j = 1;
                }
                2 => i += self.nx * self.step,
                3 => j = 1,
                _ => {}
            }
        } else if edge < 12 {
            layer = 0;
            j = 2;
            match edge {
                9 => i += self.step,
                10 => i += self.nx * self.step + self.step,
                11 => i += self.nx * self.step,
                _ => {}
            }
        } else {
            layer = 0;
            j = 3;
        }
        (layer, 4 * i + j)
    }
    fn prepare(&mut self) {
        let v = self.v;
        self.vv = [v[0], v[1], v[3], v[2], v[4], v[5], v[7], v[6]];
        let (mut vmin, mut vmax) = (0.0f64, 0.0f64);
        for &value in &self.vv {
            if value > vmax {
                vmax = value;
            }
            if value < vmin {
                vmin = value;
            }
        }
        self.vmax = vmax - vmin;
        self.vg = [
            [v[0] - v[1], v[0] - v[3], v[0] - v[4]],
            [v[0] - v[1], v[1] - v[2], v[1] - v[5]],
            [v[3] - v[2], v[1] - v[2], v[2] - v[6]],
            [v[3] - v[2], v[0] - v[3], v[3] - v[7]],
            [v[4] - v[5], v[4] - v[7], v[0] - v[4]],
            [v[4] - v[5], v[5] - v[6], v[1] - v[5]],
            [v[7] - v[6], v[5] - v[6], v[2] - v[6]],
            [v[7] - v[6], v[4] - v[7], v[3] - v[7]],
        ];
    }
    fn center_vertex(&mut self) -> ([f64; 3], [f64; 3]) {
        if let Some(center) = self.center {
            return center;
        }
        let strength: [f64; 8] = self.v.map(|value| 1.0 / (EPSILON + value.abs()));
        let sum = |term: &dyn Fn(usize) -> f64| {
            (0..8)
                .map(term)
                .reduce(|accumulated, next| accumulated + next)
                .unwrap()
        };
        let ff = sum(&|i| strength[i]);
        let step = self.step as f64;
        let position =
            from_fn(|k| self.origin[k] as f64 + step * sum(&|i| CORNERS[i][k] * strength[i]) / ff);
        let gradient = from_fn(|k| sum(&|i| strength[i] * self.vg[i][k]));
        self.center = Some((position, gradient));
        (position, gradient)
    }
}
