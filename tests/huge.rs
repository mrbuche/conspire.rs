#![cfg(feature = "fem")]

use conspire::math::assert::Assert;
use conspire::{
    constitutive::solid::{
        elastic::AppliedLoad as AppliedDeformation,
        hyperelastic::{NeoHookean, SecondOrderMinimize as _},
    },
    fem::{
        Model, NodalReferenceCoordinates,
        block::{Block, element::linear::Tetrahedron as LinearTetrahedron, solid::SolidElements},
    },
    geometry::mesh::{Connectivity, Mesh},
    math::{
        Matrix, Scalar, Tensor, Vector, assert::AssertionError, optimize::EqualityConstraint,
        optimize::NewtonRaphson,
    },
};

const ELEMENTS: usize = 28;

const G: usize = 1;
const M: usize = 3;
const N: usize = 4;
const P: usize = 4;

const NODES_PER_EDGE: usize = ELEMENTS + 1;

fn node_index(i: usize, j: usize, k: usize) -> usize {
    i + j * NODES_PER_EDGE + k * NODES_PER_EDGE * NODES_PER_EDGE
}

fn coordinates() -> NodalReferenceCoordinates<3> {
    let mut coordinates = Vec::with_capacity(NODES_PER_EDGE.pow(3));
    for k in 0..NODES_PER_EDGE {
        for j in 0..NODES_PER_EDGE {
            for i in 0..NODES_PER_EDGE {
                coordinates.push([
                    i as Scalar / ELEMENTS as Scalar - 0.5,
                    j as Scalar / ELEMENTS as Scalar - 0.5,
                    k as Scalar / ELEMENTS as Scalar - 0.5,
                ]);
            }
        }
    }
    NodalReferenceCoordinates::from(coordinates)
}

fn connectivity() -> Vec<[usize; N]> {
    let mut connectivity = Vec::with_capacity(ELEMENTS.pow(3) * 6);
    for k in 0..ELEMENTS {
        for j in 0..ELEMENTS {
            for i in 0..ELEMENTS {
                let c000 = node_index(i, j, k);
                let c100 = node_index(i + 1, j, k);
                let c010 = node_index(i, j + 1, k);
                let c001 = node_index(i, j, k + 1);
                let c110 = node_index(i + 1, j + 1, k);
                let c101 = node_index(i + 1, j, k + 1);
                let c011 = node_index(i, j + 1, k + 1);
                let c111 = node_index(i + 1, j + 1, k + 1);
                connectivity.extend([
                    [c000, c100, c110, c111],
                    [c000, c100, c111, c101],
                    [c000, c010, c111, c110],
                    [c000, c010, c011, c111],
                    [c000, c001, c101, c111],
                    [c000, c001, c111, c011],
                ]);
            }
        }
    }
    connectivity
}

fn bcs_nodes_for_x_faces() -> Vec<usize> {
    coordinates()
        .iter()
        .enumerate()
        .filter(|(_, coordinate)| coordinate[0].abs() == 0.5)
        .map(|(node, _)| node)
        .collect()
}

#[test]
fn huge_hyperelastic() -> Result<(), AssertionError> {
    let strain = 13.0;
    let ref_coordinates = coordinates();
    let connectivity = connectivity();
    let num_nodes = ref_coordinates.len();
    let model = NeoHookean {
        bulk_modulus: 13.0,
        shear_modulus: 3.0,
    };
    let x_face_nodes = bcs_nodes_for_x_faces();
    let length = x_face_nodes.len() + 3;
    let width = num_nodes * 3;
    let mut matrix = Matrix::zero(length, width);
    let mut vector = Vector::zero(length);
    x_face_nodes.iter().enumerate().for_each(|(index, &node)| {
        let coordinate = &ref_coordinates[node];
        matrix[index][3 * node] = 1.0;
        if coordinate[0] > 0.0 {
            vector[index] = coordinate[0] + strain
        } else {
            vector[index] = coordinate[0]
        }
    });
    let anchor_1 = node_index(0, 0, 0);
    let anchor_2 = node_index(0, ELEMENTS, 0);
    matrix[length - 3][anchor_1 * 3 + 1] = 1.0;
    matrix[length - 2][anchor_1 * 3 + 2] = 1.0;
    matrix[length - 1][anchor_2 * 3 + 2] = 1.0;
    vector[length - 3] = -0.5;
    vector[length - 2] = -0.5;
    vector[length - 1] = -0.5;
    let mut time = std::time::Instant::now();
    println!("Solving...");
    let mesh = Mesh::from((
        vec![Connectivity::Tetrahedral(connectivity.into())],
        coordinates(),
    ));
    let fem_model: Model<Block<_, LinearTetrahedron, G, M, N, P>, 3> =
        (mesh, model.clone()).try_into()?;
    let solution = conspire::fem::SecondOrderMinimize::minimize(
        &fem_model,
        EqualityConstraint::Linear(matrix, vector),
        NewtonRaphson::default(),
    )?;
    println!("Done ({:?}).", time.elapsed());
    time = std::time::Instant::now();
    println!("Verifying...");
    let deformation_gradient = model.minimize(
        AppliedDeformation::UniaxialStress(strain + 1.0),
        NewtonRaphson::default(),
    )?;
    fem_model
        .blocks()
        .deformation_gradients(&solution)
        .iter()
        .try_for_each(|deformation_gradients_e| {
            deformation_gradients_e
                .iter()
                .try_for_each(|deformation_gradient_g| {
                    Assert::default().eq_within_tols(deformation_gradient_g, &deformation_gradient)
                })
        })?;
    println!("Done ({:?}).", time.elapsed());
    Ok(())
}
