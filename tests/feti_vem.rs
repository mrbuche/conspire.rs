#![cfg(feature = "vem")]

use conspire::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, SecondOrderMinimize, feti::Feti,
        solid::elastic::ElasticElements,
    },
    geometry::{
        Coordinates,
        grid::Voxels,
        mesh::{Connectivities, Connectivity, Mesh, PolytopalConnectivity},
    },
    math::{
        Tensor,
        optimize::{Direct, EqualityConstraint, NewtonRaphson, Tolerances},
    },
    units::Stress,
    vem::block::{Block, element::Element},
};

type VemBlock = Block<NeoHookean, Element>;

/// FETI-DP solves iteratively, so the residual cannot be driven as far as a
/// factorization drives it.
const TOLERANCES: Tolerances = Tolerances {
    constraint: 1e-9,
    residual: 1e-9,
};

fn perturbed(reference: [f64; 3], extent: f64) -> [f64; 3] {
    let [x, y, z] = reference;
    [
        x + 0.02 * y + 0.005 * y * z / extent,
        y + 0.01 * z,
        z + 0.015 * x * x / extent,
    ]
}

/// The six faces of a hexahedron, each listed so that its normal points out.
const FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
];

/// Each hexahedron as a polyhedron with faces of its own, since the faces of a
/// polyhedron are listed once per element with the outward orientation of that
/// element.
fn polyhedra(connectivities: Connectivities) -> PolytopalConnectivity<3> {
    let hexahedra: Vec<[usize; 8]> = connectivities
        .into_members()
        .into_iter()
        .flat_map(|connectivity| match connectivity {
            Connectivity::Hexahedral(hexahedra) => hexahedra.into_iter().collect::<Vec<_>>(),
            _ => panic!("expected a hexahedral mesh"),
        })
        .collect();
    let faces_nodes: Vec<Vec<usize>> = hexahedra
        .iter()
        .flat_map(|hexahedron| {
            FACES
                .iter()
                .map(|face| face.iter().map(|&node| hexahedron[node]).collect())
        })
        .collect();
    let elements_faces: Vec<Vec<usize>> = (0..hexahedra.len())
        .map(|element| (6 * element..6 * element + 6).collect())
        .collect();
    PolytopalConnectivity::from((elements_faces, faces_nodes))
}

fn mesh(nel: [usize; 3]) -> Mesh<3> {
    Mesh::from_voxels(Voxels::new(vec![1u8; nel.iter().product()], nel), None)
}

/// The model of a polyhedral block whose x = 0 face is fixed, started from a
/// smooth non-affine configuration, and the constraint that fixes that face.
fn problem(nel: [usize; 3]) -> (Model<VemBlock, 3>, EqualityConstraint) {
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let reference: Vec<[f64; 3]> = coordinates
        .iter()
        .map(|c| [c[0].value(), c[1].value(), c[2].value()])
        .collect();
    let fixed = reference
        .iter()
        .enumerate()
        .filter(|(_, point)| point[0].abs() < 1e-9)
        .flat_map(|(node, _)| (0..3).map(move |component| 3 * node + component))
        .collect();
    let block = VemBlock::from((
        NeoHookean {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        polyhedra(connectivities),
        &coordinates
            .iter()
            .map(|coordinate| coordinate.clone().with_unit())
            .collect(),
    ));
    let start = NodalReferenceCoordinates::from(
        reference
            .iter()
            .map(|&point| perturbed(point, nel[0] as f64))
            .collect::<Vec<_>>(),
    );
    (
        Model::from((block, start)),
        EqualityConstraint::Fixed(fixed),
    )
}

fn largest_difference(a: &NodalCoordinates<3>, b: &NodalCoordinates<3>) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(a, b)| (0..3).map(move |i| (a[i].value() - b[i].value()).abs()))
        .fold(0.0, f64::max)
}

/// The faces are oriented outward, so the polyhedra have positive volume and
/// the undeformed block is stress free.
#[test]
fn the_polyhedral_block_is_stress_free_at_its_reference_configuration() {
    let nel = [2; 3];
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let block = VemBlock::from((
        NeoHookean {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        polyhedra(connectivities),
        &coordinates
            .iter()
            .map(|coordinate| coordinate.clone().with_unit())
            .collect(),
    ));
    let start = NodalReferenceCoordinates::from(
        coordinates
            .iter()
            .map(|c| [c[0].value(), c[1].value(), c[2].value()])
            .collect::<Vec<_>>(),
    );
    let model = Model::from((block, start.clone()));
    let forces = model
        .nodal_forces(&NodalCoordinates::from(
            start
                .iter()
                .map(|c| [c[0].value(), c[1].value(), c[2].value()])
                .collect::<Vec<_>>(),
        ))
        .unwrap_or_else(|error| panic!("{error}"));
    let largest = forces
        .iter()
        .flat_map(|force| (0..3).map(move |i| force[i].value().abs()))
        .fold(0.0, f64::max);
    assert!(largest < 1e-8, "force {largest:e} at the reference");
}

/// Newton with FETI-DP as its linear solver against Newton on the sparse
/// factorization, on the same polyhedral block from the same start.
#[test]
fn newton_with_feti_matches_newton_with_the_sparse_solve_on_polyhedra() {
    let nel = [6; 3];
    let (model, constraint) = problem(nel);
    let sparse = model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Direct,
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("sparse solve failed: {error}"));
    drop(model);
    let (model, constraint) = problem(nel);
    let decomposed = model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Feti {
                    partition: mesh(nel).partition_box([2; 3]),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("FETI solve failed: {error}"));
    let difference = largest_difference(&sparse, &decomposed);
    assert!(difference < 1e-6, "coordinates differ by {difference:e}");
}
