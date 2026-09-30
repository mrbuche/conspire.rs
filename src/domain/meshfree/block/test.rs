use super::Block;
use crate::{
    EPSILON,
    constitutive::solid::{elastic::Elastic, hyperelastic::NeoHookean},
    domain::{
        ElementModel, Elements, FirstOrderRoot, NodalCoordinates, NodalReferenceCoordinates,
        block::{finalize_node_neighbors, solver_from_neighbors},
        meshfree::{Discretization, Support},
        solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidElements, elastic::ElasticElements},
    },
    geometry::mesh::{Mesh, test::tetrahedra},
    math::{
        Matrix, Quantity, Tensor, Vector,
        assert::{Assert, AssertionError, perturbation},
        optimize::{EqualityConstraint, FirstOrderRootFinding, NewtonRaphson},
    },
    mechanics::{DeformationGradient, Traction},
    units::{Length, Stress},
};

fn constitutive_model() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(13.0),
        shear_modulus: Stress::pascals(3.0),
    }
}

fn mesh() -> Mesh<3> {
    tetrahedra(8)
}

fn discretization() -> Discretization {
    let support = |spacing, reach| Support {
        spacing: Quantity::new(spacing),
        reach,
    };
    Discretization::new(&mesh(), support(0.4, 2.6), support(0.2, 3.6), 3).unwrap()
}

fn apply(
    deformation_gradient: &DeformationGradient,
    reference: &NodalReferenceCoordinates<3>,
) -> NodalCoordinates<3> {
    reference
        .iter()
        .map(|reference_coordinate| deformation_gradient * reference_coordinate)
        .collect()
}

fn deformation_gradient() -> DeformationGradient {
    DeformationGradient::from([[1.1, 0.05, 0.0], [0.0, 0.9, 0.02], [-0.03, 0.0, 1.2]])
}

fn on_the_boundary(reference: &NodalReferenceCoordinates<3>, node: usize) -> bool {
    (0..3).any(|c| {
        let x = reference[node][c].value();
        x < 1e-12 || x > 1.0 - 1e-12
    })
}

#[test]
fn patch_test_uniform_deformation_gradient() {
    let discretization = discretization();
    let reference = discretization.coordinates().clone();
    let block = Block::from(((), discretization));
    let current = apply(&deformation_gradient(), &reference);
    let gradients = block.deformation_gradients(&current);
    assert!(!gradients.is_empty());
    gradients
        .iter()
        .try_for_each(|gradient| {
            Assert::default().eq_within_tols(gradient, &deformation_gradient())
        })
        .unwrap()
}

#[test]
fn nodal_forces_and_stiffnesses_finite_difference() -> Result<(), AssertionError> {
    let discretization = discretization();
    let reference = discretization.coordinates().clone();
    let block = Block::from((constitutive_model(), discretization));
    let mut coordinates = apply(&deformation_gradient(), &reference);
    coordinates[1] += crate::mechanics::Displacement::from([0.03, -0.02, 0.015]);
    let nodal_stiffnesses = block.nodal_stiffnesses(&coordinates).unwrap();
    let number_of_nodes = reference.len();
    let mut finite_difference = NodalStiffnessesSolid::<3>::zero(number_of_nodes);
    (0..number_of_nodes).for_each(|node_b| {
        (0..3).for_each(|j| {
            let mut perturbed = coordinates.clone();
            perturbed[node_b][j] += perturbation::<Length>(0.5 * EPSILON);
            let forces_plus = block.nodal_forces(&perturbed).unwrap();
            perturbed[node_b][j] -= perturbation::<Length>(EPSILON);
            let forces_minus = block.nodal_forces(&perturbed).unwrap();
            (0..number_of_nodes).for_each(|node_a| {
                (0..3).for_each(|i| {
                    finite_difference[node_a][node_b][i][j] = (forces_plus[node_a][i]
                        - forces_minus[node_a][i])
                        / perturbation::<Length>(EPSILON);
                })
            })
        })
    });
    Assert::default().eq_within_fd_tol(&nodal_stiffnesses, &finite_difference)
}

// The one seed off the boundary has a basis function that is nonzero on the
// constrained boundary, so the exact affine field is not an equilibrium of it
// without tractions, and the solution is only close to that field. With the
// tractions, it is exact, as the test below shows.
#[test]
fn solve_approximates_an_affine_field_from_its_boundary_values() -> Result<(), AssertionError> {
    let discretization = discretization();
    let reference = discretization.coordinates().clone();
    let number_of_nodes = reference.len();
    let boundary: Vec<usize> = (0..number_of_nodes)
        .filter(|&node| on_the_boundary(&reference, node))
        .collect();
    assert!(boundary.len() < number_of_nodes, "no interior seed");
    let mut a = Matrix::zero(3 * boundary.len(), 3 * number_of_nodes);
    let mut b = Vector::zero(3 * boundary.len());
    for (index, &node) in boundary.iter().enumerate() {
        let target = &deformation_gradient() * &reference[node];
        for c in 0..3 {
            a[3 * index + c][3 * node + c] = 1.0;
            b[3 * index + c] = target[c].value();
        }
    }
    let block = Block::from((constitutive_model(), discretization.clone()));
    let model = discretization.model(constitutive_model());
    let coordinates = FirstOrderRoot::root(
        &model,
        EqualityConstraint::Linear(a, b),
        NewtonRaphson::default(),
    )?;
    let expected = apply(&deformation_gradient(), &reference);
    boundary.iter().try_for_each(|&node| {
        Assert::default().eq_within_tols(&coordinates[node], &expected[node])
    })?;
    let largest = |errors: Vec<f64>| errors.into_iter().fold(0.0, f64::max);
    let position_error = largest(
        (0..number_of_nodes)
            .flat_map(|node| (0..3).map(move |c| (node, c)))
            .map(|(node, c)| (coordinates[node][c] - expected[node][c]).value().abs())
            .collect(),
    );
    let gradient_error = largest(
        block
            .deformation_gradients(&coordinates)
            .iter()
            .flat_map(|gradient| (0..3).flat_map(move |i| (0..3).map(move |j| (gradient, i, j))))
            .map(|(gradient, i, j)| {
                (gradient[i][j] - deformation_gradient()[i][j])
                    .value()
                    .abs()
            })
            .collect(),
    );
    assert!(position_error < 0.05, "position error {position_error}");
    assert!(gradient_error < 0.05, "gradient error {gradient_error}");
    Ok(())
}

#[test]
fn solve_recovers_an_affine_field_exactly_with_tractions() -> Result<(), AssertionError> {
    let mesh = mesh();
    let discretization = discretization();
    let reference = discretization.coordinates().clone();
    let number_of_nodes = reference.len();
    let boundary: Vec<usize> = (0..number_of_nodes)
        .filter(|&node| on_the_boundary(&reference, node))
        .collect();
    assert!(boundary.len() < number_of_nodes, "no interior seed");
    let mut a = Matrix::zero(3 * boundary.len(), 3 * number_of_nodes);
    let mut b = Vector::zero(3 * boundary.len());
    for (index, &node) in boundary.iter().enumerate() {
        let target = &deformation_gradient() * &reference[node];
        for c in 0..3 {
            a[3 * index + c][3 * node + c] = 1.0;
            b[3 * index + c] = target[c].value();
        }
    }
    let stress = constitutive_model()
        .first_piola_kirchhoff_stress(&deformation_gradient())
        .unwrap();
    let mut external = NodalForcesSolid::<3>::zero(number_of_nodes);
    for axis in 0..3 {
        for (side, sign) in [(0.0, -1.0), (1.0, 1.0)] {
            let faces: Vec<Vec<usize>> = mesh
                .exterior_faces()
                .into_iter()
                .filter(|face| {
                    face.iter()
                        .all(|&node| (mesh.coordinates()[node][axis].value() - side).abs() < 1e-12)
                })
                .collect();
            let traction = Traction::from(std::array::from_fn(|i| sign * stress[i][axis].value()));
            external += &discretization.traction(&mesh, &faces, &traction).unwrap();
        }
    }
    let block = Block::from((constitutive_model(), discretization.clone()));
    let model = discretization.model(constitutive_model());
    let constraint = EqualityConstraint::Linear(a, b);
    let mut neighbors = vec![Vec::new(); number_of_nodes];
    model.node_neighbors(&mut neighbors);
    finalize_node_neighbors(&mut neighbors);
    let sparse = solver_from_neighbors(&neighbors, &constraint, 3, false);
    let coordinates = NewtonRaphson::default().root(
        |x: &NodalCoordinates<3>| Ok(model.nodal_forces(x)? - &external),
        |x: &NodalCoordinates<3>| Ok(model.nodal_stiffnesses(x)?),
        model.coordinates().clone().into(),
        constraint,
        Some(sparse),
    )?;
    let expected = apply(&deformation_gradient(), &reference);
    (0..number_of_nodes).try_for_each(|node| {
        Assert::default().eq_within_tols(&coordinates[node], &expected[node])
    })?;
    block
        .deformation_gradients(&coordinates)
        .iter()
        .try_for_each(|gradient| {
            Assert::default().eq_within_tols(gradient, &deformation_gradient())
        })
}
