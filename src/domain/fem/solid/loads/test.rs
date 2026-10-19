use super::traction_forces;
use crate::{
    fem::NodalReferenceCoordinates,
    geometry::mesh::SideSet,
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
    mechanics::Force,
};

fn coordinates() -> NodalReferenceCoordinates<3> {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]
    .into()
}

fn side_set(faces: Vec<Vec<usize>>) -> SideSet {
    SideSet::new((0..faces.len()).map(|element| (element, 0)).collect()).with_faces(faces)
}

#[test]
fn a_quadrilateral_gives_a_quarter_of_its_load_to_each_node() -> Result<(), AssertionError> {
    let forces = traction_forces(
        &coordinates(),
        &side_set(vec![vec![0, 1, 4, 3]]),
        &[0.0, 0.0, 8.0].into(),
    )
    .unwrap();
    Assert::default().eq_within_tols(
        &forces,
        &[
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0; 3],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0; 3],
        ]
        .into(),
    )
}

#[test]
fn a_node_shared_by_faces_gets_the_load_of_each() -> Result<(), AssertionError> {
    let forces = traction_forces(
        &coordinates(),
        &side_set(vec![vec![0, 1, 4, 3], vec![1, 2, 5, 4]]),
        &[0.0, 0.0, 8.0].into(),
    )
    .unwrap();
    Assert::default().eq_within_tols(
        &forces,
        &[
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 2.0],
        ]
        .into(),
    )
}

#[test]
fn triangles_and_quadrilaterals_together_carry_the_traction_on_the_area()
-> Result<(), AssertionError> {
    let forces = traction_forces(
        &coordinates(),
        &side_set(vec![vec![0, 1, 4, 3], vec![1, 2, 5], vec![1, 5, 4]]),
        &[1.0, 2.0, 3.0].into(),
    )
    .unwrap();
    let total = forces.iter().cloned().sum::<Force>();
    Assert::default().eq_within_tols(&total, &[2.0, 4.0, 6.0].into())
}

#[test]
fn a_side_set_without_faces_is_an_error() {
    let error = traction_forces(
        &coordinates(),
        &SideSet::new(vec![(0, 0)]),
        &[0.0, 0.0, 1.0].into(),
    )
    .unwrap_err();
    assert!(error.contains("no faces"));
}

#[test]
fn a_face_that_is_not_a_triangle_or_a_quadrilateral_is_an_error() {
    let error = traction_forces(
        &coordinates(),
        &side_set(vec![vec![0, 1, 2, 5, 4]]),
        &[0.0, 0.0, 1.0].into(),
    )
    .unwrap_err();
    assert!(error.contains("5 nodes"));
}

mod patch {
    use crate::{
        constitutive::solid::hyperelastic::NeoHookean,
        domain::ElementModel,
        fem::{
            Minimize, Model, NodalCoordinates, Root,
            block::{Block, element::linear::Hexahedron},
            boundary::{BoundaryConditions, ConstrainAt, Displacement, Loads, Solid},
            solid::{NodalForcesSolid, loads::DeadLoad},
        },
        geometry::mesh::{Connectivity, Mesh, NodeSet, SideSet},
        math::{
            assert::{Assert, AssertionError},
            optimize::NewtonRaphson,
        },
        mechanics::Traction,
        units::{Stress, Time},
    };
    type Hex = Block<NeoHookean, Hexahedron, 8, 3, 8, 8>;
    const PULL: f64 = 1e-3;
    const BULK: f64 = 13.0;
    const SHEAR: f64 = 3.0;
    fn model() -> Model<Hex, 3> {
        let mut mesh = Mesh::from((
            vec![Connectivity::Hexahedral(
                vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
            )],
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
            .into(),
        ));
        mesh.set_side_sets(vec![SideSet::new(vec![(0, 5)])].into());
        (
            mesh,
            NeoHookean {
                bulk_modulus: Stress::pascals(BULK),
                shear_modulus: Stress::pascals(SHEAR),
            },
        )
            .try_into()
            .unwrap()
    }
    fn loads(model: &Model<Hex, 3>) -> Loads<Solid<3>, ((), DeadLoad)> {
        Loads::new()
            .traction(
                model.coordinates(),
                model.side_sets().iter().next().unwrap(),
                &Traction::from([0.0, 0.0, PULL]),
            )
            .unwrap()
    }
    fn supports() -> [NodeSet; 4] {
        [
            NodeSet::new(vec![0, 1, 2, 3]),
            NodeSet::new(vec![0]),
            NodeSet::new(vec![1, 0]),
            NodeSet::new(vec![3, 0]),
        ]
    }
    #[test]
    fn a_pulled_bar_stretches_by_the_traction_over_the_modulus() {
        let model = model();
        let [bottom, pinned, along_x, along_y] = supports();
        let conditions = BoundaryConditions::<Solid<3>>::new()
            .fix(Displacement::<2>, &bottom)
            .fix(Displacement::<0>, &pinned)
            .fix(Displacement::<1>, &along_x)
            .fix(Displacement::<0>, &along_y);
        let loads = loads(&model);
        let by_root = model
            .root(&conditions, &loads, NewtonRaphson::default())
            .unwrap();
        let by_minimize = model
            .minimize(&conditions, &loads, NewtonRaphson::default())
            .unwrap();
        let youngs = 9.0 * BULK * SHEAR / (3.0 * BULK + SHEAR);
        let stretch = PULL / youngs;
        (4..8).for_each(|node| {
            let root = by_root[node][2].value();
            let minimum = by_minimize[node][2].value();
            assert!((root - minimum).abs() < 1e-9);
            assert!((root - 1.0 - stretch).abs() < 1e-3 * stretch);
        });
    }
    fn at_rest(model: &Model<Hex, 3>) -> NodalCoordinates<3> {
        model.coordinates().clone().into()
    }
    fn residual_at(
        loads: &impl crate::fem::boundary::LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        model: &Model<Hex, 3>,
        time: f64,
    ) -> NodalForcesSolid<3> {
        let mut residual = NodalForcesSolid::zero(8);
        loads
            .subtract_at(Time::seconds(time), &at_rest(model), &mut residual)
            .unwrap();
        residual
    }
    #[test]
    fn a_traction_that_depends_on_time_takes_the_time_it_is_given() -> Result<(), AssertionError> {
        let model = model();
        let loads = Loads::<Solid<3>>::new()
            .traction_with(
                model.coordinates(),
                model.side_sets().iter().next().unwrap(),
                |time| Traction::from([0.0, 0.0, time.value()]),
            )
            .unwrap();
        let mut expected = [[0.0; 3]; 8];
        (4..8).for_each(|node| expected[node][2] = -3.0 / 4.0);
        Assert::default().eq_within_tols(
            residual_at(&loads, &model, 3.0),
            NodalForcesSolid::<3>::from(expected),
        )
    }
    #[test]
    fn a_constant_traction_that_depends_on_time_matches_the_static_one()
    -> Result<(), AssertionError> {
        let model = model();
        let side_set = model.side_sets().iter().next().unwrap();
        let timed = Loads::<Solid<3>>::new()
            .traction_with(model.coordinates(), side_set, |_| {
                Traction::from([0.0, 0.0, PULL])
            })
            .unwrap();
        Assert::default().eq_within_tols(
            residual_at(&timed, &model, 7.0),
            residual_at(&loads(&model), &model, 0.0),
        )
    }
    #[test]
    fn forces_on_the_nodes_are_the_same_as_the_traction_they_add_up_to() {
        let model = model();
        let [bottom, pinned, along_x, along_y] = supports();
        let conditions = BoundaryConditions::<Solid<3>>::new()
            .fix(Displacement::<2>, &bottom)
            .fix(Displacement::<0>, &pinned)
            .fix(Displacement::<1>, &along_x)
            .fix(Displacement::<0>, &along_y);
        let mut forces = [[0.0; 3]; 8];
        (4..8).for_each(|node| forces[node][2] = PULL / 4.0);
        let nodal = Loads::<Solid<3>>::new()
            .nodal_forces(model.coordinates(), forces.into())
            .unwrap();
        let by_nodes = model
            .root(&conditions, &nodal, NewtonRaphson::default())
            .unwrap();
        let by_traction = model
            .root(&conditions, &loads(&model), NewtonRaphson::default())
            .unwrap();
        (0..8).for_each(|node| {
            (0..3).for_each(|i| {
                assert!((by_nodes[node][i].value() - by_traction[node][i].value()).abs() < 1e-12)
            })
        });
    }
    #[test]
    fn forces_need_a_node_each() {
        let model = model();
        let error = Loads::<Solid<3>>::new()
            .nodal_forces(model.coordinates(), NodalForcesSolid::<3>::zero(3))
            .err()
            .unwrap();
        assert!(error.contains("one force for each node"));
    }
}
