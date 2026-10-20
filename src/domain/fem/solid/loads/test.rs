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
    fn a_traction_that_depends_on_time_loads_the_forces_of_a_coupled_residual_only() {
        use crate::{
            domain::multiphysics::solid_thermal::{
                NodalForcesSolidThermal, NodalStatesSolidThermal, join_states,
            },
            fem::boundary::{LoadAt, SolidThermal},
        };
        let model = model();
        let loads = Loads::<SolidThermal<3>>::new()
            .traction_with(
                model.coordinates(),
                model.side_sets().iter().next().unwrap(),
                |time| Traction::from([0.0, 0.0, time.value()]),
            )
            .unwrap();
        let states: NodalStatesSolidThermal = join_states(
            &at_rest(&model),
            &crate::fem::diffusion::NodalFields::<crate::units::Power>::zero(8),
        );
        let mut residual = (0..8)
            .map(|_| {
                crate::math::TensorTuple(
                    crate::mechanics::Force::from([0.0; 3]),
                    crate::math::Quantity::<crate::units::Power>::new(0.0),
                )
            })
            .collect::<NodalForcesSolidThermal>();
        loads
            .subtract_at(Time::seconds(3.0), &states, &mut residual)
            .unwrap();
        (0..8).for_each(|node| {
            let force = if node < 4 { 0.0 } else { -3.0 / 4.0 };
            assert!((residual[node].0[2].value() - force).abs() < 1e-12);
            assert_eq!(residual[node].1.value(), 0.0)
        });
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

mod rate {
    use crate::{
        constitutive::{
            canonical::Canonical, fluid::hyperviscous::Newtonian,
            solid::elastic::AlmansiHamelEulerian,
        },
        domain::ElementModel,
        fem::{
            Model,
            block::{Block, element::linear::Hexahedron},
            boundary::{BoundaryConditions, ConstrainAt, Loads, Solid, SolidRate, Velocity},
            solid::{
                elastic_hyperviscous::Minimize as _, loads::DeadLoad, viscoelastic::Root as _,
            },
        },
        geometry::mesh::{Connectivity, Mesh, NodeSet, SideSet},
        math::{integrate::DormandPrince, optimize::NewtonRaphson},
        mechanics::Traction,
        units::{Stress, Time, Viscosity},
    };
    type Hex = Block<Canonical<AlmansiHamelEulerian, Newtonian>, Hexahedron, 8, 3, 8, 8>;
    const PULL: f64 = 1e-2;
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
            Canonical::from((
                AlmansiHamelEulerian {
                    bulk_modulus: Stress::pascals(13.0),
                    shear_modulus: Stress::pascals(3.0),
                },
                Newtonian {
                    bulk_viscosity: Viscosity::pascal_seconds(11.0),
                    shear_viscosity: Viscosity::pascal_seconds(1.0),
                },
            )),
        )
            .try_into()
            .unwrap()
    }
    fn loads(model: &Model<Hex, 3>) -> Loads<Solid<3>, ((), DeadLoad)> {
        Loads::<Solid<3>>::new()
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
    fn the_rate_potential_and_the_forces_of_a_viscous_solid_agree_under_a_load() {
        let model = model();
        let [bottom, pinned, along_x, along_y] = supports();
        let conditions = BoundaryConditions::<SolidRate<3>>::new()
            .fix(Velocity::<2>, &bottom)
            .fix(Velocity::<0>, &pinned)
            .fix(Velocity::<1>, &along_x)
            .fix(Velocity::<0>, &along_y);
        let loads = loads(&model);
        let times = [Time::seconds(0.0), Time::seconds(1.0)];
        let integrator = || DormandPrince {
            abs_tol: 1e-8,
            rel_tol: 1e-8,
            ..Default::default()
        };
        let (_, by_forces, _) = model
            .root(
                &conditions,
                &loads,
                integrator(),
                &times,
                NewtonRaphson::default(),
            )
            .unwrap();
        let (_, by_potential, _) = model
            .minimize(
                &conditions,
                &loads,
                integrator(),
                &times,
                NewtonRaphson::default(),
            )
            .unwrap();
        let (forces, potential) = (
            by_forces.into_iter().last().unwrap(),
            by_potential.into_iter().last().unwrap(),
        );
        (4..8).for_each(|node| {
            let (a, b) = (forces[node][2].value(), potential[node][2].value());
            assert!(a > 1.0, "the pulled face should move out: {a}");
            assert!((a - b).abs() < 1e-6, "node {node}: {a} != {b}");
        });
    }
}

mod body {
    use crate::{
        constitutive::solid::hyperelastic::NeoHookean,
        domain::{ElementModel, solid::loads::BodyForce},
        fem::{
            Model, NodalCoordinates,
            block::{Block, element::linear::Hexahedron},
            boundary::{LoadAt, Loads, Solid},
            solid::NodalForcesSolid,
        },
        geometry::mesh::{Connectivity, Mesh},
        math::Tensor,
        units::{Stress, Time},
    };
    type Hex = Block<NeoHookean, Hexahedron, 8, 3, 8, 8>;
    fn model() -> Model<Hex, 3> {
        let mesh = Mesh::from((
            vec![Connectivity::Hexahedral(
                vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
            )],
            vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [2.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
            ]
            .into(),
        ));
        (
            mesh,
            NeoHookean {
                bulk_modulus: Stress::pascals(13.0),
                shear_modulus: Stress::pascals(3.0),
            },
        )
            .try_into()
            .unwrap()
    }
    fn residual_at(
        loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        model: &Model<Hex, 3>,
        time: f64,
    ) -> Vec<[f64; 3]> {
        let mut residual = NodalForcesSolid::<3>::zero(8);
        loads
            .subtract_at(
                Time::seconds(time),
                &model.coordinates().clone().into(),
                &mut residual,
            )
            .unwrap();
        residual
            .iter()
            .map(|force| [force[0].value(), force[1].value(), force[2].value()])
            .collect()
    }
    fn force(z: f64) -> BodyForce {
        [0.0, 0.0, z].into()
    }
    #[test]
    fn a_uniform_force_gives_each_node_its_share_of_the_volume() {
        let model = model();
        let loads = Loads::<Solid<3>>::new()
            .body_force(model.coordinates(), &model.nodal_volumes(), &force(-10.0))
            .unwrap();
        residual_at(&loads, &model, 0.0)
            .iter()
            .for_each(|force| assert!((force[2] - 10.0 * 6.0 / 8.0).abs() < 1e-12));
    }
    #[test]
    fn a_force_that_depends_on_position_is_integrated_against_the_shape_functions() {
        let model = model();
        let loads = Loads::<Solid<3>>::new()
            .body_force_field(&model, |coordinate| force(coordinate[2].value()))
            .unwrap();
        let residual = residual_at(&loads, &model, 0.0);
        (0..4).for_each(|node| assert!((residual[node][2] + 0.25).abs() < 1e-12));
        (4..8).for_each(|node| assert!((residual[node][2] + 0.5).abs() < 1e-12));
        let total = residual.iter().map(|force| force[2]).sum::<f64>();
        assert!((total + 3.0).abs() < 1e-12);
    }
    #[test]
    fn a_force_that_depends_on_time_takes_the_time_it_is_given() {
        let model = model();
        let loads = Loads::<Solid<3>>::new()
            .body_force_with(model.coordinates(), &model.nodal_volumes(), |time| {
                force(time.value())
            })
            .unwrap();
        (0..8).for_each(|node| {
            assert!((residual_at(&loads, &model, 2.0)[node][2] + 2.0 * 6.0 / 8.0).abs() < 1e-12);
            assert!((residual_at(&loads, &model, 3.0)[node][2] + 3.0 * 6.0 / 8.0).abs() < 1e-12)
        });
    }
    #[test]
    fn a_force_needs_a_volume_for_every_node() {
        let model = model();
        let error = Loads::<Solid<3>>::new()
            .body_force_with(
                model.coordinates(),
                &crate::fem::volume::NodalVolumes::zero(3),
                |_| force(1.0),
            )
            .err()
            .unwrap();
        assert!(error.contains("one weight for each node"));
    }
}

mod follower {
    use crate::{
        constitutive::solid::hyperelastic::NeoHookean,
        domain::ElementModel,
        fem::{
            Model, NodalCoordinates, Root,
            block::{Block, element::linear::Hexahedron},
            boundary::{BoundaryConditions, ConstrainAt as _, Displacement, Loads, Solid},
        },
        geometry::mesh::{Connectivity, Mesh, NodeSet, SideSet},
        math::{
            Quantity,
            optimize::{Direct, NewtonRaphson},
        },
        mechanics::Traction,
        units::Stress,
    };
    type Hex = Block<NeoHookean, Hexahedron, 8, 3, 8, 8>;
    fn model() -> Model<Hex, 3> {
        let mut mesh = Mesh::from((
            vec![Connectivity::Hexahedral(
                vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
            )],
            vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [2.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
            ]
            .into(),
        ));
        mesh.set_side_sets(vec![SideSet::new(vec![(0, 5)])].into());
        (
            mesh,
            NeoHookean {
                bulk_modulus: Stress::pascals(13.0),
                shear_modulus: Stress::pascals(3.0),
            },
        )
            .try_into()
            .unwrap()
    }
    fn conditions<'a>(
        sets: &'a [NodeSet; 4],
    ) -> impl crate::fem::boundary::Constrain<Solid<3>> + crate::fem::boundary::ConstrainAt<Solid<3>> + 'a
    {
        let [bottom, pinned, along_x, along_y] = sets;
        BoundaryConditions::<Solid<3>>::new()
            .fix(Displacement::<2>, bottom)
            .fix(Displacement::<0>, pinned)
            .fix(Displacement::<1>, along_x)
            .fix(Displacement::<0>, along_y)
    }
    fn sets() -> [NodeSet; 4] {
        [
            NodeSet::new(vec![0, 1, 2, 3]),
            NodeSet::new(vec![0]),
            NodeSet::new(vec![1, 0]),
            NodeSet::new(vec![3, 0]),
        ]
    }
    fn shortening(x: &NodalCoordinates<3>) -> f64 {
        1.0 - x[4][2].value()
    }
    fn pressed(pressure: f64) -> NodalCoordinates<3> {
        let model = model();
        let sets = sets();
        let loads = Loads::<Solid<3>>::new()
            .pressure(
                model.side_sets().iter().next().unwrap(),
                Quantity::<Stress>::new(pressure),
            )
            .unwrap();
        model
            .root(
                &conditions(&sets),
                &loads,
                NewtonRaphson {
                    max_steps: 10,
                    linear_solver: Direct,
                    ..Default::default()
                },
            )
            .unwrap()
    }
    fn pulled_by_the_same(pressure: f64) -> NodalCoordinates<3> {
        let model = model();
        let sets = sets();
        let loads = Loads::<Solid<3>>::new()
            .traction(
                model.coordinates(),
                model.side_sets().iter().next().unwrap(),
                &Traction::from([0.0, 0.0, -pressure]),
            )
            .unwrap();
        model
            .root(&conditions(&sets), &loads, NewtonRaphson::default())
            .unwrap()
    }
    #[test]
    fn a_small_pressure_shortens_a_bar_as_the_traction_it_is_the_same_as_does() {
        let (follower, dead) = (
            shortening(&pressed(1e-4)),
            shortening(&pulled_by_the_same(1e-4)),
        );
        assert!(follower > 0.0);
        assert!(
            (follower - dead).abs() < 1e-4 * dead,
            "{follower} != {dead}"
        );
    }
    #[test]
    fn a_large_pressure_shortens_a_bar_more_because_the_face_it_follows_widens() {
        let (follower, dead) = (
            shortening(&pressed(0.8)),
            shortening(&pulled_by_the_same(0.8)),
        );
        assert!(
            follower > 1.005 * dead,
            "{follower} is not more than {dead}"
        );
    }
    mod timed {
        use crate::{
            domain::{
                boundary::{LoadAt, LoadStiffnessAt, Loads, Solid, SolidThermal},
                multiphysics::solid_thermal::{
                    NodalForcesSolidThermal, NodalStiffnessesSolidThermal, join_states,
                },
            },
            fem::{
                NodalCoordinates,
                block::thermal::NodalTemperatures,
                solid::{NodalForcesSolid, NodalStiffnessesSolid},
            },
            geometry::mesh::SideSet,
            math::{
                Quantity, Tensor, TensorArray, TensorRank1, TensorTuple,
                assert::{Assert, AssertionError},
            },
            units::{Power, Stress, Temperature, Time},
        };
        fn side_set() -> SideSet {
            SideSet::new(vec![(0, 5)]).with_faces(vec![vec![4, 5, 6, 7]])
        }
        fn deformed() -> NodalCoordinates<3> {
            [
                [0.0, 0.0, 0.0],
                [2.1, 0.1, 0.0],
                [2.0, 3.2, 0.1],
                [0.1, 3.0, 0.0],
                [0.0, 0.1, 1.1],
                [2.2, 0.0, 0.9],
                [2.0, 3.1, 1.2],
                [0.1, 3.0, 1.0],
            ]
            .into()
        }
        fn temperatures() -> NodalTemperatures {
            (0..8)
                .map(|_| Quantity::<Temperature>::new(300.0))
                .collect()
        }
        #[test]
        fn a_pressure_that_is_a_function_of_time_is_the_pressure_it_gives_at_that_time()
        -> Result<(), AssertionError> {
            let side_set = side_set();
            let x = deformed();
            let time = Quantity::<Time>::new(2.0);
            let fixed = Loads::<Solid<3>>::new()
                .pressure(&side_set, Quantity::<Stress>::new(1.5))
                .unwrap();
            let timed = Loads::<Solid<3>>::new()
                .pressure_with(&side_set, |time| {
                    Quantity::<Stress>::new(0.75) * time.value()
                })
                .unwrap();
            let (mut a, mut b) = (
                NodalForcesSolid::<3>::zero(8),
                NodalForcesSolid::<3>::zero(8),
            );
            fixed.subtract_at(time, &x, &mut a).unwrap();
            timed.subtract_at(time, &x, &mut b).unwrap();
            Assert::default().eq_within_tols(&a, &b)?;
            let (mut k, mut l) = (
                NodalStiffnessesSolid::<3>::zero(8),
                NodalStiffnessesSolid::<3>::zero(8),
            );
            fixed.add_stiffness_at(time, &x, &mut k).unwrap();
            timed.add_stiffness_at(time, &x, &mut l).unwrap();
            Assert::default().eq_within_tols(&k, &l)
        }
        #[test]
        fn a_pressure_with_a_temperature_pushes_as_without_and_adds_no_heat()
        -> Result<(), AssertionError> {
            let side_set = side_set();
            let x = deformed();
            let time = Quantity::<Time>::new(0.0);
            let plain = Loads::<Solid<3>>::new()
                .pressure(&side_set, Quantity::<Stress>::new(1.5))
                .unwrap();
            let coupled = Loads::<SolidThermal<3>>::new()
                .pressure(&side_set, Quantity::<Stress>::new(1.5))
                .unwrap();
            let states = join_states(&x, &temperatures());
            let mut forces = NodalForcesSolid::<3>::zero(8);
            plain.subtract_at(time, &x, &mut forces).unwrap();
            let mut residual: NodalForcesSolidThermal = (0..8)
                .map(|_| TensorTuple(TensorRank1::zero(), Quantity::<Power>::new(0.0)))
                .collect();
            coupled.subtract_at(time, &states, &mut residual).unwrap();
            let mechanical: NodalForcesSolid<3> = residual.iter().map(|r| r.0.clone()).collect();
            Assert::default().eq_within_tols(&mechanical, &forces)?;
            assert!(residual.iter().all(|r| r.1.value() == 0.0));
            let mut stiffnesses = NodalStiffnessesSolid::<3>::zero(8);
            plain.add_stiffness_at(time, &x, &mut stiffnesses).unwrap();
            let mut tangent = NodalStiffnessesSolidThermal::zero(8);
            coupled
                .add_stiffness_at(time, &states, &mut tangent)
                .unwrap();
            stiffnesses.iter().enumerate().try_for_each(|(a, row)| {
                row.entries().try_for_each(|(b, block)| {
                    Assert::default().eq_within_tols(&tangent[a][b].0.0, block)
                })
            })
        }
    }
    mod dynamics {
        use super::*;
        use crate::{
            fem::{NodalReferenceCoordinates, NodalVelocities, solid::ElasticImplicitDynamics},
            math::{integrate::Newmark, optimize::Tolerances},
            units::{Density, Time},
        };
        type Dense = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
        fn dense() -> Model<Dense, 3> {
            let coordinates = NodalReferenceCoordinates::from([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [2.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
            ]);
            (
                Dense::from((
                    NeoHookean {
                        bulk_modulus: Stress::pascals(13.0),
                        shear_modulus: Stress::pascals(3.0),
                    },
                    Density::kilograms_per_cubic_meter(1.0),
                    vec![[0, 1, 2, 3, 4, 5, 6, 7]],
                    &coordinates,
                )),
                coordinates,
            )
                .into()
        }
        #[test]
        fn a_pressure_that_follows_the_face_converges_in_few_steps_of_an_implicit_motion() {
            let model = dense();
            let sets = sets();
            let side_set = SideSet::new(vec![(0, 5)]).with_faces(vec![vec![4, 5, 6, 7]]);
            let loads = Loads::<Solid<3>>::new()
                .pressure(&side_set, Quantity::<Stress>::new(0.8))
                .unwrap();
            let (times, coordinates, ..) = ElasticImplicitDynamics::integrate(
                &model,
                &Newmark::new(Time::seconds(0.05)),
                NewtonRaphson {
                    abs_tol: Tolerances {
                        constraint: 1e-12,
                        residual: 1e-10,
                    },
                    max_steps: 6,
                    linear_solver: Direct,
                    ..Default::default()
                },
                &[Time::seconds(0.0), Time::seconds(0.25)],
                (
                    model.coordinates().clone().into(),
                    NodalVelocities::from(vec![[0.0; 3]; 8]),
                ),
                &loads,
                &model.nodal_lumped_masses(),
                &conditions(&sets),
            )
            .unwrap();
            assert!(times.into_iter().count() > 2);
            assert!(coordinates.into_iter().last().unwrap()[4][2].value() < 1.0);
        }
    }
}
