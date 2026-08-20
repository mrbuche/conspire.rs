use crate::{
    geometry::{
        Coordinate,
        primitive::{
            Cylinder, Solid, Union,
            test::{Spread, assert_length, thicket, upright},
        },
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};

/// A cylinder along the x-axis long enough to run clear through [`upright`],
/// so that the two bury one another's surfaces where they cross.
fn spanning() -> Cylinder {
    Cylinder::new(
        [
            Coordinate::const_from([-2.0, 0.0, 1.0]),
            Coordinate::const_from([2.0, 0.0, 1.0]),
        ],
        Length::meters(1.0),
    )
}

fn crossed() -> Union<Cylinder> {
    Union::new(vec![upright(), spanning()])
}

#[test]
fn outside_takes_the_nearest_member() {
    // Beside the upright cylinder by two, but only one past the spanning cap.
    assert_length(
        crossed().signed_distance(&Coordinate::const_from([3.0, 0.0, 1.0])),
        1.0,
    )
}

#[test]
fn inside_either_member_is_inside_the_union() {
    assert_length(
        crossed().signed_distance(&Coordinate::const_from([1.5, 0.0, 1.0])),
        -0.5,
    )
}

#[test]
fn outside_both_members_is_outside_the_union() {
    assert_length(
        crossed().signed_distance(&Coordinate::const_from([0.0, 3.0, 1.0])),
        2.0,
    )
}

/// Uniting is order-independent, being a minimum over the members.
#[test]
fn order_does_not_matter() {
    let point = Coordinate::const_from([0.7, 0.4, 1.3]);
    assert_eq!(
        crossed().signed_distance(&point),
        Union::new(vec![spanning(), upright()]).signed_distance(&point)
    )
}

#[test]
fn closest_point_outside_takes_the_nearest_member() {
    let (point, normal) = crossed().closest_point(&Coordinate::const_from([3.0, 0.0, 1.0]));
    assert_eq!(point, Coordinate::const_from([2.0, 0.0, 1.0]));
    assert_eq!(normal, [1.0, 0.0, 0.0].into())
}

/// At the heart of the crossing the upright cylinder's nearest surface lies
/// buried a full radius inside the spanning one, so it is no longer boundary
/// and the spanning cylinder's exposed surface has to be taken instead.
#[test]
fn closest_point_rejects_a_buried_candidate() {
    let center = Coordinate::const_from([0.0, 0.0, 1.0]);
    let buried = upright().closest_point(&center).0;
    assert_length(spanning().signed_distance(&buried), -1.0);
    let (point, _) = crossed().closest_point(&center);
    assert_length(crossed().signed_distance(&point), 0.0);
    assert_ne!(point, buried)
}

/// Deep in the crossing the two members bury each other's nearest surface by
/// the same `1 - sqrt(3)/2`, leaving no exposed candidate to choose between,
/// so the answer has to come from projecting rather than from choosing.
#[test]
fn closest_point_projects_where_every_candidate_is_buried() {
    let crossed = crossed();
    let point = Coordinate::const_from([0.5, 0.5, 1.5]);
    crossed.solids().iter().for_each(|solid| {
        let candidate = solid.closest_point(&point).0;
        assert_length(
            crossed.signed_distance(&candidate),
            3.0_f64.sqrt() / 2.0 - 1.0,
        )
    });
    let (closest, _) = crossed.closest_point(&point);
    assert_length(crossed.signed_distance(&closest), 0.0)
}

/// Every closest point the union reports has to land on its surface, however
/// the members overlap there.
#[test]
fn closest_points_land_on_the_surface() {
    let crossed = crossed();
    [
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 1.5],
        [3.0, 0.0, 1.0],
        [0.0, 0.0, 5.0],
        [1.0, 1.0, 0.0],
        [-1.5, 0.2, 0.9],
    ]
    .into_iter()
    .for_each(|point| {
        let (closest, _) = crossed.closest_point(&Coordinate::const_from(point));
        assert_length(crossed.signed_distance(&closest), 0.0)
    })
}

#[test]
fn bounding_box_covers_every_member() {
    let extent = crossed().bounding_box();
    assert_eq!(extent.minimum(), &[-2.0, -1.0, 0.0].into());
    assert_eq!(extent.maximum(), &[2.0, 1.0, 2.0].into())
}

#[test]
fn contains_follows_the_sign() {
    let crossed = crossed();
    assert!(crossed.contains(&Coordinate::const_from([0.0, 0.0, 1.0])));
    assert!(crossed.contains(&Coordinate::const_from([1.8, 0.0, 1.0])));
    assert!(!crossed.contains(&Coordinate::const_from([2.2, 0.0, 1.0])))
}

#[test]
#[should_panic(expected = "a union needs at least one solid")]
fn empty_union_is_refused() {
    Union::<Cylinder>::new(vec![]);
}

/// Member counts either side of the count at which pruning takes over, so that
/// the pruned answers and the plain ones are held to the same account.
const COUNTS: [usize; 2] = [16, 400];

/// Pruning may only decide which members to skip, never what the answer is, so
/// it has to agree with asking every member outright.
#[test]
fn pruning_agrees_with_asking_every_member() {
    COUNTS.into_iter().for_each(|count| {
        let solids = thicket(count);
        let united = Union::new(thicket(count));
        let mut spread = Spread::default();
        (0..2000).for_each(|_| {
            let point = spread.point(14.0);
            let brute = solids
                .iter()
                .map(|solid| solid.signed_distance(&point))
                .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
            assert_length(united.signed_distance(&point), brute.value())
        })
    })
}

/// Whatever the members do where they overlap, and however few of them a query
/// looks at, the point it comes back with has to lie on the surface.
#[test]
fn closest_points_land_on_a_crowded_surface() {
    COUNTS.into_iter().for_each(|count| {
        let united = Union::new(thicket(count));
        let mut spread = Spread::default();
        (0..500).for_each(|_| {
            let point = spread.point(14.0);
            let (closest, normal) = united.closest_point(&point);
            assert_length(united.signed_distance(&closest), 0.0);
            assert!((normal.norm().value() - 1.0).abs() < 1.0e-12)
        })
    })
}

/// A point beyond every member still has to find the nearest of them, which is
/// the case a search starting at the point has to widen its way out to.
#[test]
fn a_distant_point_still_finds_the_nearest_member() {
    COUNTS.into_iter().for_each(|count| {
        let solids = thicket(count);
        let united = Union::new(thicket(count));
        let point = Coordinate::const_from([500.0, -400.0, 300.0]);
        let brute = solids
            .iter()
            .map(|solid| solid.signed_distance(&point))
            .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
        assert_length(united.signed_distance(&point), brute.value())
    })
}

/// One member alone is just that member, pruning and all.
#[test]
fn a_lone_member_is_itself() {
    let united = Union::new(vec![upright()]);
    let mut spread = Spread::default();
    (0..200).for_each(|_| {
        let point = spread.point(4.0);
        assert_length(
            united.signed_distance(&point),
            upright().signed_distance(&point).value(),
        )
    })
}

/// Cylinders straight through to hexahedra, with no surface anywhere between.
///
/// A lattice covering the union, trimmed back to it, and buffered onto it —
/// the same three steps the pipeline takes from a tessellation, asking the
/// cylinders themselves instead. Every element has to come out uninverted,
/// which is what tells the boundary was met rather than merely approached.
#[test]
fn cylinders_mesh_into_hexahedra() {
    use crate::{
        geometry::mesh::{Fitting, Mesh, Verdict, trim_to},
        math::Scalar,
    };
    let crossed = crossed();
    let mut mesh = Mesh::lattice_over(&crossed, Length::meters(0.25));
    let background = mesh.number_of_elements();
    trim_to(&crossed, &mut mesh).unwrap();
    assert!(mesh.number_of_elements() < background);
    assert!(mesh.number_of_elements() > 0);
    let mesh = mesh.buffer(&crossed, Fitting::Soft).unwrap();
    let worst = mesh
        .minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(Scalar::INFINITY, Scalar::min);
    assert!(worst > 0.05, "worst scaled jacobian {worst}");
    // The boundary should sit on the union, not merely near it.
    let surface = mesh.exterior_faces();
    let drift = surface
        .iter()
        .flatten()
        .map(|&node| {
            crossed
                .signed_distance(&mesh.coordinates()[node])
                .value()
                .abs()
        })
        .fold(0.0, Scalar::max);
    assert!(drift < 0.06, "boundary drifts {drift} from the union")
}

/// Outside the union the nearest surface point is exactly as far off as the
/// distance says it is, since the member that measured the distance owns that
/// point and no other member can bury it without being nearer itself.
///
/// Landing farther means the answer jumped to some other part of the solid,
/// which is what choosing among the members' own closest points used to do
/// wherever the nearest of them happened to be buried.
#[test]
fn closest_points_outside_are_exactly_as_far_as_told() {
    let united = Union::new(thicket(60));
    let mut spread = Spread::default();
    let mut checked = 0;
    (0..20000).for_each(|_| {
        let point = spread.point(14.0);
        let told = united.signed_distance(&point);
        if told.value() > 1.0e-6 {
            checked += 1;
            let (closest, _) = united.closest_point(&point);
            assert_length((&closest - &point).norm(), told.value())
        }
    });
    assert!(checked > 1000, "only {checked} points fell outside")
}
