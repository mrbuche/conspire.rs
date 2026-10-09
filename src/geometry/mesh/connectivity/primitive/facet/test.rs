use super::LocalFacets;
use crate::geometry::mesh::{Connectivity, PrimitiveConnectivity, Surface};

#[test]
fn a_facet_takes_the_nodes_of_its_element() {
    let facet = <PrimitiveConnectivity<3, 8> as LocalFacets>::facet(
        &[0, 1, 5, 4],
        &[10, 11, 12, 13, 14, 15, 16, 17],
    );
    assert_eq!(facet.as_ref(), [10, 11, 15, 14]);
}

#[test]
fn a_triangle_or_quadrilateral_in_a_wedge_has_its_own_size() {
    let nodes = [10, 11, 12, 13, 14, 15];
    let triangle = <PrimitiveConnectivity<3, 6> as LocalFacets>::facet(&[0, 2, 1], &nodes);
    let quadrilateral = <PrimitiveConnectivity<3, 6> as LocalFacets>::facet(&[0, 1, 4, 3], &nodes);
    assert_eq!(triangle.as_ref(), [10, 12, 11]);
    assert_eq!(quadrilateral.as_ref(), [10, 11, 14, 13]);
}

#[test]
#[should_panic(expected = "a facet does not have 3 nodes")]
fn a_tetrahedron_facet_has_three_nodes() {
    <PrimitiveConnectivity<3, 4> as LocalFacets>::facet(&[0, 1, 2, 3], &[0, 1, 2, 3]);
}

#[test]
#[should_panic(expected = "a facet has 5 nodes, not three or four")]
fn a_mixed_facet_has_three_or_four_nodes() {
    <PrimitiveConnectivity<3, 6> as LocalFacets>::facet(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4, 5]);
}

fn gathers_the_table<T: LocalFacets>(nodes: usize) {
    let identity = (0..nodes).collect::<Vec<_>>();
    T::LOCAL_FACETS
        .iter()
        .for_each(|local| assert_eq!(T::facet(local, &identity).as_ref(), *local));
}

#[test]
fn gathering_through_the_identity_gives_the_table() {
    gathers_the_table::<PrimitiveConnectivity<3, 8>>(8);
    gathers_the_table::<PrimitiveConnectivity<3, 4>>(4);
    gathers_the_table::<PrimitiveConnectivity<3, 5>>(5);
    gathers_the_table::<PrimitiveConnectivity<3, 6>>(6);
    gathers_the_table::<PrimitiveConnectivity<2, 4>>(4);
    gathers_the_table::<PrimitiveConnectivity<2, 3>>(3);
}

fn sizes<T: LocalFacets>() -> Vec<usize> {
    T::LOCAL_FACETS.iter().map(|face| face.len()).collect()
}

fn sphere<T: LocalFacets>() {
    let surface = Surface::try_from(T::LOCAL_FACETS).unwrap();
    assert!(surface.is_sphere());
}

fn loop_of_edges<T: LocalFacets>(nodes: usize) {
    let edges = T::LOCAL_FACETS;
    assert_eq!(edges.len(), nodes);
    assert!(edges.iter().all(|edge| edge.len() == 2));
    for node in 0..nodes {
        assert_eq!(edges.iter().filter(|edge| edge[0] == node).count(), 1);
        assert_eq!(edges.iter().filter(|edge| edge[1] == node).count(), 1);
    }
}

#[test]
fn a_hexahedron_has_six_quadrilaterals_that_close_up() {
    assert_eq!(sizes::<PrimitiveConnectivity<3, 8>>(), [4; 6]);
    sphere::<PrimitiveConnectivity<3, 8>>();
}

#[test]
fn a_tetrahedron_has_four_triangles_that_close_up() {
    assert_eq!(sizes::<PrimitiveConnectivity<3, 4>>(), [3; 4]);
    sphere::<PrimitiveConnectivity<3, 4>>();
}

#[test]
fn a_pyramid_has_four_triangles_and_a_quadrilateral_that_close_up() {
    assert_eq!(sizes::<PrimitiveConnectivity<3, 5>>(), [3, 3, 3, 3, 4]);
    sphere::<PrimitiveConnectivity<3, 5>>();
}

#[test]
fn a_wedge_has_three_quadrilaterals_and_two_triangles_that_close_up() {
    assert_eq!(sizes::<PrimitiveConnectivity<3, 6>>(), [4, 4, 4, 3, 3]);
    sphere::<PrimitiveConnectivity<3, 6>>();
}

#[test]
fn polygons_have_a_loop_of_edges() {
    loop_of_edges::<PrimitiveConnectivity<2, 4>>(4);
    loop_of_edges::<PrimitiveConnectivity<2, 3>>(3);
}

#[test]
fn every_node_is_in_a_facet() {
    fn nodes<T: LocalFacets>(count: usize) {
        let mut nodes = T::LOCAL_FACETS.concat();
        nodes.sort_unstable();
        nodes.dedup();
        assert_eq!(nodes, (0..count).collect::<Vec<_>>());
    }
    nodes::<PrimitiveConnectivity<3, 8>>(8);
    nodes::<PrimitiveConnectivity<3, 4>>(4);
    nodes::<PrimitiveConnectivity<3, 5>>(5);
    nodes::<PrimitiveConnectivity<3, 6>>(6);
    nodes::<PrimitiveConnectivity<2, 4>>(4);
    nodes::<PrimitiveConnectivity<2, 3>>(3);
}

#[test]
fn the_connectivity_gives_the_facets_of_its_type() {
    assert_eq!(
        Connectivity::Hexahedral(Vec::new().into()).local_facets(),
        PrimitiveConnectivity::<3, 8>::LOCAL_FACETS
    );
    assert_eq!(
        Connectivity::Tetrahedral(Vec::new().into()).local_facets(),
        PrimitiveConnectivity::<3, 4>::LOCAL_FACETS
    );
    assert_eq!(
        Connectivity::Pyramidal(Vec::new().into()).local_facets(),
        PrimitiveConnectivity::<3, 5>::LOCAL_FACETS
    );
    assert_eq!(
        Connectivity::Wedge(Vec::new().into()).local_facets(),
        PrimitiveConnectivity::<3, 6>::LOCAL_FACETS
    );
    assert_eq!(
        Connectivity::Quadrilateral(Vec::new().into()).local_facets(),
        PrimitiveConnectivity::<2, 4>::LOCAL_FACETS
    );
    assert_eq!(
        Connectivity::Triangular(Vec::new().into()).local_facets(),
        PrimitiveConnectivity::<2, 3>::LOCAL_FACETS
    );
}
