use super::super::{Loop, test::unit_cube};
use crate::geometry::mesh::Connectivity;

#[test]
fn tessellates_unit_cube() {
    let tessellation = unit_cube().tessellate().unwrap();
    let mesh = tessellation.mesh();
    assert_eq!(mesh.number_of_nodes(), 8);
    let Connectivity::Triangular(block) = &mesh.connectivities()[0] else {
        panic!("expected a triangular mesh");
    };
    let triangles: Vec<[usize; 3]> = block.iter().copied().collect();
    assert_eq!(triangles.len(), 12);

    let point = |node: usize| {
        let coordinate = &mesh.coordinates()[node];
        [
            coordinate[0].value(),
            coordinate[1].value(),
            coordinate[2].value(),
        ]
    };
    let mut area = 0.0f64;
    for &[a, b, c] in triangles.iter() {
        let (pa, pb, pc) = (point(a), point(b), point(c));
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let normal = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        area += 0.5 * (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        let centroid = [
            (pa[0] + pb[0] + pc[0]) / 3.0 - 0.5,
            (pa[1] + pb[1] + pc[1]) / 3.0 - 0.5,
            (pa[2] + pb[2] + pc[2]) / 3.0 - 0.5,
        ];
        let outward = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2];
        assert!(outward > 0.0, "triangle {:?} winds inward", [a, b, c]);
    }
    assert!((area - 6.0).abs() < 1e-9, "surface area was {area}");
}

#[test]
fn rejects_faces_with_holes() {
    let mut brep = unit_cube();
    brep.faces[0].bounds.push(Loop { half_edges: vec![] });
    assert_eq!(
        brep.tessellate().err(),
        Some("faces with holes are not yet supported")
    );
}
