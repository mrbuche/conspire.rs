use crate::geometry::mesh::{
    IsotropicSizing, Remeshing, RemeshingMetric,
    test::{mesh, mesh_with_node_sets},
};

#[test]
fn remesh_clears_node_sets() {
    let remeshed = mesh_with_node_sets()
        .remesh(Remeshing {
            iterations: 1,
            metric: RemeshingMetric::Isotropic(IsotropicSizing::Uniform { length: None }),
        })
        .unwrap();
    assert!(remeshed.node_sets().is_empty());
}

#[test]
fn zero_iterations_keeps_node_sets() {
    let remeshed = mesh_with_node_sets()
        .remesh(Remeshing {
            iterations: 0,
            metric: RemeshingMetric::Isotropic(IsotropicSizing::Uniform { length: None }),
        })
        .unwrap();
    assert!(!remeshed.node_sets().is_empty());
}

#[test]
fn sanity_lone_block_remesh() {
    let remeshed = mesh()
        .remesh(Remeshing {
            iterations: 1,
            metric: RemeshingMetric::Isotropic(IsotropicSizing::Uniform { length: None }),
        })
        .unwrap();
    assert!(remeshed.number_of_nodes() > 0);
}
