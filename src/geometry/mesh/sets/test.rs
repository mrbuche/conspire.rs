use super::{NodeSet, NodeSets, SideSet, SideSets};
use crate::geometry::{
    Coordinates,
    mesh::{Connectivity, Mesh},
};

fn sets() -> NodeSets {
    vec![
        NodeSet::new(vec![0, 1]).with_label(7).with_name("bottom"),
        NodeSet::new(vec![2]).with_name("top"),
        NodeSet::new(vec![1, 2]),
    ]
    .into()
}

#[test]
fn by_name() {
    assert_eq!(sets().by_name("top").unwrap().nodes(), [2]);
    assert!(sets().by_name("side").is_none());
}

#[test]
fn by_label() {
    assert_eq!(sets().by_label(7).unwrap().name(), Some("bottom"));
    assert!(sets().by_label(8).is_none());
}

#[test]
fn by_name_is_first() {
    let sets: NodeSets = vec![
        NodeSet::new(vec![0]).with_name("a"),
        NodeSet::new(vec![1]).with_name("a"),
    ]
    .into();
    assert_eq!(sets.by_name("a").unwrap().nodes(), [0]);
}

#[test]
fn resolved_labels_skip_taken_ones() {
    assert_eq!(sets().resolved_labels(), [7, 1, 2]);
    let taken: NodeSets = vec![
        NodeSet::new(vec![0]),
        NodeSet::new(vec![1]).with_label(1),
        NodeSet::new(vec![2]),
    ]
    .into();
    assert_eq!(taken.resolved_labels(), [2, 1, 3]);
}

#[test]
fn from_unlabeled_members() {
    let sets: SideSets = vec![vec![(0, 1)], vec![(1, 2), (2, 0)]].into();
    assert_eq!(sets.len(), 2);
    assert_eq!(sets.iter().last().unwrap().sides(), [(1, 2), (2, 0)]);
    assert!(sets.iter().all(|set| set.label().is_none()));
}

#[test]
fn from_labeled_members() {
    let sets: NodeSets = (vec![vec![0], vec![1]], vec![10, 20]).into();
    assert_eq!(sets.by_label(20).unwrap().nodes(), [1]);
}

#[test]
#[should_panic(expected = "exactly one label")]
fn from_labeled_members_with_too_few_labels() {
    let _: NodeSets = (vec![vec![0], vec![1]], vec![10]).into();
}

#[test]
fn restrict_keeps_label_and_name() {
    let set = NodeSet::new(vec![0, 1, 2]).with_label(4).with_name("all");
    let kept = set
        .restrict(|node| (node != 1).then_some(node + 10))
        .unwrap();
    assert_eq!(kept.nodes(), [10, 12]);
    assert_eq!(kept.label(), Some(4));
    assert_eq!(kept.name(), Some("all"));
}

#[test]
fn restrict_to_nothing() {
    assert!(SideSet::new(vec![(0, 1)]).restrict(|_| None).is_none());
}

#[test]
fn select() {
    let mesh = Mesh::from((
        vec![Connectivity::Triangular(vec![[0, 1, 2]].into())],
        Coordinates::<3>::from([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ));
    assert_eq!(NodeSet::select(&mesh, |x| x[0].value() > 0.5).nodes(), [1]);
    assert!(
        NodeSet::select(&mesh, |x| x[2].value() > 0.5)
            .nodes()
            .is_empty()
    );
}
