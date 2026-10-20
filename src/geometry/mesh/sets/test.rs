use super::{NodeSet, NodeSets, SideSet, SideSets};

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
fn a_side_set_has_no_faces_until_it_is_resolved() {
    assert!(SideSet::new(vec![(0, 1)]).faces().is_empty());
}

#[test]
fn a_side_set_has_the_face_of_each_side() {
    let set = SideSet::new(vec![(0, 1), (2, 3)]).with_faces(vec![vec![1, 2, 3], vec![4, 5, 6]]);
    assert_eq!(set.sides(), [(0, 1), (2, 3)]);
    assert_eq!(set.faces(), [vec![1, 2, 3], vec![4, 5, 6]]);
}

#[test]
#[should_panic(expected = "exactly one face for each side")]
fn a_side_set_needs_a_face_for_every_side() {
    SideSet::new(vec![(0, 1), (2, 3)]).with_faces(vec![vec![1, 2, 3]]);
}

#[test]
fn restricting_a_side_set_keeps_the_faces_of_the_sides_it_keeps() {
    let set = SideSet::new(vec![(0, 1), (2, 3), (4, 5)])
        .with_label(6)
        .with_faces(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
    let kept = set
        .restrict(|(element, ordinal)| (element != 2).then_some((element + 10, ordinal)))
        .unwrap();
    assert_eq!(kept.sides(), [(10, 1), (14, 5)]);
    assert_eq!(kept.faces(), [vec![1, 2, 3], vec![7, 8, 9]]);
    assert_eq!(kept.label(), Some(6));
}

#[test]
fn restricting_a_side_set_without_faces_leaves_it_without() {
    let kept = SideSet::new(vec![(0, 1), (2, 3)]).restrict(Some).unwrap();
    assert!(kept.faces().is_empty());
}

#[test]
fn side_sets_are_equal_whatever_their_faces() {
    let set = SideSet::new(vec![(0, 1)]).with_name("a");
    assert_eq!(set, set.clone().with_faces(vec![vec![1, 2, 3]]));
    assert_ne!(set, SideSet::new(vec![(0, 2)]).with_name("a"));
    assert_ne!(set, SideSet::new(vec![(0, 1)]).with_name("b"));
}
