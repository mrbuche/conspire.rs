use super::{Groups, Name};
use crate::geometry::mesh::tessellation::corefine::{Feature, Key};

#[test]
fn a_name_does_not_depend_on_the_order_of_the_features() {
    let (a, b, c) = (Feature::Vertex(4), Feature::Edge([1, 2]), Feature::Face(7));
    let name = Name::new([a.clone(), b.clone(), c.clone()]);
    assert_eq!(name, Name::new([c.clone(), a.clone(), b.clone()]));
    assert_eq!(
        name,
        Name::new([b.clone(), c.clone(), a.clone(), b.clone()])
    );
    assert_ne!(name, Name::new([a, b]));
}

#[test]
fn the_key_seen_from_either_surface_has_one_name() {
    let (first, second) = (Feature::Edge([0, 1]), Feature::Face(3));
    let key = Key {
        first: first.clone(),
        second: second.clone(),
    };
    let flipped = Key {
        first: second,
        second: first,
    };
    assert_eq!(Name::from(&key), Name::from(&flipped));
}

#[test]
fn items_alone_are_their_own_root() {
    let mut groups = Groups::<usize>::default();
    assert_eq!(groups.find(&5), 5);
    assert!(groups.roots().is_empty());
}

#[test]
fn joined_items_share_the_smallest_as_root_in_any_order() {
    let mut groups = Groups::<usize>::default();
    groups.join(&9, &4);
    groups.join(&7, &9);
    groups.join(&2, &11);
    groups.join(&11, &7);
    [2, 4, 7, 9, 11]
        .iter()
        .for_each(|item| assert_eq!(groups.find(item), 2));
    let roots = groups.roots();
    assert_eq!(roots.len(), 4);
    assert!(roots.values().all(|&root| root == 2));
    assert!(!roots.contains_key(&2));
}

#[test]
fn separate_groups_stay_separate() {
    let mut groups = Groups::<Name>::default();
    let name = |n| Name::new([Feature::Vertex(n)]);
    groups.join(&name(3), &name(1));
    groups.join(&name(8), &name(6));
    assert_eq!(groups.find(&name(3)), name(1));
    assert_eq!(groups.find(&name(8)), name(6));
    groups.join(&name(3), &name(8));
    assert_eq!(groups.find(&name(8)), name(1));
}
