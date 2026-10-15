use super::Labels;

#[test]
fn as_slice() {
    assert_eq!(Labels::new(vec![7, 3, 9], 3).as_slice(), [7, 3, 9]);
}

#[test]
fn index_of() {
    let labels = Labels::new(vec![7, 3, 9], 3);
    assert_eq!(labels.index_of(9), Some(2));
    assert_eq!(labels.index_of(7), Some(0));
    assert_eq!(labels.index_of(4), None);
}

#[test]
fn index_of_first_duplicate() {
    assert_eq!(Labels::new(vec![5, 5], 2).index_of(5), Some(0));
}

#[test]
#[should_panic(expected = "exactly one label")]
fn new_with_too_few_labels() {
    Labels::new(vec![1, 2], 3);
}

#[test]
#[should_panic(expected = "exactly one label")]
fn new_with_too_many_labels() {
    Labels::new(vec![1, 2, 3], 2);
}
