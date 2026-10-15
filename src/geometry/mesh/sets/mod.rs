use crate::math::Labels;

macro_rules! sets {
    ($sets: ident, $member: ty, $doc: literal) => {
        #[doc = $doc]
        pub struct $sets {
            members: Vec<Vec<$member>>,
            labels: Option<Labels>,
        }
        impl $sets {
            pub fn members(&self) -> &[Vec<$member>] {
                &self.members
            }
            pub fn labels(&self) -> Option<&[usize]> {
                self.labels.as_ref().map(Labels::as_slice)
            }
        }
        impl From<Vec<Vec<$member>>> for $sets {
            fn from(members: Vec<Vec<$member>>) -> Self {
                Self {
                    members,
                    labels: None,
                }
            }
        }
        impl From<(Vec<Vec<$member>>, Vec<usize>)> for $sets {
            fn from((members, labels): (Vec<Vec<$member>>, Vec<usize>)) -> Self {
                let labels = Some(Labels::new(labels, members.len()));
                Self { members, labels }
            }
        }
    };
}

sets!(
    NodeSets,
    usize,
    "Sets of nodes, optionally labeled with a number for each set."
);
sets!(
    SideSets,
    (usize, usize),
    "Sets of element sides, optionally labeled with a number for each set."
);
