pub mod octree;
pub mod quadtree;

#[derive(Clone, Copy)]
pub enum Balancing {
    Strong,
    Weak,
}
