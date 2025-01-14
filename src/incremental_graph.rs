// pub use crate::builder::*;
pub use crate::traits::GraphInterface;
pub use crate::traits::PointInterface;
// use crate::utils::*;
// use std::cmp::Reverse;
// use std::collections::BinaryHeap;
// use rustc_hash::FxHashSet;

// pub struct GraphParams {
//     a: f32,
//     r: usize,
//     l: usize,
// }

pub trait IncrementalGraph<P: PointInterface, const R: usize> {
    fn new() -> Self;
    fn search(&self, point: P, l: usize) -> Vec<(f32, u32)>;
    fn insert(&mut self, point: P, a: f32) -> u32;
    fn delete(&mut self, index: u32, a: f32);
    fn remove(&mut self, index: u32, a: f32);
    fn remove_list(&self) -> Vec<u32>;
}

#[derive(Clone)]
pub struct Node<P: PointInterface, const R: usize> {
    point: P,
    edges: [u32; R],
}

pub trait Storage<P: PointInterface, const R: usize> {
    fn new() -> Self;

    fn get(&self, index: &u32) -> Option<Node<P, R>>;

    fn set(&mut self, index: u32, node: Node<P, R>) -> Option<Node<P, R>>;

    fn delete(&mut self, index: u32);

    fn remove(&mut self, index: u32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ic_cdk::trap;
    use itertools::Itertools;
    use std::collections::BTreeMap;
    // use itertools::Itertools;
    use rustc_hash::{FxHashMap, FxHashSet};
    // use crate::utils::*;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    // use rustc_hash::FxHashSet;

    const R: usize = 90;

    struct TestStorage<P: PointInterface> {
        btree: BTreeMap<u32, Node<P, 90>>,
    }

    impl<P> Storage<P, R> for TestStorage<P>
    where
        P: PointInterface,
    {
        fn new() -> Self {
            Self {
                btree: BTreeMap::new(),
            }
        }

        fn get(&self, index: &u32) -> Option<Node<P, R>> {
            self.btree.get(index).cloned()
        }

        fn set(&mut self, index: u32, node: Node<P, R>) -> Option<Node<P, R>> {
            self.btree.insert(index, node)
        }

        fn delete(&mut self, _index: u32) {
            todo!()
        }

        fn remove(&mut self, _index: u32) {
            todo!()
        }
    }

    pub struct Graph<P: PointInterface, S: Storage<P, R>, const R: usize> {
        storage: S,
        phantom: std::marker::PhantomData<P>,
    }

    // Partial orderなものをTotal orderにする
    #[derive(PartialEq, PartialOrd, Clone, Copy)]
    pub struct Dist<T>(pub T);

    impl<T: PartialEq> Eq for Dist<T> {}

    impl<T: PartialOrd> Ord for Dist<T> {
        fn cmp(&self, other: &Dist<T>) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).expect("NaN was compared.")
        }
    }

    impl<P, S, const R: usize> IncrementalGraph<P, R> for Graph<P, S, R>
    where
        P: PointInterface,
        S: Storage<P, R>,
    {
        fn new() -> Self {
            todo!()
        }

        fn search(&self, query: P, l: usize) -> Vec<(f32, u32)> {
            let entry_index = 0; // should be random later
            let entry_node = self.storage.get(&entry_index).unwrap();
            let dist_from_entry = Dist(query.distance(&entry_node.point));

            // let mut dist_cache = FxHashMap::default(); // Set of nodes that have been calculated the distance from query.
            let mut visited_nodes_and_dists = FxHashMap::default(); // Set of nodes that have been visited.
            let mut touched_nodes = FxHashSet::default();
            let mut heap: BinaryHeap<Reverse<(Dist<f32>, u32)>> = BinaryHeap::new();

            visited_nodes_and_dists.insert(entry_index, dist_from_entry);
            touched_nodes.insert(entry_index);
            heap.push(Reverse((dist_from_entry, entry_index)));

            let mut num_skips = 0;

            while let Some(Reverse((dist, node_index))) = heap.peek().cloned() {
                if num_skips > l {
                    break;
                }

                // If the node already has been visited, skip this node.
                let edge_nodes = if visited_nodes_and_dists.insert(node_index, dist).is_some() {
                    num_skips += 1;
                    continue;
                } else {
                    num_skips = 0;
                    self.storage.get(&node_index).expect("")
                };

                // Pushing uncompared edge nodes into heap
                for edge_node_index in edge_nodes.edges {
                    // If the edge node has not been compared before,
                    if !touched_nodes.insert(edge_node_index) {
                        let edge_node = self.storage.get(&edge_node_index).expect("msg");
                        let dist = Dist(query.distance(&edge_node.point));
                        heap.push(Reverse((dist, edge_node_index)));
                    }
                }
            }

            // Return sorted visited nodes. The first k nodes might be used for results of query search.
            visited_nodes_and_dists
                .into_iter()
                .map(|v| (v.1, v.0))
                .sorted()
                .map(|(dist, node_index)| (dist.0, node_index))
                .collect()
        }

        fn insert(&mut self, _point: P, _a: f32) -> u32 {
            todo!()
        }

        fn delete(&mut self, _index: u32, _a: f32) {
            todo!()
        }

        fn remove(&mut self, _index: u32, _a: f32) {
            todo!()
        }

        fn remove_list(&self) -> Vec<u32> {
            todo!()
        }
    }

    #[test]
    fn test_get() {}
}
