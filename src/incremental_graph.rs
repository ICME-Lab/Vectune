use itertools::Itertools;
use log::debug;
use rand::Rng;
use rustc_hash::FxHashSet;

pub use crate::traits::GraphInterface;
pub use crate::traits::PointInterface;

// Partial orderなものをTotal orderにする
#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct Dist<T>(pub T);

impl<T: PartialEq> Eq for Dist<T> {}

impl<T: PartialOrd> Ord for Dist<T> {
    fn cmp(&self, other: &Dist<T>) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).expect("NaN was compared.")
    }
}

#[derive(Clone)]
pub struct ActiveNode<P: PointInterface<D>, const R: usize, const D: usize> {
    point: P,
    edges: [u32; R],
}

#[derive(Clone)]
pub struct DeletedNode<P: PointInterface<D>, const R: usize, const D: usize> {
    point: P,
    edges: [u32; R],
}

#[derive(Clone)]
pub enum Node<P: PointInterface<D>, const R: usize, const D: usize> {
    Active(ActiveNode<P, R, D>),
    Deleted(DeletedNode<P, R, D>),
}

impl<P, const R: usize, const D: usize> Node<P, R, D>
where
    P: PointInterface<D>,
{
    pub fn new(point: P, edges: [u32; R]) -> Self {
        Self::Active(ActiveNode { point, edges })
    }

    pub fn point(&self) -> P {
        let point = match self {
            Node::Active(node) => node.point.clone(),
            Node::Deleted(node) => node.point.clone(),
        };
        point
    }

    pub fn edges(&self) -> Vec<u32> {
        let raw_edges = self.raw_edges();
        raw_edges.into_iter().sorted().dedup().collect()
    }

    pub fn raw_edges(&self) -> [u32; R] {
        let edges = match self {
            Node::Active(node) => node.edges,
            Node::Deleted(node) => node.edges,
        };
        edges
    }

    pub fn mark_as_deleted(&mut self) {
        match self {
            Node::Active(active_node) => {
                *self = Self::Deleted(DeletedNode {
                    point: active_node.point.clone(),
                    edges: active_node.edges,
                })
            }
            Node::Deleted(_) => {}
        }
    }
}

pub trait Storage<P: PointInterface<D>, const R: usize, const D: usize, T: Rng> {
    fn new() -> Self;

    fn get(&self, index: &u32) -> Option<Node<P, R, D>>;

    fn set(&mut self, index: u32, active_node: ActiveNode<P, R, D>) -> Option<Node<P, R, D>>;

    fn delete(&mut self, index: u32);

    fn alloc(&mut self) -> u32;

    fn pick_random_active_node(&self, rng: &mut T) -> (u32, ActiveNode<P, R, D>);

    fn backlink(&self, index: u32) -> Vec<u32>;
}

pub struct Graph<P: PointInterface<D>, S: Storage<P, R, D, T>, const R: usize, const D: usize, T>
where
    T: Rng,
{
    storage: S,
    phantom1: std::marker::PhantomData<P>,
    phantom2: std::marker::PhantomData<T>,
}

impl<P, S, const R: usize, const D: usize, T> Graph<P, S, R, D, T>
where
    P: PointInterface<D>,
    S: Storage<P, R, D, T>,
    T: Rng,
{
    pub fn new(init_point: P) -> Self {
        let mut storage = S::new();
        let init_node = ActiveNode {
            point: init_point,
            edges: [0; R],
        };
        storage.set(0, init_node);
        Self {
            storage,
            phantom1: std::marker::PhantomData,
            phantom2: std::marker::PhantomData,
        }
    }

    pub fn search(&self, query: &P, l: usize, rng: &mut T) -> Vec<(Dist<f32>, u32)> {
        let mut list: Vec<(Dist<f32>, u32)> = vec![]; // This list has working nodes that is visited or candidate for next searching.
        let mut visited_nodes = FxHashSet::default();
        let mut touched_nodes = FxHashSet::default(); // Used for avoiding recalculation of distance already in list

        // Set entry node and its edges
        let (entry_node_index, entry_node) = self.storage.pick_random_active_node(rng);
        list.push((Dist(entry_node.point.distance(query)), entry_node_index));
        touched_nodes.insert(entry_node_index);

        // Merge two Vecs into a Vec of max_length by order.
        fn merge_lists(
            list_a: Vec<(Dist<f32>, u32)>,
            list_b: Vec<(Dist<f32>, u32)>,
            max_length: usize,
        ) -> Vec<(Dist<f32>, u32)> {
            let mut new_list = Vec::new();
            let mut list_a = list_a.into_iter();
            let mut list_b = list_b.into_iter();

            let mut smallest_in_a = list_a.next();
            let mut smallest_in_b = list_b.next();

            // Pushes smaller values.
            while new_list.len() < max_length {
                match (smallest_in_a, smallest_in_b) {
                    (None, None) => break,
                    (Some(a), None) => {
                        new_list.push(a);
                        smallest_in_a = list_a.next();
                    }
                    (None, Some(b)) => {
                        new_list.push(b);
                        smallest_in_b = list_b.next();
                    }
                    (Some(a), Some(b)) => {
                        if a < b {
                            new_list.push(a);
                            smallest_in_a = list_a.next();
                        } else {
                            new_list.push(b);
                            smallest_in_b = list_b.next();
                        }
                    }
                }
            }

            new_list
        }

        // Visit to a node that is nearest to query of current candidate nodes.
        loop {
            // Find an unvisited node that has smallest dist
            let Some((_, working_node_index)) = list
                .iter()
                .find(|(_, node_index)| visited_nodes.insert(*node_index))
            else {
                // If all nodes in the list have been visited, searching is finished
                // The results are the fist k nodes in the list
                // debug!("search: break");
                break;
            };

             // Get next node from the storage
             let Some(working_node) = self.storage.get(&working_node_index) else {
                panic!("working node should be active")
            };

            // New candidate nodes
            let candidates: Vec<_> = working_node
                .edges()
                .into_iter()
                .filter_map(|edge_node_index| {
                    let Some(edge_node) = self.storage.get(&edge_node_index) else {
                        panic!("")
                    };
                    // If a distance of this edge node is already calculated, skip this node
                    if !touched_nodes.insert(edge_node_index) {
                        return None;
                    }
                    
                    Some((Dist(edge_node.point().distance(query)), edge_node_index))
                })
                .sorted() // candidates should be order for merging to list
                .dedup() // Is this necessary?
                .collect();

            // If the working node is deleted, remove this node from the list
            let (new_list, candidates) = match working_node {
                Node::Active(_) => {(list.clone(), candidates)},
                Node::Deleted(_) => {
                    let remove_deleted_node = |(d, i)| {
                        if i == *working_node_index {
                            None
                        } else {
                            Some((d,i))
                        }
                    };
                    let new_list: Vec<_> = list.clone().into_iter().filter_map(remove_deleted_node).collect();
                    let candidates: Vec<_> = candidates.into_iter().filter_map(remove_deleted_node).collect();
                    (new_list, candidates)
                },
            };

            // Merge current candidates lists and new ones by order
            list = merge_lists(new_list, candidates, l);
        }

        list
    }

    // Purne nodes in order to maintain edge diversity.
    pub fn prune(&mut self, mut candidates: Vec<(Dist<f32>, u32)>, a: f32) -> [u32; R] {
        // Edge list is saved as R length sized array.
        let mut new_edge_nodes = [0; R]; // or u32::MAX

        if candidates.len() <= R {
            for (i, (_, node_index)) in candidates.into_iter().enumerate() {
                new_edge_nodes[i] = node_index;
            }
        } else {
            let mut edge_index = 0;

            debug!("prune: candidates.len(): {}", candidates.len());

            while let Some((first, rest)) = candidates.split_first() {
                let (_, pa) = *first; // pa is p asterisk (p*), which is nearest point to p in this loop
                new_edge_nodes[edge_index] = pa;

                candidates = rest.to_vec();

                // if α · d(p*, p') <= d(p, p') then remove p' from v
                candidates.retain(|&(dist_xp_pd, pd)| {
                    let Some(Node::Active(pa_node)) = self.storage.get(&pa) else {
                        panic!("")
                    };
                    let Some(Node::Active(pd_node)) = self.storage.get(&pd) else {
                        panic!("")
                    };
                    let dist_pa_pd = pa_node.point.distance(&pd_node.point);

                    a * dist_pa_pd > dist_xp_pd.0
                });

                edge_index += 1;
                if edge_index == R {
                    break;
                }
            }
        }

        new_edge_nodes
    }

    pub fn insert(&mut self, point: P, a: f32, l: usize, rng: &mut T) -> u32 {
        // Search the new point and use route nodes as candidates of its edges
        let candidates = self.search(&point, l, rng);
        let edges = Self::prune(self, candidates, a);

        // Add new node to the storage
        let new_node_index = self.storage.alloc();
        self.storage
            .set(new_node_index, ActiveNode { point, edges });

        // Update graph edges: add the new node to its edge node's edges
        let edges = edges
            .to_vec()
            .into_iter()
            .sorted()
            .dedup()
            .collect::<Vec<_>>();
        debug!("node {new_node_index}: {:?}", edges);
        for edge_node_index in edges {
            // If candidate list has delete nodes, uses undeleted edges of delete node instead.
            self.add_edge(&edge_node_index, new_node_index);
        }

        debug!("\n");

        new_node_index
    }

    pub fn delete(&mut self, index: u32, _a: f32) {
        /*
        方針:
        vamanaのdelete algorithと同等の効果があるローカルノードの置き換えをオンデマンドで行う。
        削除されたnode-pをedgeに持つnodeは、node-pのedgeを追加の候補としてits edgeに追加する。
        これを検索時に行うことで、同等の効果が得られるはずだ。

        vamanaの性能を保持したまま、検索時の計算コストの増加だけ済む。
        さらに、このコストは逐次的にrobust-pruneをかけていく事で減らすことができるので、backlinksを保存することが難しい場合には向いている。
        また、挿入の実装も小さくすることができる。

        削除したノードのエッジのindexをとっておく必要がある。
        */

        self.storage.delete(index);
    }

    pub fn remove(&mut self, _index: u32, _a: f32) {
        todo!()
    }

    pub fn remove_list(&self) -> Vec<u32> {
        todo!()
    }

    // Add a new edge node to the node.
    fn add_edge(&mut self, target_node_index: &u32, new_edge_node_index: u32) {
        debug!("add_edge: {target_node_index}, new_edge_node_index: {new_edge_node_index}");
        let Some(target_node) = self.storage.get(target_node_index) else {
            panic!("")
        };
        let mut edges = target_node.edges();
        edges.push(new_edge_node_index);
        debug!("add_edge: {:?}", edges);

        let new_edges = if edges.len() > R {
            let candidates: Vec<_> = edges
                .into_iter()
                .map(|candidate_node_index| {
                    let Some(candidate_node) = self.storage.get(&candidate_node_index) else {
                        panic!("")
                    };
                    (
                        Dist(target_node.point().distance(&candidate_node.point())),
                        candidate_node_index,
                    )
                })
                .sorted() // candidates should be order for merging to list
                .dedup()
                .collect();

            self.prune(candidates, 2.0)
        } else {
            let mut new_edge_nodes = [0; R];
            for (i, edge_node_index) in edges.into_iter().enumerate() {
                new_edge_nodes[i] = edge_node_index;
            }

            new_edge_nodes
        };

        debug!(
            "add_edge: afger: node {target_node_index}: {:?}",
            new_edges
                .to_vec()
                .into_iter()
                .sorted()
                .dedup()
                .collect::<Vec<_>>()
        );

        self.storage.set(
            *target_node_index,
            ActiveNode {
                point: target_node.point(),
                edges: new_edges,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::point::Point;

    use super::*;
    use itertools::Itertools;
    use rand::SeedableRng;
    use rand::{rngs::SmallRng, seq::SliceRandom};
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rustc_hash::{FxHashMap, FxHashSet};

    const R: usize = 20;
    const L: usize = 125;

    fn normalize_to_unit_length(vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|&v| v * v).sum::<f32>().sqrt(); // ユークリッドノルムを計算
        if norm.abs() < f32::EPSILON {
            // ノルムがゼロの場合（すべての要素が0）、そのまま返す
            return vector;
        }
        vector.into_iter().map(|v| v / norm).collect()
    }

    struct TestStorage {
        nodes: FxHashMap<u32, Node<Point<DIM>, R, DIM>>,
        backlinks: FxHashMap<u32, FxHashSet<u32>>,
        index: u32,
    }

    impl Storage<Point<DIM>, R, DIM, SmallRng> for TestStorage {
        fn new() -> Self {
            Self {
                nodes: FxHashMap::default(),
                index: 0,
                backlinks: FxHashMap::default(),
            }
        }

        fn get(&self, index: &u32) -> Option<Node<Point<DIM>, R, DIM>> {
            self.nodes.get(index).cloned()
        }

        fn set(&mut self, index: u32, active_node: ActiveNode<Point<DIM>, R, DIM>) -> Option<Node<Point<DIM>, R, DIM>> {
            // Set rc of this node
            match self.backlinks.get(&index) {
                Some(_) => {}
                None => {
                    self.backlinks.insert(index, FxHashSet::default());
                }
            }
            //
            let old_node = self.nodes.insert(index, Node::Active(active_node.clone()));
            let old_edges = match &old_node {
                Some(node) => node.edges(),
                None => {
                    vec![]
                }
            };
            // Delete backlinks // 前に持っていたedgeからのbacklinkを一度全て削除する。
            for edge_index in old_edges {
                let Some(set) = self.backlinks.get_mut(&edge_index) else {
                    panic!("")
                };
                set.remove(&index);
            }
            // Add backlinks
            for edge_index in active_node.edges.into_iter().sorted().dedup() {
                let Some(set) = self.backlinks.get_mut(&edge_index) else {
                    panic!("")
                };
                set.remove(&index);
            }

            old_node
        }

        fn delete(&mut self, index: u32) {
            let Some(node) = self.nodes.get_mut(&index) else {
                return;
            };

            // Mark the node as "deleted"
            node.mark_as_deleted();
        }

        fn alloc(&mut self) -> u32 {
            self.index += 1;
            self.index
        }

        fn pick_random_active_node(&self, rng: &mut SmallRng) -> (u32, ActiveNode<Point<DIM>, R, DIM>) {
            let keys: Vec<_> = self.nodes.keys().collect();

            loop {
                let Some(&random_key) = keys.get(rng.gen_range(0..keys.len())) else {
                    panic!("btree is empty")
                };

                let Some(Node::Active(active_node)) = self.get(random_key) else {
                    continue;
                };

                return (*random_key, active_node);
            }
        }

        fn backlink(&self, _index: u32) -> Vec<u32> {
            todo!()
        }
    }

    const DIM: usize = 384;

    #[test]
    fn test_get() {
        let mut rng: SmallRng = SmallRng::from_entropy();
        let mut graph: Graph<Point<DIM>, TestStorage, R, DIM, SmallRng> = Graph::new(gen_point(&mut rng));

        env_logger::init();

        let iter_count = 500;
        let split_count = 0;

        let mut test_points: Vec<(u32, Point<DIM>)> = (0..iter_count)
            .into_iter()
            .map(|i| (i + 1, gen_point(&mut rng)))
            .collect();

        for (index, p) in test_points.iter() {
            let new_index = graph.insert(p.clone(), 2.0, L, &mut rng);
            assert_eq!(*index, new_index);
        }

        test_points.shuffle(&mut rng);

        for (i, _) in test_points[0..split_count].to_vec() {
            graph.delete(i, 2.0);
        }

        let hit_counts: f32 = test_points[split_count as usize..]
            .par_iter()
            .map_init(
                || SmallRng::from_entropy(),
                |rng, (_, p)| {
                    let p = p.clone();
                    let ground_truth: Vec<(Dist<f32>, u32)> = test_points
                        .iter()
                        .map(|(i, q)| (Dist(p.distance(q)), *i))
                        .sorted()
                        .collect();

                    let result = graph.search(&p, L, rng);

                    // println!("{:?}, {:?}", ground_truth.split(6), result[0..6]);
                    let g_list: Vec<_> = ground_truth[0..5]
                        .into_iter()
                        .map(|(_, index)| index)
                        .collect();
                    let r_list: Vec<_> =
                        result[0..5].into_iter().map(|(_, index)| index).collect();

                    // for delete_index in 1..split_count {
                    //     assert!(!r_list.contains(delete_index));
                    // }
                    // println!("{:?}, {:?}",g_list, r_list);

                    let set1: FxHashSet<_> = ground_truth[0..5]
                        .into_iter()
                        .map(|(_, index)| index)
                        .collect();
                    let set2: FxHashSet<_> =
                        result[0..5].into_iter().map(|(_, index)| index).collect();

                    let intersection_count = set1.intersection(&set2).count();

                    intersection_count as f32
                },
            )
            .sum();

        println!("{:?}", hit_counts / ((5 * iter_count) as f32));
    }

    fn gen_point(rng: &mut SmallRng) -> Point<DIM> {
        let new_point: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
        let new_point = normalize_to_unit_length(new_point);
        Point(new_point)
    }

    #[test]
    fn test_backlinks() {
        // rcが、一致しているかをテストする。
        // edgeを変更した時に、rcが正しく増減するかをテストする。
    }
}
