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
pub struct ActiveNode<P: PointInterface, const R: usize> {
    point: P,
    edges: [u32; R],
    rc: usize, // Referece counter
}

#[derive(Clone)]
pub struct DeletedNode<const R: usize> {
    edges: [u32; R],
    rc: usize, // Referece counter
}

#[derive(Clone)]
pub enum Node<P: PointInterface, const R: usize> {
    Active(ActiveNode<P, R>),
    Deleted(DeletedNode<R>),
}

impl<P, const R: usize> Node<P, R>
where
    P: PointInterface,
{
    pub fn new(point: P, edges: [u32; R]) -> Self {
        Self::Active(ActiveNode {
            point,
            edges,
            rc: 0,
        })
    }
}

pub trait Storage<P: PointInterface, const R: usize, T: Rng> {
    fn new() -> Self;

    fn get(&self, index: &u32) -> Option<Node<P, R>>;

    fn set(&mut self, index: u32, node: Node<P, R>) -> Option<Node<P, R>>;

    fn delete(&mut self, index: u32);

    fn remove(&mut self, index: u32);

    fn alloc(&mut self) -> u32;

    fn random_active_node(&self, rng: &mut T) -> (u32, ActiveNode<P, R>);
}

pub struct Graph<P: PointInterface, S: Storage<P, R, T>, const R: usize, T>
where
    T: Rng,
{
    storage: S,
    phantom1: std::marker::PhantomData<P>,
    phantom2: std::marker::PhantomData<T>,
}

impl<P, S, const R: usize, T> Graph<P, S, R, T>
where
    P: PointInterface,
    S: Storage<P, R, T>,
    T: Rng,
{
    pub fn new(init_point: P) -> Self {
        let mut storage = S::new();
        let init_node = Node::Active(ActiveNode {
            point: init_point,
            edges: [0; R],
            rc: 0,
        });
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
        let (entry_node_index, entry_node) = self.storage.random_active_node(rng);
        list.push((Dist(entry_node.point.distance(query)), entry_node_index));
        touched_nodes.insert(entry_node_index);

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

        loop {
            // Find an unvisited node that has smallest dist
            let Some((_, next_visit_node)) = list
                .iter()
                .find(|(_, node_index)| visited_nodes.insert(*node_index))
            else {
                // If all nodes in the list have been visited, searching is finished
                // The results are the fist k nodes in the list
                // debug!("search: break");
                break;
            };

            // Get next node from the storage
            let Some(Node::Active(working_node)) = self.storage.get(&next_visit_node) else {
                panic!("working node should be active")
            };

            let candidates: Vec<_> = working_node
                .edges
                .into_iter()
                .flat_map(|edge_node_index| {
                    // ActiveNode のみ抽出し、(edge_node_index, ActiveNode) の Vec を返す
                    self.collect_active_edges(edge_node_index)
                })
                .filter_map(|(edge_node_index, active_edge_node)| {
                    // If a distance of this edge node is already calculated, skip this node
                    if !touched_nodes.insert(edge_node_index) {
                        return None;
                    }
                    Some((edge_node_index, active_edge_node))
                })
                .map(|(edge_node_index, active_edge_node)| {
                    (
                        Dist(active_edge_node.point.distance(query)),
                        edge_node_index,
                    )
                })
                .sorted() // candidates should be order for merging to list
                .collect();

            // Merge the two lists by order
            list = merge_lists(list.clone(), candidates, l);
        }

        list
    }

    pub fn prune(&mut self, mut candidates: Vec<(Dist<f32>, u32)>, a: f32) -> [u32; R] {
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
        let new_node = Node::new(point, edges);
        let new_node_index = self.storage.alloc();
        self.storage.set(new_node_index, new_node);

        // Update graph edges: add the new node to its edge node's edges
        let edges = edges
            .to_vec()
            .into_iter()
            .sorted()
            .dedup()
            .collect::<Vec<_>>();
        debug!("node {new_node_index}: {:?}", edges);
        for edge_node_index in edges {
            self.add_edge(&edge_node_index, new_node_index);
        }

        debug!("\n");

        new_node_index
    }

    pub fn delete(&mut self, _index: u32, _a: f32) {
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
        todo!()
    }

    pub fn remove(&mut self, _index: u32, _a: f32) {
        todo!()
    }

    pub fn remove_list(&self) -> Vec<u32> {
        todo!()
    }

    fn collect_active_edges(&self, node_index: u32) -> Vec<(u32, ActiveNode<P, R>)> {
        match self.storage.get(&node_index) {
            Some(Node::Active(active_node)) => {
                // ActiveNode ならそのまま一つの要素のみ
                vec![(node_index, active_node)]
            }
            Some(Node::Deleted(deleted_node)) => {
                // DeletedNode なら、そこが持つ edges を辿って ActiveNode のみ取得
                deleted_node
                    .edges
                    .iter()
                    .filter_map(|&child_index| match self.storage.get(&child_index) {
                        Some(Node::Active(child_active)) => Some((child_index, child_active)),
                        _ => None,
                    })
                    .collect()
            }
            None => {
                vec![]
            }
        }
    }

    fn add_edge(&mut self, target_node_index: &u32, new_edge_node_index: u32) {
        debug!("add_edge: {target_node_index}, new_edge_node_index: {new_edge_node_index}");
        let Some(Node::Active(target_node)) = self.storage.get(target_node_index) else {
            panic!("")
        };
        let mut edges = target_node
            .edges
            .to_vec()
            .into_iter()
            .sorted()
            .dedup()
            .collect::<Vec<_>>();
        edges.push(new_edge_node_index);
        debug!("add_edge: {:?}", edges);

        let new_edges = if edges.len() > R {
            //
            let candidates: Vec<_> = edges
                .into_iter()
                .flat_map(|candidate_node_index| {
                    // ActiveNode のみ抽出し、(edge_node_index, ActiveNode) の Vec を返す
                    self.collect_active_edges(candidate_node_index)
                })
                .map(|(candidate_node_index, active_candidate_node)| {
                    (
                        Dist(target_node.point.distance(&active_candidate_node.point)),
                        candidate_node_index,
                    )
                })
                .sorted() // candidates should be order for merging to list
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
            // WIP: rcをどうするか？
            Node::new(target_node.point, new_edges),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::point::Point;

    use super::*;
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rustc_hash::FxHashSet;
    use std::collections::BTreeMap;

    const R: usize = 90;
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
        btree: BTreeMap<u32, Node<Point, R>>,
        index: u32,
    }

    impl Storage<Point, R, SmallRng> for TestStorage
    // where
    //     P: PointInterface,
    //     T: Rng,
    {
        fn new() -> Self {
            Self {
                btree: BTreeMap::new(),
                index: 0,
            }
        }

        fn get(&self, index: &u32) -> Option<Node<Point, R>> {
            self.btree.get(index).cloned()
        }

        fn set(&mut self, index: u32, node: Node<Point, R>) -> Option<Node<Point, R>> {
            self.btree.insert(index, node)
        }

        fn delete(&mut self, _index: u32) {
            todo!()
        }

        fn remove(&mut self, _index: u32) {
            todo!()
        }

        fn alloc(&mut self) -> u32 {
            self.index += 1;
            self.index
        }

        fn random_active_node(&self, rng: &mut SmallRng) -> (u32, ActiveNode<Point, R>) {
            let keys: Vec<_> = self.btree.keys().collect();

            loop {
                let Some(&random_key) = keys.get(rng.gen_range(0..keys.len())) else {
                    panic!("btree is empty")
                };

                let Some(Node::Active(active_node)) = self.get(random_key) else {
                    continue;
                };

                return (*random_key, active_node);
            }

            // while let Some(Node::Active(active_node))

            // *random_key
        }
    }

    const DIM: usize = 384;

    #[test]
    fn test_get() {
        let mut rng: SmallRng = SmallRng::from_entropy();
        let mut graph: Graph<Point, TestStorage, R, SmallRng> = Graph::new(gen_point(&mut rng));

        env_logger::init();

        let iter_count = 500;

        let test_points: Vec<Point> = (0..iter_count)
            .into_iter()
            .map(|_| gen_point(&mut rng))
            .collect();

        for p in test_points.iter() {
            graph.insert(p.clone(), 2.0, L, &mut rng);
        }

        let hit_counts: f32 = test_points
            .par_iter()
            .map_init(
                || SmallRng::from_entropy(),
                |rng, p| {
                    let p = p.clone();
                    let ground_truth: Vec<(Dist<f32>, u32)> = test_points
                        .iter()
                        .enumerate()
                        .map(|(i, q)| (Dist(p.distance(q)), (i + 1) as u32))
                        .sorted()
                        .collect();

                    let result = graph.search(&p, L, rng);

                    debug!("{:?}, {:?}", ground_truth[0], result[0]);

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

    fn gen_point(rng: &mut SmallRng) -> Point {
        let new_point: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
        let new_point = normalize_to_unit_length(new_point);
        Point(new_point)
    }
}
