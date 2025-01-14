use super::{GraphInterface as VGraph, PointInterface as VPoint, *};

use serde::{Deserialize, Serialize};

use anyhow::Result;
use std::fs::File;
use std::io::{Read, Write};

use crate::graph_store::GraphStore;
// use crate::point::Point;
use crate::storage::StorageTrait;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Point(Vec<f32>);

impl PointInterface for Point {
    fn distance(&self, other: &Self) -> f32 {
        -cosine_similarity(&self, &other) + 1.0
    }

    fn add(&self, other: &Self) -> Self {
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .zip(other.to_f32_vec())
                .map(|(x, y)| x + y)
                .collect(),
        )
    }
    fn div(&self, divisor: &usize) -> Self {
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .map(|v| v / *divisor as f32)
                .collect(),
        )
    }

    fn to_f32_vec(&self) -> Vec<f32> {
        self.0.iter().copied().collect()
    }
    fn from_f32_vec(a: Vec<f32>) -> Self {
        Point(a.into_iter().collect())
    }
}

fn dot_product(vec1: &Point, vec2: &Point) -> f32 {
    assert_eq!(vec1.0.len(), vec2.0.len());
    let dim: usize = vec1.0.len();
    let mut result = 0.0;
    for i in 0..dim {
        result += vec1.0[i] * vec2.0[i];
    }
    result
}

fn norm(vec: &Point) -> f32 {
    let dim = vec.0.len();
    let mut result = 0.0;
    for i in 0..dim {
        result += vec.0[i] * vec.0[i];
    }
    result.sqrt()
}

fn cosine_similarity(vec1: &Point, vec2: &Point) -> f32 {
    let dot = dot_product(vec1, vec2);
    let norm1 = norm(vec1);
    let norm2 = norm(vec2);

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    dot / (norm1 * norm2)
}

#[derive(Clone)]
pub struct Graph<S: StorageTrait> {
    size_l: usize,
    size_r: usize,
    size_a: f32,

    graph_store: GraphStore<S>,
    start_node_index: u32,
}

impl<S: StorageTrait> Graph<S> {
    pub fn new(graph_store: GraphStore<S>) -> Self {
        Self {
            size_l: 125,
            size_r: graph_store.max_edge_degrees(),
            size_a: 2.0,
            start_node_index: graph_store.start_id() as u32,
            graph_store,
        }
    }

    pub fn set_start_node_index(&mut self, index: u32) {
        self.start_node_index = index;
    }

    pub fn set_size_l(&mut self, size_l: usize) {
        self.size_l = size_l;
    }
}

impl<S: StorageTrait, P: PointInterface> GraphInterface<P> for Graph<S> {
    fn alloc(&mut self, _point: P) -> u32 {
        todo!()
    }

    fn free(&mut self, _id: &u32) {
        todo!()
    }

    fn cemetery(&self) -> Vec<u32> {
        vec![]
    }

    fn clear_cemetery(&mut self) {
        todo!()
    }

    fn backlink(&self, _id: &u32) -> Vec<u32> {
        todo!()
    }

    fn get(&mut self, node_index: &u32) -> (P, Vec<u32>) {
        let store_index = node_index;
        let (vector, edges) = self.graph_store.read_node(&store_index).unwrap();

        (P::from_f32_vec(vector), edges)
    }

    fn size_l(&self) -> usize {
        self.size_l
    }

    fn size_r(&self) -> usize {
        self.size_r
    }

    fn size_a(&self) -> f32 {
        self.size_a
    }

    fn start_id(&self) -> u32 {
        self.start_node_index
    }

    fn overwirte_out_edges(&mut self, _id: &u32, _edges: Vec<u32>) {
        todo!()
    }
}
