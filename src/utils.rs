pub fn diff_ids(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let mut result = Vec::new();
    let mut a_idx = 0;
    let mut b_idx = 0;

    while a_idx < a.len() && b_idx < b.len() {
        if a[a_idx] == b[b_idx] {
            a_idx += 1; // Skip common elements
            b_idx += 1;
        } else if a[a_idx] < b[b_idx] {
            // Elements present only in a
            result.push(a[a_idx]);
            a_idx += 1;
        } else {
            // Ignore elements that exist only in b
            b_idx += 1;
        }
    }

    // Add the remaining elements of a (since they do not exist in b)
    while a_idx < a.len() {
        result.push(a[a_idx]);
        a_idx += 1;
    }

    result
}

pub fn intersect_ids(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let mut result = Vec::new();
    let mut a_idx = 0;
    let mut b_idx = 0;

    while a_idx < a.len() && b_idx < b.len() {
        if a[a_idx] == b[b_idx] {
            result.push(a[a_idx]);
            a_idx += 1;
            b_idx += 1;
        } else if a[a_idx] < b[b_idx] {
            a_idx += 1;
        } else {
            b_idx += 1;
        }
    }

    result
}

pub fn sort_list_by_dist(list: &mut Vec<(f32, u32, bool)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

pub fn sort_list_by_dist_v1(list: &mut Vec<(f32, u32)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

pub fn sort_list_by_dist_v3(list: &mut Vec<(f32, u32, bool, Vec<u32>)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

pub fn is_contained_in(i: &u32, vec: &Vec<(f32, u32)>) -> bool {
    !vec.iter()
        .filter(|(_, id)| *id == *i)
        .collect::<Vec<&(f32, u32)>>()
        .is_empty()
}

pub fn insert_id(value: u32, vec: &mut Vec<u32>) {
    match vec.binary_search(&value) {
        Ok(_index) => { // If already exsits
        }
        Err(index) => {
            vec.insert(index, value);
        }
    }
}

pub fn insert_dist(value: (f32, u32), vec: &mut Vec<(f32, u32)>) {
    match vec.binary_search_by(|probe| {
        probe
            .0
            .partial_cmp(&value.0)
            .unwrap_or(std::cmp::Ordering::Less)
    }) {
        Ok(index) => {
            // identify a range of groups of elements with the same f32 value
            let mut start = index;
            while start > 0 && vec[start - 1].0 == value.0 {
                start -= 1;
            }
            let mut end = index;
            while end < vec.len() - 1 && vec[end + 1].0 == value.0 {
                end += 1;
            }

            // Check for elements with the same usize value within the specified range
            if !(start..=end).any(|i| vec[i].1 == value.1) {
                vec.insert(index, value);
            }
        }
        Err(index) => {
            vec.insert(index, value);
        }
    };
}

use crate::graph_store::GraphHeader;

type Vector = [f32];
type Edges = [u32];
type SerializedVector = [u8];
type SerializedEdges = [u8];

pub fn serialize_node(vector: &Vector, edges: &Edges) -> Vec<u8> {
    let serialize_vector: &[u8] = serialize_vector(vector);
    let (serialize_edges, serialize_edges_len) = serialize_edges(edges);
    let mut combined = Vec::with_capacity(serialize_vector.len() + serialize_edges.len());
    combined.extend_from_slice(serialize_vector);
    combined.extend_from_slice(&serialize_edges_len);
    combined.extend_from_slice(serialize_edges);

    combined
}

pub fn serialize_vector(vector: &Vector) -> &SerializedVector {
    let serialize_vector: &SerializedVector = bytemuck::cast_slice(vector)
        .try_into()
        .expect("Failed to try into &[u8; DIM*4]");
    serialize_vector
}

pub fn serialize_edges(edges: &Edges) -> (&SerializedEdges, [u8; 4]) {
    let serialize_edges_len = (edges.len() as u32).to_le_bytes();
    let serialize_edges: &SerializedEdges = bytemuck::cast_slice(edges)
        .try_into()
        .expect("Failed to try into &[u8; DIGREE*4]");
    (serialize_edges, serialize_edges_len)
}

pub fn deserialize_node(
    bytes: &[u8],
    vector_dim: usize,
    edge_max_digree: usize,
) -> (&Vector, &Edges) {
    let vector_end = vector_dim * 4;
    let edges_start = vector_end + 4;
    let edges_len = u32::from_le_bytes(bytes[vector_end..edges_start].try_into().unwrap()) as usize;
    let edges_end = edges_start + std::cmp::min(edge_max_digree, edges_len) * 4;

    let vector: &Vector = deserialize_vector(&bytes[..vector_end]);
    let edges: &Edges = deserialize_edges(&bytes[edges_start..edges_end]);

    (vector, edges)
}

pub fn deserialize_vector(serialize_vector: &SerializedVector) -> &Vector {
    let vector: &Vector = bytemuck::try_cast_slice(serialize_vector)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    vector
}

pub fn deserialize_edges(serialize_edges: &SerializedEdges) -> &Edges {
    let edges: &Edges = bytemuck::try_cast_slice(serialize_edges)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    edges
}

pub fn node_byte_size(vector_dim: usize, max_edge_degrees: usize) -> usize {
    let node_byte_size = vector_dim * 4 + max_edge_degrees * 4 + 4;
    node_byte_size
}

pub fn file_byte_size(node_byte_size: usize, num_vectors: usize) -> usize {
    std::mem::size_of::<GraphHeader>() + num_vectors * node_byte_size
}
