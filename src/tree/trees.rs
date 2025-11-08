use super::vec_tree::{Traversable, TreeNode, VecTree};
use crate::arch::CpuFeatures;
use crate::loader::{ModelError, ModelLoader, XGBoostParser};
use crate::objective::Objective;
use crate::predicates::{Condition, Predicate};
use crate::tree::{FeatureTreeError, FeatureType};
use arrow::array::{Array, ArrayRef, BooleanArray, Float32Array, Float32Builder, Int64Array};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;
use std::sync::OnceLock;

pub type VecTreeNodes = VecTree<TreeNode>;

#[derive(Debug)]
enum PruneAction {
    Keep,
    PruneLeft,
    PruneRight,
}

enum NodeDefinition {
    Leaf {
        weight: f32,
    },
    Split {
        feature_index: i32,
        default_left: bool,
        split_value: f32,
        left: usize,
        right: usize,
    },
}

impl VecTreeNodes {
    #[inline(always)]
    pub fn predict(&self, features: &[f32]) -> f32 {
        static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
        let cpu_features = CPU_FEATURES.get_or_init(CpuFeatures::new);

        let mut current_idx = self.get_root_index();
        let nodes = &self.nodes;

        while let Some(current) = nodes.get(current_idx) {
            if current.is_leaf() {
                return current.weight();
            }

            let feature_idx = current.feature_index() as usize;
            let split_value = unsafe { *features.get_unchecked(feature_idx) };

            let left_child = unsafe { nodes.get_unchecked(current.left()) };

            cpu_features.prefetch(left_child as *const _);

            let go_right = if split_value.is_nan() {
                !current.default_left()
            } else {
                split_value >= current.split_value()
            };

            current_idx = if go_right {
                current.right()
            } else {
                current.left()
            };
        }

        0.0
    }

    pub fn depth(&self) -> usize {
        fn recursive_depth(tree: &VecTreeNodes, node: &TreeNode) -> usize {
            if node.is_leaf() {
                0
            } else {
                1 + tree
                    .get_left_child(node)
                    .map(|n| recursive_depth(tree, n))
                    .unwrap_or(0)
                    .max(
                        tree.get_right_child(node)
                            .map(|n| recursive_depth(tree, n))
                            .unwrap_or(0),
                    )
            }
        }

        self.get_node(self.get_root_index())
            .map(|root| recursive_depth(self, root))
            .unwrap_or(0)
    }

    pub fn num_nodes(&self) -> usize {
        fn count_reachable_nodes(tree: &VecTreeNodes, node: &TreeNode) -> usize {
            if node.is_leaf() {
                1
            } else {
                1 + tree
                    .get_left_child(node)
                    .map(|n| count_reachable_nodes(tree, n))
                    .unwrap_or(0)
                    + tree
                        .get_right_child(node)
                        .map(|n| count_reachable_nodes(tree, n))
                        .unwrap_or(0)
            }
        }

        self.get_node(self.get_root_index())
            .map(|root| count_reachable_nodes(self, root))
            .unwrap_or(0)
    }
    #[inline]
    pub fn prune(&self, predicate: &Predicate, feature_names: &[String]) -> Option<VecTreeNodes> {
        if self.is_empty() {
            return None;
        }

        let mut new_tree = VecTreeNodes::new();

        fn evaluate_node(
            node: &TreeNode,
            feature_index: usize,
            feature_names: &[String],
            predicate: &Predicate,
        ) -> PruneAction {
            if node.is_leaf() {
                return PruneAction::Keep;
            }

            if let Some(feature_name) = feature_names.get(feature_index) {
                if let Some(conditions) = predicate.conditions.get(feature_name) {
                    for condition in conditions {
                        match condition {
                            Condition::LessThan(value) => {
                                if node.should_prune_right(*value) {
                                    return PruneAction::PruneRight;
                                }
                            }
                            Condition::GreaterThanOrEqual(value) => {
                                if node.should_prune_left(*value) {
                                    return PruneAction::PruneLeft;
                                }
                            }
                        }
                    }
                }
            }
            PruneAction::Keep
        }

        fn prune_recursive(
            old_tree: &VecTreeNodes,
            new_tree: &mut VecTreeNodes,
            node_idx: usize,
            feature_names: &[String],
            predicate: &Predicate,
        ) -> Option<usize> {
            let node = old_tree.get_node(node_idx)?;
            let feature_index = node.feature_index() as usize;

            match evaluate_node(node, feature_index, feature_names, predicate) {
                PruneAction::Keep => {
                    let new_idx = new_tree.nodes.len();
                    new_tree.nodes.push(node.clone());

                    if !node.is_leaf() {
                        let left_idx = prune_recursive(
                            old_tree,
                            new_tree,
                            node.left(),
                            feature_names,
                            predicate,
                        );

                        let right_idx = prune_recursive(
                            old_tree,
                            new_tree,
                            node.right(),
                            feature_names,
                            predicate,
                        );

                        if let Some(left_idx) = left_idx {
                            new_tree.connect_left(new_idx, left_idx).ok()?;
                        }
                        if let Some(right_idx) = right_idx {
                            new_tree.connect_right(new_idx, right_idx).ok()?;
                        }
                    }

                    Some(new_idx)
                }
                PruneAction::PruneLeft => {
                    prune_recursive(old_tree, new_tree, node.right(), feature_names, predicate)
                }
                PruneAction::PruneRight => {
                    prune_recursive(old_tree, new_tree, node.left(), feature_names, predicate)
                }
            }
        }

        let root_idx = self.get_root_index();
        prune_recursive(self, &mut new_tree, root_idx, feature_names, predicate)?;

        Some(new_tree)
    }

    fn update_feature_metadata(&mut self, feature_index_map: &HashMap<usize, usize>) {
        self.update_feature_indices(feature_index_map);
    }

    pub fn builder() -> FeatureTreeBuilder {
        FeatureTreeBuilder::new()
    }

    fn from_nodes(nodes: Vec<NodeDefinition>) -> Result<Self, FeatureTreeError> {
        if nodes.is_empty() {
            return Err(FeatureTreeError::InvalidStructure("Empty tree".to_string()));
        }
        if nodes.is_empty() {
            return Err(FeatureTreeError::InvalidStructure("Empty tree".to_string()));
        }

        let mut vec_tree = VecTreeNodes::new();
        let mut node_map: HashMap<usize, usize> = HashMap::new();
        for (builder_idx, node_def) in nodes.iter().enumerate() {
            let tree_node = match node_def {
                NodeDefinition::Split {
                    feature_index,
                    split_value,
                    default_left,
                    ..
                } => TreeNode::new_split(*feature_index, *split_value, *default_left),
                NodeDefinition::Leaf { weight } => TreeNode::new_leaf(*weight),
            };

            let tree_idx = if builder_idx == 0 {
                vec_tree.add_root(tree_node)
            } else {
                vec_tree.add_orphan_node(tree_node)
            };

            node_map.insert(builder_idx, tree_idx);
        }

        for (builder_idx, node_def) in nodes.iter().enumerate() {
            if let NodeDefinition::Split { left, right, .. } = node_def {
                let parent_idx = node_map[&builder_idx];
                let left_idx = node_map[left];
                let right_idx = node_map[right];

                vec_tree.connect_left(parent_idx, left_idx).map_err(|_| {
                    FeatureTreeError::InvalidStructure("Invalid left child connection".to_string())
                })?;
                vec_tree.connect_right(parent_idx, right_idx).map_err(|_| {
                    FeatureTreeError::InvalidStructure("Invalid right child connection".to_string())
                })?;
            }
        }

        if !vec_tree.validate_connections() {
            return Err(FeatureTreeError::InvalidStructure(
                "Tree has disconnected nodes".into(),
            ));
        }

        Ok(vec_tree)
    }
}

pub struct FeatureTreeBuilder {
    split_indices: Vec<i32>,
    split_conditions: Vec<f32>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    base_weights: Vec<f32>,
    default_left: Vec<bool>,
}

impl FeatureTreeBuilder {
    pub fn new() -> Self {
        Self {
            split_indices: Vec::new(),
            split_conditions: Vec::new(),
            left_children: Vec::new(),
            right_children: Vec::new(),
            base_weights: Vec::new(),
            default_left: Vec::new(),
        }
    }

    pub fn split_indices(self, indices: Vec<i32>) -> Self {
        Self {
            split_indices: indices,
            ..self
        }
    }

    pub fn split_conditions(self, conditions: Vec<f32>) -> Self {
        Self {
            split_conditions: conditions,
            ..self
        }
    }

    pub fn children(self, left: Vec<u32>, right: Vec<u32>) -> Self {
        Self {
            left_children: left,
            right_children: right,
            ..self
        }
    }

    pub fn base_weights(self, weights: Vec<f32>) -> Self {
        Self {
            base_weights: weights,
            ..self
        }
    }

    pub fn default_left(self, indices: Vec<bool>) -> Self {
        Self {
            default_left: indices,
            ..self
        }
    }

    pub fn build(self) -> Result<VecTreeNodes, FeatureTreeError> {
        let node_count = self.split_indices.len();
        if self.split_conditions.len() != node_count
            || self.left_children.len() != node_count
            || self.right_children.len() != node_count
            || self.base_weights.len() != node_count
        {
            return Err(FeatureTreeError::InvalidStructure(
                "Inconsistent array lengths in tree definition".to_string(),
            ));
        }

        let mut nodes = Vec::with_capacity(node_count);
        for i in 0..node_count {
            let is_leaf = self.left_children[i] == u32::MAX;
            let node = if is_leaf {
                NodeDefinition::Leaf {
                    weight: self.base_weights[i],
                }
            } else {
                NodeDefinition::Split {
                    feature_index: self.split_indices[i],
                    split_value: self.split_conditions[i],
                    left: self.left_children[i] as usize,
                    right: self.right_children[i] as usize,
                    default_left: self.default_left[i],
                }
            };
            nodes.push(node);
        }

        VecTreeNodes::from_nodes(nodes)
    }
}

impl Default for FeatureTreeBuilder {
    fn default() -> Self {
        FeatureTreeBuilder::new()
    }
}

#[derive(Debug, Clone)]
pub struct PredictorConfig {
    pub row_chunk_size: usize,
    pub tree_chunk_size: usize,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            row_chunk_size: 8,
            tree_chunk_size: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GradientBoostedDecisionTrees {
    pub trees: Vec<VecTreeNodes>,
    pub feature_names: Arc<Vec<String>>,
    pub base_score: f32,
    pub feature_types: Arc<Vec<FeatureType>>,
    pub objective: Objective,
    pub config: PredictorConfig,
    pub required_features: HashSet<usize>,
}

//SAFETY: Send + Sync as all fields are Send + Sync
unsafe impl Send for GradientBoostedDecisionTrees {}
unsafe impl Sync for GradientBoostedDecisionTrees {}

impl Default for GradientBoostedDecisionTrees {
    fn default() -> Self {
        GradientBoostedDecisionTrees {
            trees: vec![],
            feature_names: Arc::new(vec![]),
            feature_types: Arc::new(vec![]),
            base_score: 0.0,
            objective: Objective::SquaredError,
            config: PredictorConfig::default(),
            required_features: HashSet::new(),
        }
    }
}

impl GradientBoostedDecisionTrees {
    pub fn config(&self) -> &PredictorConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PredictorConfig) {
        self.config = config;
    }

    pub fn get_required_features(&self) -> &HashSet<usize> {
        &self.required_features
    }

    fn collect_required_features(trees: &[VecTreeNodes]) -> HashSet<usize> {
        let mut required_features = HashSet::new();

        for tree in trees {
            if let Some(root) = tree.get_node(tree.get_root_index()) {
                let mut stack = vec![root];

                while let Some(node) = stack.pop() {
                    if !node.is_leaf() {
                        required_features.insert(node.feature_index() as usize);

                        if let Some(right) = tree.get_right_child(node) {
                            stack.push(right);
                        }
                        if let Some(left) = tree.get_left_child(node) {
                            stack.push(left);
                        }
                    }
                }
            }
        }
        required_features
    }

    pub fn predict_batches(&self, batches: &[RecordBatch]) -> Result<Float32Array, ArrowError> {
        if batches.len() == 1 {
            let required_columns: Vec<ArrayRef> = batches[0]
                .columns()
                .iter()
                .enumerate()
                .filter(|(i, _)| self.required_features.contains(i))
                .map(|(_, col)| col.clone())
                .collect();
            return self.predict_arrays(&required_columns);
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let mut builder = Float32Builder::with_capacity(total_rows);

        for batch in batches {
            let required_columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .enumerate()
                .filter(|(i, _)| self.required_features.contains(i))
                .map(|(_, col)| col.clone())
                .collect();

            let predictions = self.predict_arrays(&required_columns)?;
            builder.append_slice(predictions.values());
        }
        Ok(builder.finish())
    }

    #[inline]
    pub fn predict_arrays(&self, feature_arrays: &[ArrayRef]) -> Result<Float32Array, ArrowError> {
        let features = self.extract_features(feature_arrays)?;
        self.predict_internal(&features)
    }

    #[inline]
    fn predict_internal(&self, features: &[Vec<f32>]) -> Result<Float32Array, ArrowError> {
        let (num_rows, num_features) = (features[0].len(), features.len());

        let predictions: Vec<f32> = (0..num_rows)
            .into_par_iter()
            .chunks(self.config.row_chunk_size)
            .fold(
                || Vec::with_capacity(self.config.row_chunk_size),
                |mut chunk_results, row_indices| {
                    let mut row_features = vec![0.0; num_features];
                    let mut chunk_scores = vec![self.base_score; row_indices.len()];

                    for tree_chunk in self.trees.chunks(self.config.tree_chunk_size) {
                        for (chunk_idx, &row_idx) in row_indices.iter().enumerate() {
                            // Unroll by 8 for better vectorization
                            let mut j = 0;
                            while j + 8 <= num_features {
                                row_features[j] = features[j][row_idx];
                                row_features[j + 1] = features[j + 1][row_idx];
                                row_features[j + 2] = features[j + 2][row_idx];
                                row_features[j + 3] = features[j + 3][row_idx];
                                row_features[j + 4] = features[j + 4][row_idx];
                                row_features[j + 5] = features[j + 5][row_idx];
                                row_features[j + 6] = features[j + 6][row_idx];
                                row_features[j + 7] = features[j + 7][row_idx];
                                j += 8;
                            }
                            while j < num_features {
                                row_features[j] = features[j][row_idx];
                                j += 1;
                            }

                            let tree_chunk_score: f32 = tree_chunk
                                .iter()
                                .map(|tree| tree.predict(&row_features))
                                .sum();
                            chunk_scores[chunk_idx] += tree_chunk_score;
                        }
                    }

                    chunk_results.extend(
                        chunk_scores
                            .into_iter()
                            .map(|score| self.objective.compute_score(score)),
                    );

                    chunk_results
                },
            )
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });

        let mut builder = Float32Builder::with_capacity(predictions.len());
        builder.append_slice(&predictions);
        Ok(builder.finish())
    }

    #[inline]
    fn extract_features(&self, feature_arrays: &[ArrayRef]) -> Result<Vec<Vec<f32>>, ArrowError> {
        let num_rows = feature_arrays[0].len();
        let mut feature_values = Vec::with_capacity(feature_arrays.len());

        for array in feature_arrays.iter() {
            let values = match array.data_type() {
                DataType::Float32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError("Expected Float32Array".into())
                        })?;

                    if array.nulls().is_none() {
                        array.values().to_vec()
                    } else {
                        let values_slice = array.values();
                        let null_bitmap = array.nulls().unwrap();

                        values_slice
                            .iter()
                            .enumerate()
                            .map(|(i, &val)| {
                                if null_bitmap.is_null(i) {
                                    f32::NAN
                                } else {
                                    val
                                }
                            })
                            .collect()
                    }
                }
                DataType::Int64 => {
                    let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                        ArrowError::InvalidArgumentError("Expected Int64Array".into())
                    })?;

                    if array.nulls().is_none() {
                        array.values().iter().map(|&x| x as f32).collect()
                    } else {
                        let values_slice = array.values();
                        let null_bitmap = array.nulls().unwrap();

                        values_slice
                            .iter()
                            .enumerate()
                            .map(|(i, &val)| {
                                if null_bitmap.is_null(i) {
                                    f32::NAN
                                } else {
                                    val as f32
                                }
                            })
                            .collect()
                    }
                }
                DataType::Boolean => {
                    let array = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError("Expected BooleanArray".into())
                        })?;

                    if array.nulls().is_none() {
                        array
                            .values()
                            .iter()
                            .map(|x| if x { 1.0 } else { 0.0 })
                            .collect()
                    } else {
                        let null_bitmap = array.nulls().unwrap();

                        (0..num_rows)
                            .map(|i| {
                                if null_bitmap.is_null(i) {
                                    f32::NAN
                                } else if array.value(i) {
                                    1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    }
                }
                actual_type => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "Unsupported data type: {:?}",
                        actual_type
                    )));
                }
            };

            feature_values.push(values);
        }

        Ok(feature_values)
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn tree_depths(&self) -> Vec<usize> {
        self.trees.iter().map(|tree| tree.depth()).collect()
    }

    pub fn prune(&self, predicate: &Predicate) -> Self {
        let pruned_trees: Vec<VecTreeNodes> = self
            .trees
            .iter()
            .filter_map(|tree| tree.prune(predicate, &self.feature_names))
            .collect();

        let required_features = Self::collect_required_features(&pruned_trees);

        let mut model = GradientBoostedDecisionTrees {
            trees: pruned_trees,
            feature_names: self.feature_names.clone(),
            feature_types: self.feature_types.clone(),
            base_score: self.base_score,
            objective: self.objective.clone(),
            config: self.config.clone(),
            required_features,
        };

        model.update_feature_metadata();
        model
    }

    fn update_feature_metadata(&mut self) {
        if self.required_features.len() != self.feature_names.len() {
            let mut required_indices: Vec<_> = self.required_features.iter().copied().collect();
            required_indices.sort();

            let feature_index_map: HashMap<usize, usize> = required_indices
                .into_iter()
                .enumerate()
                .map(|(new_idx, global_idx)| (global_idx, new_idx))
                .collect();

            for tree in &mut self.trees {
                tree.update_feature_metadata(&feature_index_map);
            }
        }
    }
}

impl std::fmt::Display for GradientBoostedDecisionTrees {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let depths = self.tree_depths();
        let avg_depth = depths.iter().sum::<usize>() as f64 / depths.len() as f64;
        let max_depth = depths.iter().max().unwrap_or(&0);
        let total_nodes = self
            .trees
            .iter()
            .map(|tree| tree.num_nodes())
            .sum::<usize>();

        writeln!(f, "Total number of trees: {}", self.num_trees())?;
        writeln!(f, "Tree depths: {:?}", depths)?;
        writeln!(f, "Average tree depth: {:.2}", avg_depth)?;
        writeln!(f, "Max tree depth: {}", max_depth)?;
        writeln!(f, "Total number of nodes: {}", total_nodes)
    }
}

impl ModelLoader for GradientBoostedDecisionTrees {
    fn json_load(path: &str) -> Result<Self, ModelError> {
        let data = fs::read_to_string(path).map_err(|e| ModelError::IoError(e.to_string()))?;
        let result: Value =
            serde_json::from_str(&data).map_err(|e| ModelError::IoError(e.to_string()))?;
        let model = Self::json_loads(&result)?;
        Ok(model)
    }

    fn json_loads(json: &Value) -> Result<Self, ModelError> {
        let objective_type = XGBoostParser::parse_objective(json)?;
        let (feature_names, feature_types) = XGBoostParser::parse_feature_metadata(json)?;
        let mut base_score = XGBoostParser::parse_base_score(json)?;
        // logistic objectives, base_score is stored as probability
        // https://stackoverflow.com/questions/78818308/how-does-xgboost-calculate-base-score
        // TODO: do we need special handling for reg:squarederror?
        if objective_type == Objective::Logistic {
            base_score = (base_score / (1.0 - base_score)).ln();
        }
        let trees_json = XGBoostParser::parse_trees(json)?;

        let trees = trees_json
            .iter()
            .map(|tree_json| {
                let arrays = XGBoostParser::parse_tree_arrays(tree_json)?;

                let tree = FeatureTreeBuilder::new()
                    .split_indices(arrays.split_indices)
                    .split_conditions(arrays.split_conditions)
                    .children(arrays.left_children, arrays.right_children)
                    .base_weights(arrays.base_weights)
                    .default_left(arrays.default_left)
                    .build()
                    .map_err(ModelError::from)?;
                Ok::<VecTreeNodes, ModelError>(tree)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let required_features = Self::collect_required_features(&trees);

        let mut model = Self {
            base_score,
            trees,
            feature_names: Arc::new(feature_names),
            feature_types: Arc::new(feature_types),
            objective: objective_type,
            config: PredictorConfig::default(),
            required_features,
        };

        // Update feature indices and metadata
        model.update_feature_metadata();

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::datatypes::Field;
    use arrow::datatypes::Schema;
    use std::sync::Arc;

    fn create_simple_tree() -> Result<VecTreeNodes, FeatureTreeError> {
        // Creates a simple decision tree:
        //          [age < 30]
        //         /          \
        //    [-1.0]        [income < 50k]
        //                  /           \
        //               [0.0]         [1.0]

        FeatureTreeBuilder::new()
            .split_indices(vec![0, -1, 1, -1, -1])
            .split_conditions(vec![30.0, 0.0, 50000.0, 0.0, 0.0])
            .children(
                vec![1, u32::MAX, 3, u32::MAX, u32::MAX],
                vec![2, u32::MAX, 4, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, -1.0, 0.0, 0.0, 1.0])
            .default_left(vec![true, false, false, false, false])
            .build()
    }

    fn create_sample_tree() -> VecTreeNodes {
        // Create a simple tree:
        //          [feature0 < 0.5]
        //         /               \
        //    [-1.0]               [1.0]

        FeatureTreeBuilder::new()
            .split_indices(vec![0, -1, -1])
            .split_conditions(vec![0.5, 0.0, 0.0])
            .children(vec![1, u32::MAX, u32::MAX], vec![2, u32::MAX, u32::MAX])
            .base_weights(vec![0.0, -1.0, 1.0])
            .default_left(vec![false, false, false])
            .build()
            .unwrap()
    }

    fn create_sample_tree_deep() -> VecTreeNodes {
        // Create a deeper tree:
        //                    [feature0 < 0.5]
        //                   /               \
        //      [feature1 < 0.3]            [feature1 < 0.6]
        //     /               \            /               \
        // [feature2 < 0.7]    [-1.0]    [1.0]       [feature2 < 0.8]
        //   /        \                                /            \
        // [-2.0]    [2.0]                          [2.0]         [3.0]
        FeatureTreeBuilder::new()
            .split_indices(vec![0, 1, 2, -1, -1, -1, 1, -1, 2, -1, -1])
            .split_conditions(vec![0.5, 0.3, 0.7, 0.0, 0.0, 0.0, 0.6, 0.0, 0.8, 0.0, 0.0])
            .children(
                vec![
                    1,
                    3,
                    4,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                    7,
                    u32::MAX,
                    9,
                    u32::MAX,
                    u32::MAX,
                ],
                vec![
                    6,
                    2,
                    5,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                    8,
                    u32::MAX,
                    10,
                    u32::MAX,
                    u32::MAX,
                ],
            )
            .base_weights(vec![
                0.0, 0.0, 0.0, -2.0, 2.0, -1.0, 0.0, 1.0, 0.0, 2.0, 3.0,
            ])
            .default_left(vec![
                true, true, true, false, false, true, false, false, true, false, false,
            ])
            .build()
            .unwrap()
    }

    fn create_sample_record_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("age", DataType::Float32, false),
            Field::new("income", DataType::Float32, false),
        ]);

        let age_array = Float32Array::from(vec![25.0, 35.0, 35.0, 28.0]);
        let income_array = Float32Array::from(vec![30000.0, 60000.0, 40000.0, 35000.0]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(age_array), Arc::new(income_array)],
        )
        .unwrap()
    }

    #[test]
    fn test_feature_tree_basic_predictions() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;

        // Test various prediction paths
        assert_eq!(tree.predict(&[25.0, 30000.0]), -1.0); // young age -> left path
        assert_eq!(tree.predict(&[35.0, 60000.0]), 1.0); // older age, high income -> right path
        assert_eq!(tree.predict(&[35.0, 40000.0]), 0.0); // older age, low income -> middle path

        Ok(())
    }

    #[test]
    fn test_tree_depth_and_size() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;

        assert_eq!(tree.depth(), 2); // Root -> Income split -> Leaf
        assert_eq!(tree.num_nodes(), 5); // 2 internal nodes + 3 leaf nodes

        Ok(())
    }

    #[test]
    fn test_gbdt_basic() -> Result<(), FeatureTreeError> {
        let tree1 = create_simple_tree()?;
        let tree2 = create_simple_tree()?; // Using same tree structure for simplicity

        let gbdt = GradientBoostedDecisionTrees {
            trees: vec![tree1, tree2],
            feature_names: Arc::new(vec!["age".to_string(), "income".to_string()]),
            feature_types: Arc::new(vec![FeatureType::Float, FeatureType::Float]),
            base_score: 0.5,
            objective: Objective::SquaredError,
            config: PredictorConfig::default(),
            required_features: HashSet::from([0, 1]),
        };

        let batch = create_sample_record_batch();
        let predictions = gbdt.predict_batches(&[batch]).unwrap();

        assert_eq!(predictions.len(), 4);

        let expected_values: Vec<f32> = vec![-1.0, 1.0, 0.0, -1.0]
            .into_iter()
            .map(|x| 0.5 + 2.0 * x)
            .collect();

        for (i, &expected) in expected_values.iter().enumerate() {
            assert!((predictions.value(i) - expected).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_pruning() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;

        let mut conditions = HashMap::new();
        conditions.insert("age".to_string(), vec![Condition::GreaterThanOrEqual(30.0)]);
        let predicate = Predicate { conditions };

        let feature_names = vec!["age".to_string(), "income".to_string()];

        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();

        let test_cases = vec![
            vec![25.0, 30000.0], // Would have gone left in original tree
            vec![35.0, 60000.0],
            vec![35.0, 40000.0],
        ];

        for test_case in test_cases {
            let prediction = pruned_tree.predict(&test_case);
            assert!(
                prediction == 0.0 || prediction == 1.0,
                "Prediction {} should only follow right path",
                prediction
            );
        }

        Ok(())
    }
    #[test]
    fn test_tree_prune() {
        let tree = create_sample_tree();
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.49));
        let pruned_tree = tree.prune(&predicate, &["feature0".to_string()]).unwrap();
        assert_eq!(pruned_tree.nodes.len(), 1);
        assert_eq!(pruned_tree.get_node(0).unwrap().weight(), -1.0);
    }

    #[test]
    fn test_tree_prune_deep() {
        let tree = create_sample_tree_deep();
        let feature_names = [
            "feature0".to_string(),
            "feature1".to_string(),
            "feature2".to_string(),
        ];

        // Test case 1: Prune right subtree of root
        let mut predicate1 = Predicate::new();
        predicate1.add_condition("feature1".to_string(), Condition::LessThan(0.30));
        let pruned_tree1 = tree.prune(&predicate1, &feature_names).unwrap();
        assert_eq!(pruned_tree1.predict(&[0.6, 0.75, 0.8]), 1.0);

        let mut predicate2 = Predicate::new();
        predicate2.add_condition("feature2".to_string(), Condition::LessThan(0.70));
        let pruned_tree2 = tree.prune(&predicate2, &feature_names).unwrap();
        assert_eq!(pruned_tree2.predict(&[0.4, 0.2, 0.8]), -2.0);

        let mut predicate3 = Predicate::new();
        predicate3.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.50));
        let pruned_tree3 = tree.prune(&predicate3, &feature_names).unwrap();
        assert_eq!(pruned_tree3.predict(&[0.6, 0.7, 0.9]), 3.0);
    }

    #[test]
    fn test_tree_prune_multiple_conditions() {
        let tree = create_sample_tree_deep();
        let feature_names = vec![
            "feature0".to_string(),
            "feature1".to_string(),
            "feature2".to_string(),
        ];

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.5));
        predicate.add_condition("feature1".to_string(), Condition::LessThan(0.4));
        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.2, 0.0, 0.5]), 1.0);
        assert_eq!(pruned_tree.predict(&[0.4, 0.0, 1.0]), 1.0);

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.4));
        predicate.add_condition("feature2".to_string(), Condition::GreaterThanOrEqual(0.7));
        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.6, 0.3, 0.5]), 1.0);
        assert_eq!(pruned_tree.predict(&[0.8, 0.29, 1.0]), 1.0);
    }

    #[test]
    fn test_xgboost_style_builder() -> Result<(), FeatureTreeError> {
        // This represents a simple tree:
        //          [age < 30]
        //         /          \
        //    [-1.0]        [income < 50k]
        //                  /           \
        //               [0.0]         [1.0]

        let tree = FeatureTreeBuilder::new()
            .split_indices(vec![0, -1, 1, -1, -1])
            .split_conditions(vec![30.0, 0.0, 50000.0, 0.0, 0.0])
            .children(
                vec![1, u32::MAX, 3, u32::MAX, u32::MAX], // left children
                vec![2, u32::MAX, 4, u32::MAX, u32::MAX], // right children
            )
            .base_weights(vec![0.0, -1.0, 0.0, 0.0, 1.0])
            .default_left(vec![false, false, false, false, false])
            .build()?;

        assert!(tree.predict(&[25.0, 0.0]) < 0.0); // young
        assert!(tree.predict(&[35.0, 60000.0]) > 0.0); // old, high income
        assert!(tree.predict(&[35.0, 40000.0]) == 0.0); // old, low income

        Ok(())
    }

    #[test]
    fn test_array_length_mismatch() {
        let result = FeatureTreeBuilder::new()
            .split_indices(vec![0])
            .split_conditions(vec![30.0])
            .children(vec![1], vec![2, 3]) // mismatched lengths
            .base_weights(vec![0.0])
            .build();

        assert!(matches!(result, Err(FeatureTreeError::InvalidStructure(_))));
    }

    fn create_mixed_type_record_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float32, false), // float feature
            Field::new("f1", DataType::Int64, false),   // integer feature
            Field::new("f2", DataType::Boolean, false), // boolean feature
        ]);

        let float_array = Float32Array::from(vec![0.5, 0.3, 0.7, 0.4]);
        let int_array = Int64Array::from(vec![100, 50, 75, 25]);
        let bool_array = BooleanArray::from(vec![true, false, true, false]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(float_array),
                Arc::new(int_array),
                Arc::new(bool_array),
            ],
        )
        .unwrap()
    }

    fn create_mixed_type_tree() -> VecTreeNodes {
        // Create a tree that uses all feature types:
        //                [f0 < 0.5]
        //               /          \
        //        [f1 < 60]        [f2 == true]
        //        /       \        /           \
        //    [-1.0]    [0.0]  [1.0]        [2.0]

        FeatureTreeBuilder::new()
            .split_indices(vec![0, 1, -1, -1, 2, -1, -1])
            .split_conditions(vec![0.5, 60.0, 0.0, 0.0, 0.5, 0.0, 0.0])
            .children(
                vec![1, 2, u32::MAX, u32::MAX, 5, u32::MAX, u32::MAX],
                vec![4, 3, u32::MAX, u32::MAX, 6, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 2.0])
            .default_left(vec![false, false, false, false, false, false, false, false])
            .build()
            .unwrap()
    }

    #[test]
    fn test_predict_arrays_mixed_types() {
        let tree = create_mixed_type_tree();
        let batch = create_mixed_type_record_batch();

        let gbdt = GradientBoostedDecisionTrees {
            trees: vec![tree],
            feature_names: Arc::new(vec!["f0".to_string(), "f1".to_string(), "f2".to_string()]),
            feature_types: Arc::new(vec![
                FeatureType::Float,
                FeatureType::Int,
                FeatureType::Indicator,
            ]),
            base_score: 0.0,
            objective: Objective::SquaredError,
            config: PredictorConfig::default(),
            required_features: HashSet::from([0, 1, 2]),
        };

        let predictions = gbdt.predict_arrays(batch.columns()).unwrap();

        // Row 0: f0=0.5, f1=100, f2=true(1.0)
        //   f0=0.5 >= 0.5 -> right path -> f2=1.0 >= 0.5 -> 2.0
        // Row 1: f0=0.3, f1=50, f2=false(0.0)
        //   f0=0.3 < 0.5 -> left path -> f1=50 < 60 -> -1.0
        // Row 2: f0=0.7, f1=75, f2=true(1.0)
        //   f0=0.7 >= 0.5 -> right path -> f2=1.0 >= 0.5 -> 2.0
        // Row 3: f0=0.4, f1=25, f2=false(0.0)
        //   f0=0.4 < 0.5 -> left path -> f1=25 < 60 -> -1.0

        let expected = [2.0, -1.0, 2.0, -1.0];
        for (i, &expected_value) in expected.iter().enumerate() {
            assert!(
                (predictions.value(i) - expected_value).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                predictions.value(i)
            );
        }
    }

    #[test]
    fn test_predict_arrays_batch_processing() {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float32, false),
            Field::new("f1", DataType::Float32, false),
        ]);

        let n_rows = 1000;
        let f0_data: Vec<f32> = (0..n_rows).map(|i| (i as f32) / n_rows as f32).collect();
        let f1_data: Vec<f32> = (0..n_rows).map(|i| (i as f32) * 2.0).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float32Array::from(f0_data)),
                Arc::new(Float32Array::from(f1_data)),
            ],
        )
        .unwrap();

        let trees: Vec<VecTreeNodes> = (0..100) // is this still needed?
            .map(|_| create_sample_tree())
            .collect();

        let gbdt = GradientBoostedDecisionTrees {
            trees,
            feature_names: Arc::new(vec!["f0".to_string(), "f1".to_string()]),
            feature_types: Arc::new(vec![FeatureType::Float, FeatureType::Float]),
            base_score: 0.0,
            objective: Objective::SquaredError,
            config: PredictorConfig::default(),
            required_features: HashSet::from([0, 1]),
        };

        let predictions = gbdt.predict_arrays(batch.columns()).unwrap();
        assert_eq!(predictions.len(), n_rows);
    }

    #[test]
    fn test_predict_arrays_error_handling() {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Utf8, false), // Unsupported type
            Field::new("f1", DataType::Float32, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["invalid"])),
                Arc::new(Float32Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let gbdt = GradientBoostedDecisionTrees {
            trees: vec![create_sample_tree()],
            feature_names: Arc::new(vec!["f0".to_string(), "f1".to_string()]),
            feature_types: Arc::new(vec![FeatureType::Float, FeatureType::Float]),
            base_score: 0.0,
            objective: Objective::SquaredError,
            config: PredictorConfig::default(),
            required_features: HashSet::from([1, 2]),
        };

        let result = gbdt.predict_arrays(batch.columns());
        assert!(matches!(result, Err(ArrowError::InvalidArgumentError(_))));
    }

    #[test]
    fn test_prune_with_default_direction_and_nulls() {
        // Create a deeper tree:
        //                    [feature0 < 0.5]
        //                   /               \
        //      [feature1 < 0.3]            [feature1 < 0.6]
        //     /               \            /               \
        // [feature2 < 0.7]    [-1.0]    [1.0]       [feature2 < 0.8]
        //   /        \                                /            \
        // [-2.0]    [2.0]                          [2.0]         [3.0]

        let feature_tree = FeatureTreeBuilder::new()
            .split_indices(vec![0, 1, 2, -1, -1, 1, 2, -1, -1])
            .split_conditions(vec![0.5, 0.3, 0.7, 0.0, 0.0, 0.6, 0.8, 0.0, 0.0])
            .children(
                vec![1, 3, 4, u32::MAX, u32::MAX, 7, 8, u32::MAX, u32::MAX],
                vec![5, 2, 6, u32::MAX, u32::MAX, 6, 8, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 2.0, 2.0, 3.0])
            .default_left(vec![
                true, false, false, false, false, false, false, false, false,
            ])
            .build()
            .unwrap();

        let predictions_right = feature_tree.predict(&[0.4, 0.3, 0.8]);

        let mut predicate1 = Predicate::new();
        predicate1.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.5));
        let pruned1 = feature_tree
            .prune(&predicate1, &["f0".to_string(), "f1".to_string()])
            .unwrap();
        let predicitons_after_pruning = pruned1.predict(&[f32::NAN, 0.3, 0.8]);
        assert_eq!(predictions_right, predicitons_after_pruning);
    }

    #[test]
    fn test_required_features() {
        let tree = create_sample_tree(); // Your existing test helper
        let gbdt = GradientBoostedDecisionTrees {
            trees: vec![tree],
            feature_names: Arc::new(vec!["f0".to_string(), "f1".to_string(), "f2".to_string()]),
            feature_types: Arc::new(vec![
                FeatureType::Float,
                FeatureType::Float,
                FeatureType::Float,
            ]),
            base_score: 0.0,
            objective: Objective::SquaredError,
            config: PredictorConfig::default(),
            required_features: HashSet::from([0]),
        };

        let required = gbdt.get_required_features();
        assert!(required.contains(&0)); // Only feature0 is used in sample tree
        assert!(!required.contains(&1));
        assert!(!required.contains(&2));
    }

    #[test]
    fn test_base_score_transformation_logistic() {
        // Test that base_score is correctly transformed from probability to logit
        // for logistic objectives during model loading
        use serde_json::json;

        // Create a minimal XGBoost JSON with a logistic objective
        let base_score_probability = 0.19499376_f32;
        let model_json = json!({
            "learner": {
                "learner_model_param": {
                    "base_score": format!("{:E}", base_score_probability),
                    "num_class": "0",
                    "num_feature": "1"
                },
                "objective": {
                    "name": "binary:logistic"
                },
                "feature_names": ["f0"],
                "feature_types": ["float"],
                "gradient_booster": {
                    "model": {
                        "gbtree_model_param": {
                            "num_trees": "1",
                            "num_parallel_tree": "1"
                        },
                        "tree_info": [0],
                        "trees": [{
                            "tree_param": {
                                "num_nodes": "1",
                                "size_leaf_vector": "1",
                                "num_feature": "1"
                            },
                            "split_indices": [-1],
                            "split_conditions": [0.0],
                            "default_left": [0],
                            "base_weights": [0.0],
                            "left_children": [u32::MAX],
                            "right_children": [u32::MAX],
                            "sum_hessian": [1.0]
                        }]
                    }
                }
            }
        });

        // Load the model
        let model = GradientBoostedDecisionTrees::json_loads(&model_json).unwrap();

        // The base_score should be transformed from probability to logit
        // logit = ln(p / (1-p)) = ln(0.19499376 / 0.80500624) ≈ -1.4178825
        let expected_logit = (base_score_probability / (1.0 - base_score_probability)).ln();

        assert!(
            (model.base_score - expected_logit).abs() < 1e-6,
            "Base score should be transformed from probability {} to logit {}, but got {}",
            base_score_probability,
            expected_logit,
            model.base_score
        );

        // Verify the objective is logistic
        assert_eq!(model.objective, Objective::Logistic);

        // Verify that when computing score with just base_score (no tree contributions),
        // we get back the original probability after sigmoid transformation
        let score_after_sigmoid = model.objective.compute_score(model.base_score);
        assert!(
            (score_after_sigmoid - base_score_probability).abs() < 1e-6,
            "Sigmoid of transformed base_score should equal original probability {}, but got {}",
            base_score_probability,
            score_after_sigmoid
        );
    }

    #[test]
    fn test_base_score_no_transformation_squarederror() {
        // Test that base_score is NOT transformed for squared error objective
        use serde_json::json;

        let base_score_value = 3932.7998_f32;
        let model_json = json!({
            "learner": {
                "learner_model_param": {
                    "base_score": format!("{:E}", base_score_value),
                    "num_class": "0",
                    "num_feature": "1"
                },
                "objective": {
                    "name": "reg:squarederror"
                },
                "feature_names": ["f0"],
                "feature_types": ["float"],
                "gradient_booster": {
                    "model": {
                        "gbtree_model_param": {
                            "num_trees": "1",
                            "num_parallel_tree": "1"
                        },
                        "tree_info": [0],
                        "trees": [{
                            "tree_param": {
                                "num_nodes": "1",
                                "size_leaf_vector": "1",
                                "num_feature": "1"
                            },
                            "split_indices": [-1],
                            "split_conditions": [0.0],
                            "default_left": [0],
                            "base_weights": [0.0],
                            "left_children": [u32::MAX],
                            "right_children": [u32::MAX],
                            "sum_hessian": [1.0]
                        }]
                    }
                }
            }
        });

        let model = GradientBoostedDecisionTrees::json_loads(&model_json).unwrap();

        // For squared error, base_score should NOT be transformed
        assert!(
            (model.base_score - base_score_value).abs() < 1e-2,
            "Base score for squared error should remain {}, but got {}",
            base_score_value,
            model.base_score
        );
    }
}
