use std::error::Error;
use std::fmt::{Display, Formatter};

pub type IndexResult<T> = Result<T, IndexError>;

#[derive(Debug)]
pub enum IndexError {
    EmptyIndex,
    EmptyLevel(usize),
    NodeNotFound(usize),
    NodeNotFoundInLevel { level_index: usize, node_id: usize },
    NoNeighborCandidates,
}

impl Display for IndexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyIndex => write!(f, "Empty index"),
            Self::EmptyLevel(level) => write!(f, "Level {level} empty"),
            Self::NodeNotFound(id) => write!(f, "Node {id} not found"),
            Self::NodeNotFoundInLevel {
                level_index,
                node_id,
            } => {
                write!(f, "Node {node_id} not found in level {level_index}")
            }
            Self::NoNeighborCandidates => write!(f, "Candidate not found"),
        }
    }
}

impl Error for IndexError {}
