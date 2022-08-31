use crate::episodes::Tensorable;

pub trait Environment {
    type Observation: Tensorable + Clone;

    fn step(&mut self, action: i64) -> (i64, Self::Observation, f64, bool);
    fn reset(&mut self) -> Self::Observation;
    fn action_space(&self) -> i64;
}
