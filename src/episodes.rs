use bounded_vec_deque::BoundedVecDeque;
use tch::{Device, IndexOp, Tensor};

pub trait Tensorable {
    fn to_tensor(&self) -> Tensor;
}

#[derive(Clone, Debug)]
pub struct ActionStep<T>
where
    T: Tensorable + Clone,
{
    pub action: i64,
    pub observation1: T,
    pub observation2: T,
    pub reward: f64,
}

#[derive(Clone, Debug)]
pub struct Episode<T>
where
    T: Tensorable + Clone,
{
    steps: Vec<ActionStep<T>>,
    size: usize,
}

impl<T> Episode<T>
where
    T: Tensorable + Clone,
{
    pub fn new() -> Self {
        Episode {
            steps: Vec::new(),
            size: 0,
        }
    }
    pub fn add_step(&mut self, step: ActionStep<T>) {
        self.steps.push(step);
        self.size += 1;
    }
}

#[derive(Debug)]
pub struct EpisodeQueue<T>
where
    T: Tensorable + Clone,
{
    pub episodes: BoundedVecDeque<Episode<T>>,
    pub model_name: String,
}

impl<T> EpisodeQueue<T>
where
    T: Tensorable + Clone,
{
    pub fn new(capacity: usize, model_name: String) -> Self {
        EpisodeQueue {
            episodes: BoundedVecDeque::new(capacity),
            model_name: model_name,
        }
    }

    pub fn add_episode(&mut self, episode: Episode<T>) {
        self.episodes.push_back(episode);
    }

    pub fn to_dataset(&self, batch_size: i64, device: Device) -> DataSet {
        let mut size = 0i64;

        for episode in self.episodes.iter() {
            for _ in episode.steps.iter() {
                size += 1;
            }
        }

        let mut actions: Vec<Tensor> = Vec::new();
        let mut observation1s: Vec<Tensor> = Vec::new();
        let mut observation2s: Vec<Tensor> = Vec::new();
        let mut rewards: Vec<Tensor> = Vec::new();

        for episode in self.episodes.iter() {
            for step in episode.steps.iter() {
                actions.push(Tensor::try_from(step.action).unwrap());
                observation1s.push(step.observation1.to_tensor());
                observation2s.push(step.observation2.to_tensor());
                rewards.push(Tensor::try_from(step.reward).unwrap());
            }
        }

        DataSet {
            name: self.model_name.clone(),
            actions: Tensor::concat(&actions, 0).to_device(device),
            observation1s: Tensor::concat(&observation1s, 0).to_device(device),
            observation2s: Tensor::concat(&observation2s, 0).to_device(device),
            rewards: Tensor::concat(&rewards, 0).to_device(device),
            batch_size: batch_size,
            device: device,
            total_size: size,
            batch_index: 0,
        }
    }
}

#[derive(Debug)]
pub struct DataSet {
    pub name: String,
    actions: Tensor,
    observation1s: Tensor,
    observation2s: Tensor,
    rewards: Tensor,
    batch_size: i64,
    batch_index: i64,
    total_size: i64,
    device: Device,
}

impl Iterator for DataSet {
    type Item = (Tensor, Tensor, Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.batch_size * self.batch_index;
        let size = std::cmp::min(self.batch_size, self.total_size - start);
        println!("{}: {} {}", self.name, start, size);
        if size <= 0 || (size < self.batch_size) {
            None
        } else {
            self.batch_index += 1;
            Some((
                self.actions.i(start..start + size).to_device(self.device),
                self.observation1s
                    .i(start..start + size)
                    .to_device(self.device),
                self.observation2s
                    .i(start..start + size)
                    .to_device(self.device),
                self.rewards.i(start..start + size).to_device(self.device),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Tensor};

    #[test]
    fn test_episode_queue() {
        impl Tensorable for f64 {
            fn to_tensor(&self) -> Tensor {
                Tensor::of_slice(&[*self]).to_device(Device::Cpu)
            }
        }

        let mut queue = EpisodeQueue::new(5, "test".to_string());
        let mut episode = Episode::new();
        let step = ActionStep {
            action: 1,
            observation1: 2.0,
            observation2: 3.0,
            reward: 4.0,
        };
        episode.add_step(step);
        queue.add_episode(episode);
        let mut episode = Episode::new();
        let step = ActionStep {
            action: 5,
            observation1: 6.0,
            observation2: 7.0,
            reward: 8.0,
        };
        episode.add_step(step);
        queue.add_episode(episode);
        let mut episode = Episode::new();
        let step = ActionStep {
            action: 9,
            observation1: 10.0,
            observation2: 11.0,
            reward: 12.0,
        };
        episode.add_step(step);
        queue.add_episode(episode);
        let mut episode = Episode::new();
        let step = ActionStep {
            action: 13,
            observation1: 14.0,
            observation2: 15.0,
            reward: 16.0,
        };
        episode.add_step(step);
        queue.add_episode(episode);
        let mut episode = Episode::new();
        let step = ActionStep {
            action: 17,
            observation1: 18.0,
            observation2: 19.0,
            reward: 20.0,
        };
        episode.add_step(step);
        queue.add_episode(episode);
        let mut episode = Episode::new();
        let step = ActionStep {
            action: 21,
            observation1: 22.0,
            observation2: 23.0,
            reward: 24.0,
        };
        episode.add_step(step);
        queue.add_episode(episode);
        let mut episode = Episode::new();
        let step = ActionStep {
            action: 25,
            observation1: 26.0,
            observation2: 27.0,
            reward: 28.0,
        };
        episode.add_step(step);
        queue.add_episode(episode);

        assert_eq!(queue.episodes.len(), 5);
        let mut i = 0;
        for (a, o1, o2, r) in queue.to_dataset(2, Device::Cpu) {
            i += 1;
            println!(
                "a:{:?} o1:{:?} o2:{:?} r:{:?}",
                a.size(),
                o1.size(),
                o2.size(),
                r.size()
            );
        }

        assert_eq!(i, 2);
    }
}
