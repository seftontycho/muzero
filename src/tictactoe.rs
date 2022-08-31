use crate::envs::Environment;
use crate::episodes::Tensorable;
use tch::{Device, Tensor};

#[derive(Clone, Debug)]
pub struct TicTacToeObservation {
    pub board: [i64; 9],
}

impl Tensorable for TicTacToeObservation {
    fn to_tensor(&self) -> Tensor {
        Tensor::of_slice(&self.board).to_device(Device::Cpu)
    }
}

#[derive(Clone, Debug)]
pub struct TicTacToeEnv {
    board: [i64; 9],
    player: i64,
}

impl TicTacToeEnv {
    pub const fn new() -> Self {
        Self {
            board: [0; 9],
            player: 1,
        }
    }

    //0, 1, 2
    //3, 4, 5
    //6, 7, 8

    fn check_win(&self) -> f64 {
        let rows = (0..3).any(|i| self.board.iter().skip(i).take(3).all(|x| *x == self.player));
        let cols = (0..3).any(|i| {
            self.board[3 * i..3 * (i + 1)]
                .iter()
                .all(|x| *x == self.player)
        });
        let diag = (0..3).all(|i| self.board[4 * i] == self.player);

        if rows || cols || diag {
            1.0
        } else {
            0.0
        }
    }
}

impl Environment for TicTacToeEnv {
    type Observation = TicTacToeObservation;

    fn reset(&mut self) -> Self::Observation {
        self.board = [0; 9];
        self.player = 1;
        TicTacToeObservation { board: self.board }
    }

    fn step(&mut self, action: usize) -> (usize, Self::Observation, f64, bool) {
        if self.board[action] != 0 {
            return (
                action,
                TicTacToeObservation { board: self.board },
                0.0,
                true,
            );
        }

        self.board[action as usize] = self.player;
        let reward = self.check_win();
        self.player *= -1;

        let draw = self.board.iter().all(|i| *i != 0);
        let won = reward > 0.0001;

        let done = won || draw;

        (
            action,
            TicTacToeObservation { board: self.board },
            reward,
            done,
        )
    }

    fn action_space(&self) -> usize {
        9
    }
}
