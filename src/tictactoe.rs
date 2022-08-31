use crate::envs::Environment;
use crate::episodes::Tensorable;
use tch::{Device, Tensor};

#[derive(Clone, Debug)]
pub struct TicTacToeObservation {
    board: [f64; 9],
}

impl Tensorable for TicTacToeObservation {
    fn to_tensor(&self) -> Tensor {
        Tensor::of_slice(&self.board).to_device(Device::Cpu)
    }
}

#[derive(Clone, Debug)]
pub struct TicTacToeEnv {
    board: [f64; 9],
    player: f64,
}

impl TicTacToeEnv {
    pub fn new() -> Self {
        TicTacToeEnv {
            board: [0.0; 9],
            player: 1.0,
        }
    }

    //0, 1, 2
    //3, 4, 5
    //6, 7, 8

    fn check_win(&self) -> f64 {
        for i in 0..3 {
            if self.board[i] == self.board[i + 3] && self.board[i] == self.board[i + 6] {
                return 1.0;
            }
            if self.board[i] == self.board[i + 1] && self.board[i] == self.board[i + 2] {
                return 1.0;
            }
        }

        if self.board[0] == self.board[4] && self.board[0] == self.board[8] {
            return 1.0;
        }
        if self.board[2] == self.board[4] && self.board[2] == self.board[6] {
            return 1.0;
        }
        0.0
    }

    fn check_draw(&self) -> bool {
        for i in 0..9 {
            if self.board[i] == 0.0 {
                return false;
            }
        }
        true
    }
}

impl Environment for TicTacToeEnv {
    type Observation = TicTacToeObservation;

    fn reset(&mut self) -> Self::Observation {
        self.board = [0.0; 9];
        self.player = 1.0;
        TicTacToeObservation { board: self.board }
    }

    fn step(&mut self, action: i64) -> (i64, Self::Observation, f64, bool) {
        if !(self.board[action as usize] == 0.0) {
            return (
                action,
                TicTacToeObservation { board: self.board },
                0.0,
                true,
            );
        }

        self.board[action as usize] = self.player;
        let reward = self.check_win();
        self.player *= -1.0;
        let done = (reward != 0.0) || self.check_draw();
        (
            action,
            TicTacToeObservation { board: self.board },
            reward,
            done,
        )
    }

    fn action_space(&self) -> i64 {
        9
    }
}
