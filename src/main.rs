#![warn(clippy::pedantic, clippy::nursery)]

use mcts::Mcts;
use tictactoe::TicTacToeEnv;

mod envs;
mod episodes;
mod mcts;
mod tictactoe;

fn main() {
    let t = TicTacToeEnv::new();
    let m = Mcts::new(t);

    println!("{}", m);

    for _ in 0..1_000_000 {
        let _ = m.search(0, 1.0);
    }

    println!("{}", m);
    println!("{}", m.arena.lock().unwrap().len());
}
