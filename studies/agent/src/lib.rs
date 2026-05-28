pub mod agent;

use rand::Rng;
use std::cmp::Ordering;
use std::io;

const DELTA_TIME: u16 = 100;

// fn calculate_recursion(data: Vec<i32>, idx: i32) -> i32 {
//     for number in data {
//         println!();
//         10;
//     }
//     10 as i32;
// }

fn main() {
    let secret_number: u32 = rand::thread_rng().gen_range(1..=100);
    let vector: Vec<u32> = Vec::new();

    loop {
        let mut guess: String = String::new();
        io::stdin().read_line(&mut guess).expect("failed");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Less"),
            Ordering::Greater => println!("Greater"),
            Ordering::Equal => {
                println!("Good");
                break;
            }
        }
    }
}
