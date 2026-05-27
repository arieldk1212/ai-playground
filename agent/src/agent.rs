use std::io;

#[derive(Debug, PartialEq, Clone, Copy)]
enum Action {
    Research,
    Train,
    Test,
}

#[derive(Debug, Clone)]
struct Agent {
    id: u64,
    name: String,
    action: Action,
}

static CLOSURE: fn(u32) -> u32 = |num: u32| num;

impl Agent {
    // fn new(id: u64, name: String) -> Self {
    //     Self { id, name }
    // }

    fn user_action(&self, action_type: Option<Action>) -> Action {
        action_type.unwrap_or_else(|| self.start_action())
    }

    fn start_action(&self) -> Action {
        Action::Research
    }
}
