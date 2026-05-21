##
#
# Finite State Machine for robot deployment.
#
##

from utils.joystick_utils import JoystickState


#######################################################################
# FSM Configuration
#######################################################################

# states: init -> damp -> home -> control
#
#                ┌──┐         ┌──┐         ┌──┐
#                │  v         │  v         │  v
#   INIT ──────> DAMP ──────> HOME ──────> CONTROL
#                 ^            │              │
#                 │            │              │
#                 └────────────┘              │
#                 └───────────────────────────┘

# all possible states
STATES = ["init", "damp", "home", "control"]

# allowable transitions
TRANSITIONS = {
    "init":    {"init", "damp"},
    "damp":    {"damp", "home"},
    "home":    {"home", "damp", "control"},
    "control": {"control", "damp"},
}

# button to target state mapping
BUTTON_STATE_MAP = {
    "LB":  "damp",    # go to "damp"
    "A":   "home",    # go to "home"
    "LMB": "control", # go to "control"
}


#######################################################################
# FSM Class
#######################################################################

class FiniteStateMachine:

    def __init__(self):

        # initialize to "init" state
        self.state = "init"

    # check that the joystick buttons are valid and transition if needed
    def step(self, joystick_state: JoystickState) -> str:

        # check each button in priority order (LB -> A -> LMB)
        for button, target in BUTTON_STATE_MAP.items():

            # check if the button is pressed
            if getattr(joystick_state, button):

                # only transition if the target state is reachable
                if target in TRANSITIONS[self.state]:

                    # log transition (skip self-loops)
                    if target != self.state:
                        print(f"FSM: {self.state} -> {target}")
                    self.state = target
                    break

        return self.state
