"""The Step 7 automatic paraphrase validation catches each kind of violation (template-level paraphrase bank)."""
from actspace_build.step7_rewrite import validate_one

OBJ = ["sofa", "table", "tv", "armchair", "larger", "smaller", "largest", "smallest"]
T1 = "What is the distance between the center of the {OBJ1} and the center of the {OBJ2}, in meters?"
T3 = "Imagine you are standing at the {OBJ1} and facing the {OBJ2}. Where is the {OBJ3} relative to you?"
T5 = "Imagine you are standing at the {OBJ1} and facing the {OBJ2}. Among the objects on your left, which one is closest to you?"
CL = "Which of the following objects is closest to the {OBJ1}?"


def errs(key, orig, p):
    return validate_one(key, orig, p, OBJ)


def test_good_sentences_pass():
    assert errs("T1", T1, "How far is the center of the {OBJ1} from the center of the {OBJ2}, in meters?") == []
    assert errs("T3", T3, "Stand at the {OBJ1} and face the {OBJ2}. Where is the {OBJ3} relative to you?") == []


def test_violations_are_caught():
    assert errs("T1", T1, "How far is the {OBJ1} from the {OBJ2}, in meters?")                       # center dropped
    assert errs("T1", T1, "How far is the center of the {OBJ1} from the center of the {OBJ2}?")       # meters dropped
    assert errs("T1", T1, "How far is the center of the {OBJ1} from the center of the {OBJ1}, in meters?")  # placeholders
    assert errs("T3", T3, "Stand at the {OBJ1} and face the {OBJ2}. Is the {OBJ3} behind you?")       # direction cue
    assert errs("T3", T3, "Stand at the {OBJ1}. Where is the {OBJ3} relative to you, given the {OBJ2}?")  # facing dropped
    assert errs("T5", T5, "Standing at the {OBJ1} and facing the {OBJ2}, which object on your right is closest to you?")
    assert errs("T2_closest", CL, "Which of the following objects is farthest from the {OBJ1}?")       # comparison direction reversed
    assert errs("T2_closest", CL, "Which object near the window is closest to the {OBJ1}?")           # new cue
    assert errs("T2_closest", CL, "Which of the 3 objects is closest to the {OBJ1}?")                 # new digit
    assert errs("T2_closest", CL, "Which object, other than the sofa, is closest to the {OBJ1}?")     # new object name
    assert errs("T2_closest", CL, "Which of the following objects is closest to {OBJ1}?")            # missing the
    assert errs("T2_closest", CL, CL)                                                               # identical to the original


def test_option_scope_required_for_closest_farthest():
    """T2 closest / farthest must be restricted to the options."""
    FA = "Which of the following objects is farthest from the {OBJ1}?"
    assert errs("T2_closest", CL, "Select the object that is closest to the {OBJ1}.")          # no scope word → fails
    assert errs("T2_farthest", FA, "Which of these objects is the farthest from the {OBJ1}?")  # "these" is not a scope word
    assert errs("T2_closest", CL, "Select, from the options below, the object that is closest to the {OBJ1}.") == []
    assert errs("T2_closest", CL, "Choose the option that is closest to the {OBJ1}.") == []    # singular option also counts
    assert errs("T2_farthest", FA, "Which of these listed objects is the farthest from the {OBJ1}?") == []
