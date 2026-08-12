"""Replay a RoomKG episode deterministically and print the hidden state.

The environment is fully deterministic: wall schedules and object motion are
periodic functions of the timestep, so re-instantiating it with the same
configuration reproduces the same trajectory every time. This script steps it to
a given timestep and serializes the hidden state, which is how the Turtle
fragment in the paper (Listing 2) was produced.

    python replay.py --step 99                 # hidden state at t=99, Turtle
    python replay.py --step 0 --format triples # one triple per line
    python replay.py --step 99 --room library  # only that room's edges

Compare the northern edge of `library` at t=0 and t=99 to see a wall schedule
complete half a period.
"""

import argparse

import gymnasium as gym
import room_env  # noqa: F401  (registers room_env:RoomEnv-v3)


def replay(step: int, room_size: str, terminates_at: int):
    """Step the environment to `step` and return its hidden-state RDF graph."""
    env = gym.make(
        "room_env:RoomEnv-v3", terminates_at=terminates_at, room_size=room_size
    )
    env.reset()
    for _ in range(step):
        _, _, terminated, truncated, _ = env.step(("wall", "stay"))
        if terminated or truncated:
            break
    return env.unwrapped.get_rdf_graph()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=int, default=99, help="timestep to stop at")
    p.add_argument("--room-size", default="large-02", help="layout name")
    p.add_argument("--terminates-at", type=int, default=99, help="episode length")
    p.add_argument("--room", default=None, help="restrict output to one room")
    p.add_argument(
        "--format", default="turtle", choices=["turtle", "triples"],
        help="turtle serialization, or one subject-predicate-object per line",
    )
    args = p.parse_args()

    g = replay(args.step, args.room_size, args.terminates_at)

    if args.room is not None:
        wanted = args.room
        triples = [
            (s, p_, o) for s, p_, o in g
            if wanted in (str(s).rsplit("/", 1)[-1], str(o).rsplit("/", 1)[-1])
        ]
    else:
        triples = list(g)

    if args.format == "turtle" and args.room is None:
        print(g.serialize(format="turtle"))
    else:
        for s, p_, o in sorted(triples, key=lambda t: (str(t[0]), str(t[1]))):
            short = lambda t: str(t).rsplit("/", 1)[-1]  # noqa: E731
            print(f"{short(s)} {short(p_)} {short(o)} .")


if __name__ == "__main__":
    main()
