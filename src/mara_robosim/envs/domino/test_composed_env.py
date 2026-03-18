#!/usr/bin/env python
"""Test script for the refactored domino environment.

This script tests that the component-based architecture works correctly
and maintains backward compatibility with the original environment.

Usage:
    python -m mara_robosim.envs.domino.test_composed_env
"""

import time

import numpy as np

from mara_robosim.config import PyBulletConfig


def test_domino_fan_env() -> None:
    """Test the domino + fan + ball environment."""
    config = PyBulletConfig(
        seed=0,
        num_train_tasks=0,
        num_test_tasks=2,
    )

    # Import and create the environment
    from mara_robosim.envs.domino import PyBulletDominoFanEnv
    from mara_robosim.structs import Action

    print("Creating PyBulletDominoFanEnv...")
    env = PyBulletDominoFanEnv(config=config, use_gui=True)

    # Print environment info
    print(f"\n{'=' * 60}")
    print("Environment Information")
    print(f"{'=' * 60}")
    print(f"Environment name: {env.get_name()}")
    print(f"\nTypes ({len(env.types)}):")
    for t in sorted(env.types, key=lambda x: x.name):
        print(f"  - {t.name}: {t.feature_names}")
    print(f"\nPredicates ({len(env.predicates)}):")
    for pred in sorted(env.predicates, key=lambda x: x.name):
        print(f"  - {pred.name}")
    print(f"\nGoal predicates ({len(env.goal_predicates)}):")
    for pred in sorted(env.goal_predicates, key=lambda x: x.name):
        print(f"  - {pred.name}")

    # Generate test tasks
    print(f"\n{'=' * 60}")
    print("Generating test tasks...")
    print(f"{'=' * 60}")
    tasks = env._generate_test_tasks()
    print(f"Generated {len(tasks)} tasks")

    # Test each task
    for i, task in enumerate(tasks):
        print(f"\n{'=' * 60}")
        print(f"Task {i + 1}")
        print(f"{'=' * 60}")

        # Reset to initial state
        env._reset_state(task.init)

        print("\nObjects in state:")
        assert env._domino_component is not None
        for obj in task.init.get_objects(env._domino_component.domino_type):
            print(f"  - {obj.name}")

        print(f"\nGoal atoms ({len(task.goal)}):")
        for atom in task.goal:
            print(f"  - {atom}")

        # Simulate for a short time
        print("\nRunning simulation...")
        try:
            for step_num in range(300):
                action = Action(np.array(env._pybullet_robot.initial_joint_positions))
                state = env.step(action)

                if all(atom.holds(state) for atom in task.goal):
                    print(f"  Goal reached at step {step_num}!")
                    time.sleep(1)
                    break

                if step_num % 100 == 0:
                    print(f"  Step {step_num}...")

                time.sleep(0.02)
        except KeyboardInterrupt:
            print("  Interrupted by user")
            continue

    print(f"\n{'=' * 60}")
    print("Test completed successfully!")
    print(f"{'=' * 60}")


def test_domino_only_env() -> None:
    """Test the domino-only environment."""
    config = PyBulletConfig(
        seed=0,
        num_train_tasks=0,
        num_test_tasks=1,
    )

    from mara_robosim.envs.domino import PyBulletDominoEnv
    from mara_robosim.structs import Action

    print("\nCreating PyBulletDominoEnv (domino-only)...")
    env = PyBulletDominoEnv(config=config, use_gui=True)

    print(f"Environment name: {env.get_name()}")
    print(f"Types: {[t.name for t in env.types]}")
    print(f"Predicates: {[pred.name for pred in env.predicates]}")

    tasks = env._generate_test_tasks()
    print(f"Generated {len(tasks)} tasks")

    if tasks:
        env._reset_state(tasks[0].init)
        print("Running simulation for 100 steps...")
        for step_num in range(100):
            action = Action(np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.02)

    print("Domino-only test completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--domino-only":
        test_domino_only_env()
    else:
        test_domino_fan_env()
