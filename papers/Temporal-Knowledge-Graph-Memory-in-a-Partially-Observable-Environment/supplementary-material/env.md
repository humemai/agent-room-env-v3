# Environment code

Only the important part of the code is shown here. 

## Env

This class loads the config and simulates the environment.

```python
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, URIRef

from ..utils import is_running_notebook, rdf_to_list
from ..utils import read_json_prod as read_json


class Env(gym.Env):
    """With deterministic object movement."""

    def __init__(
        self,
        terminates_at: int = 99,
        room_size: str = "small",
    ) -> None:
        """Initialize Environment.

        Args:
            terminates_at: When episode terminates (can be any value)
            room_size: Room configuration size
        """
        super().__init__()

        self.is_notebook = is_running_notebook()

        # Load configuration
        config = read_json(f"room-config.json")

        self.grid_length = config["grid_length"]
        self.room_names = config["room_names"]
        self.room_positions = config["room_positions"]
        self.base_room_connections = config["room_connections"]
        self.static_names = config["static_names"]
        self.moving_names = config["moving_names"]
        self.initial_static_locations = config["static_locations"]
        self.initial_moving_locations = config["moving_locations"]
        self.movement_preferences = config["movement_preferences"]
        self.initial_agent_location = config["agent_location"]
        self.selected_walls = config["selected_walls"]

        # Convert wall configs from string keys back to tuples
        self.wall_configs = {}
        for wall_key, pattern in config["wall_configs"].items():
            # Convert string key back to tuple
            parts = wall_key.split("|")
            wall_tuple = (parts[0], parts[1], parts[2])
            self.wall_configs[wall_tuple] = pattern

        self.question_objects = config["question_objects"]  # Always 100 questions

        # Environment parameters
        self.terminates_at = terminates_at

        # Dummy gym spaces
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

        self.entities = (
            self.room_names + self.static_names + self.moving_names + ["agent", "wall"]
        )
        self.relations = ["north", "east", "south", "west", "at_location"]
        self.total_maximum_episode_rewards = self.terminates_at + 1

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0

        # Reset object locations
        self.static_locations = self.initial_static_locations.copy()
        self.moving_locations = self.initial_moving_locations.copy()
        self.agent_location = self.initial_agent_location

        # Set initial wall layout
        self._update_wall_layout()

        # Get initial observations
        observations = self._get_observations()
        info = {}

        return observations, info

    def step(self, actions):
        """Take environment step.

        Args:
            actions: Tuple of (question_answer, movement_action)

        Returns:
            observations, reward, done, truncated, info
        """
        question_answer, movement_action = actions

        # Calculate reward for current question
        reward = self._calculate_reward(question_answer)

        # Move objects deterministically
        self._move_objects()

        # Move agent
        self._move_agent(movement_action)

        # Update step counter
        self.current_step += 1

        # Update wall layout based on patterns
        self._update_wall_layout()

        # Check if done
        done = self.current_step > self.terminates_at
        truncated = False

        # Get next observations
        observations = self._get_observations()
        info = {}

        return observations, reward, done, truncated, info

    def _update_wall_layout(self):
        """Update room connections based on periodic wall patterns."""
        # Start with base connections
        self.room_connections = {}
        for room_name in self.room_names:
            self.room_connections[room_name] = self.base_room_connections[
                room_name
            ].copy()

        # Apply inner walls based on their patterns
        for wall, pattern in self.wall_configs.items():
            # Get pattern state at current step using actual pattern length
            pattern_length = len(pattern)
            pattern_index = self.current_step % pattern_length
            is_wall_active = pattern[pattern_index] == 1

            if is_wall_active:
                room1, room2, wall_type = wall
                if wall_type == "horizontal":
                    self.room_connections[room1]["south"] = "wall"
                    self.room_connections[room2]["north"] = "wall"
                else:  # vertical
                    self.room_connections[room1]["east"] = "wall"
                    self.room_connections[room2]["west"] = "wall"

    def _move_objects(self):
        """Move all moving objects deterministically."""
        for obj_name in self.moving_names:
            current_location = self.moving_locations[obj_name]
            preferences = self.movement_preferences[obj_name]

            # Try each preferred direction in order
            for direction in preferences:
                next_location = self.room_connections[current_location][direction]
                if next_location != "wall":
                    self.moving_locations[obj_name] = next_location
                    break

    def _move_agent(self, action):
        """Move agent based on action."""
        if action in ["north", "east", "south", "west"]:
            next_location = self.room_connections[self.agent_location][action]
            if next_location != "wall":
                self.agent_location = next_location
        # "stay" action or invalid action keeps agent in place

    def _get_observations(self):
        """Get current observations."""
        # Room layout from agent's perspective
        agent_room_obs = []
        for direction in ["north", "east", "south", "west"]:
            connected = self.room_connections[self.agent_location][direction]
            agent_room_obs.append([self.agent_location, direction, connected])

        # Objects in agent's room
        for obj_name in self.static_names + self.moving_names:
            obj_location = self.static_locations.get(
                obj_name
            ) or self.moving_locations.get(obj_name)
            if obj_location == self.agent_location:
                agent_room_obs.append([obj_name, "at_location", obj_location])

        # Agent location
        agent_room_obs.append(["agent", "at_location", self.agent_location])

        random.shuffle(agent_room_obs)

        # Generate exactly one question for this step (not a list)
        question_index = self.current_step % 100  # Cycle through 100 questions
        obj_name = self.question_objects[question_index]
        question = [obj_name, "at_location", "?"]

        return {"room": agent_room_obs, "question": question}

    def _calculate_reward(self, answer):
        """Calculate reward for question answer."""
        question_index = self.current_step % 100
        obj_name = self.question_objects[question_index]

        # Get correct location
        if obj_name in self.static_names:
            correct_location = self.static_locations[obj_name]
        else:
            correct_location = self.moving_locations[obj_name]

        # Check answer and return single reward
        return 1 if answer == correct_location else 0

    def _find_objects_in_room(self, room_name: str) -> list:
        """Find all objects currently in a specified room, categorized by type."""
        objects_in_room = []

        # Check static objects
        for obj_name in self.static_names:
            if self.static_locations[obj_name] == room_name:
                objects_in_room.append(("static", obj_name))

        # Check moving objects
        for obj_name in self.moving_names:
            if self.moving_locations[obj_name] == room_name:
                objects_in_room.append(("moving", obj_name))

        # Check agent
        if self.agent_location == room_name:
            objects_in_room.append(("agent", "agent"))

        return objects_in_room

    def render(
        self,
        render_mode: str = "grid",
        figsize: tuple[int, int] = (12, 12),
        cell_text_size: int = 12,
        save_fig_dir: str = "./DEBUG/",
        image_format: str = "png",
        graph_layout: str = "spring",
    ) -> None:
        """Render the current state of the environment.

        Args:
            render_mode: How to render ('console', 'grid', or 'graph')
            figsize: Size of the figure
            cell_text_size: Text size in cells
            save_fig_dir: Directory to save figures
            image_format: Format to save images in
            graph_layout: Layout for graph rendering ('spring', 'circular', 'kamada_kawai')
        """
        if render_mode == "console":
            print(f"Step {self.current_step}:")
            print(f"Agent location: {self.agent_location}")
            print(f"Static objects: {self.static_locations}")
            print(f"Moving objects: {self.moving_locations}")

        elif render_mode == "graph":
            self._render_graph(figsize, save_fig_dir, image_format, graph_layout)

        elif render_mode == "grid":
            plt.figure(figsize=figsize)

            # Define colors for different object types
            colors = {
                "room": "#FFE4B5",  # Moccasin (Soft Gold)
                "moving": "#90EE90",  # Light Green
                "static": "#87CEFA",  # Light Blue
                "agent": "#D8BFD8",  # Thistle (Soft Purple)
            }

            # Draw the grid
            for i in range(self.grid_length):
                for j in range(self.grid_length):
                    room_idx = i * self.grid_length + j
                    room_name = self.room_names[room_idx]

                    # Draw room cell with room color background
                    rect = plt.Rectangle(
                        (j, self.grid_length - 1 - i),
                        1,
                        1,
                        facecolor=colors["room"],
                        edgecolor="black",
                        linewidth=1,
                    )
                    plt.gca().add_patch(rect)

                    # Get objects in this room
                    objects_in_room = self._find_objects_in_room(room_name)

                    # Group objects by type for display
                    static_objs = [
                        obj for obj_type, obj in objects_in_room if obj_type == "static"
                    ]
                    moving_objs = [
                        obj for obj_type, obj in objects_in_room if obj_type == "moving"
                    ]
                    agent_objs = [
                        obj for obj_type, obj in objects_in_room if obj_type == "agent"
                    ]

                    y_offset = 0.85  # Start position for room name
                    line_height = 0.12  # Space between lines

                    # Display room name at top in black (no background box)
                    plt.text(
                        j + 0.5,
                        self.grid_length - 1 - i + y_offset,
                        room_name,
                        ha="center",
                        va="center",
                        fontsize=cell_text_size,
                        color="black",
                        weight="bold",
                    )
                    y_offset -= line_height * 1.5

                    # Display each static object separately with blue background
                    for obj in static_objs:
                        if y_offset > 0.1:  # Only display if there's space
                            plt.text(
                                j + 0.5,
                                self.grid_length - 1 - i + y_offset,
                                obj,
                                ha="center",
                                va="center",
                                fontsize=cell_text_size - 2,
                                color="black",
                                weight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.15",
                                    facecolor=colors["static"],
                                    alpha=0.8,
                                ),
                            )
                            y_offset -= line_height

                    # Display each moving object separately with green background
                    for obj in moving_objs:
                        if y_offset > 0.1:  # Only display if there's space
                            plt.text(
                                j + 0.5,
                                self.grid_length - 1 - i + y_offset,
                                obj,
                                ha="center",
                                va="center",
                                fontsize=cell_text_size - 2,
                                color="black",
                                weight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.15",
                                    facecolor=colors["moving"],
                                    alpha=0.8,
                                ),
                            )
                            y_offset -= line_height

                    # Display agent with purple background
                    for obj in agent_objs:
                        if y_offset > 0.1:  # Only display if there's space
                            plt.text(
                                j + 0.5,
                                self.grid_length - 1 - i + y_offset,
                                obj,
                                ha="center",
                                va="center",
                                fontsize=cell_text_size - 1,
                                color="black",
                                weight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.15",
                                    facecolor=colors["agent"],
                                    alpha=0.8,
                                ),
                            )

            # Draw walls with thicker lines
            self._draw_walls()

            plt.gca().set_xlim(0, self.grid_length)
            plt.gca().set_ylim(0, self.grid_length)
            plt.gca().set_aspect("equal")

            # Remove axis ticks and labels
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

            plt.grid(True, alpha=0.3)

            plt.title(f"Bird Eye View - Step {self.current_step}")

            # Save figure
            if save_fig_dir is not None:
                os.makedirs(save_fig_dir, exist_ok=True)
                filename = f"bird-eye-view_step_{str(self.current_step).zfill(3)}.{image_format}"
                plt.savefig(
                    os.path.join(save_fig_dir, filename), dpi=150, bbox_inches="tight"
                )

            plt.show()

    def separate_overlapping_nodes(self, pos, min_distance=0.1, max_iterations=50):
        """Separate overlapping nodes by applying small adjustments to their positions."""
        import random

        import numpy as np

        nodes = list(pos.keys())
        adjusted_pos = pos.copy()

        for iteration in range(max_iterations):
            overlaps_found = False

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node1, node2 = nodes[i], nodes[j]
                    pos1 = np.array(adjusted_pos[node1])
                    pos2 = np.array(adjusted_pos[node2])

                    distance = np.linalg.norm(pos1 - pos2)

                    if distance < min_distance:
                        overlaps_found = True

                        # Calculate separation vector
                        if distance == 0:
                            # If nodes are exactly on top of each other, use random direction
                            angle = random.uniform(0, 2 * np.pi)
                            separation = (
                                np.array([np.cos(angle), np.sin(angle)]) * min_distance
                            )
                        else:
                            # Move nodes apart along the line connecting them
                            direction = (pos1 - pos2) / distance
                            separation = direction * (min_distance - distance) / 2

                        # Apply separation
                        adjusted_pos[node1] = pos1 + separation
                        adjusted_pos[node2] = pos2 - separation

            if not overlaps_found:
                break

        return adjusted_pos

    def _render_graph(
        self,
        figsize: tuple[int, int] = (12, 12),
        save_fig_dir: str = "./DEBUG/",
        image_format: str = "png",
        layout: str = "kamada_kawai",
    ) -> None:
        """Render the RDF graph as a network visualization."""
        # Get RDF graph and convert to networkx
        rdf_graph = self.get_rdf_graph()
        triples = rdf_to_list(rdf_graph)

        # Create networkx graph
        G = nx.DiGraph()

        # Add nodes and edges
        for subject, predicate, obj in triples:
            G.add_edge(subject, obj, label=predicate)

        # Create figure
        plt.figure(figsize=figsize)

        # Define node colors based on type
        node_colors = []
        color_mapping = {
            "agent": "#D8BFD8",  # Thistle (Soft Purple)
            "wall": "#D3D3D3",  # Light Gray
        }

        # Color rooms
        room_color = "#FFE4B5"  # Moccasin (Soft Gold)
        static_color = "#87CEFA"  # Light Blue
        moving_color = "#90EE90"  # Light Green

        for node in G.nodes():
            if node in self.room_names:
                node_colors.append(room_color)
            elif node in self.static_names:
                node_colors.append(static_color)
            elif node in self.moving_names:
                node_colors.append(moving_color)
            elif node == "agent":
                node_colors.append(color_mapping["agent"])
            elif node == "wall":
                node_colors.append(color_mapping["wall"])
            else:
                node_colors.append("#FFFFFF")  # White for unknown

        # Choose layout with better spacing
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            raise ValueError(f"Unknown layout: {layout}.")

        # Separate overlapping nodes
        pos = self.separate_overlapping_nodes(
            pos, min_distance=0.15, max_iterations=100
        )

        # Draw nodes with larger size for better visibility
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            edgecolors="black",
            linewidths=2,
        )

        # Draw node labels with consistent styling
        nx.draw_networkx_labels(
            G, pos, font_size=12, font_weight="bold", font_color="black"
        )

        # Draw edges with better spacing
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="gray",
            arrows=True,
            arrowsize=25,
            arrowstyle="->",
            alpha=0.7,
            width=2,
            connectionstyle="arc3,rad=0.1",  # Slight curve to avoid overlap
        )

        # Draw edge labels with consistent black text and same font size
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels,
            font_size=12,
            font_color="black",
            font_weight="bold",
            alpha=1.0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        plt.title(
            f"Graph View - Step {self.current_step}",
            fontsize=16,
            fontweight="bold",
        )
        plt.axis("off")
        plt.tight_layout()

        # Save figure
        if save_fig_dir is not None:
            os.makedirs(save_fig_dir, exist_ok=True)
            filename = (
                f"graph-view_step_{str(self.current_step).zfill(3)}.{image_format}"
            )
            plt.savefig(
                os.path.join(save_fig_dir, filename), dpi=150, bbox_inches="tight"
            )

        plt.show()

    def _draw_walls(self):
        """Draw outer walls and inner walls with thick black lines."""
        # Outer walls - very thick black
        wall_thickness = 8

        # Top wall
        plt.plot(
            [0, self.grid_length],
            [self.grid_length, self.grid_length],
            "k-",
            linewidth=wall_thickness,
        )
        # Bottom wall
        plt.plot([0, self.grid_length], [0, 0], "k-", linewidth=wall_thickness)
        # Left wall
        plt.plot([0, 0], [0, self.grid_length], "k-", linewidth=wall_thickness)
        # Right wall
        plt.plot(
            [self.grid_length, self.grid_length],
            [0, self.grid_length],
            "k-",
            linewidth=wall_thickness,
        )

        inner_wall_thickness = 4

        # Draw active inner walls based on current patterns
        for wall, pattern in self.wall_configs.items():
            pattern_length = len(pattern)
            pattern_index = self.current_step % pattern_length
            is_wall_active = pattern[pattern_index] == 1

            if is_wall_active:
                room1, room2, wall_type = wall
                pos1 = self.room_positions[room1]
                pos2 = self.room_positions[room2]

                if wall_type == "horizontal":
                    # Wall between vertically adjacent rooms
                    i1, j1 = pos1
                    i2, j2 = pos2
                    y = self.grid_length - max(i1, i2)  # Convert to plot coordinates
                    x_start = j1
                    x_end = j1 + 1
                    plt.plot(
                        [x_start, x_end], [y, y], "k-", linewidth=inner_wall_thickness
                    )

                else:  # vertical
                    # Wall between horizontally adjacent rooms
                    i1, j1 = pos1
                    i2, j2 = pos2
                    x = max(j1, j2)  # Convert to plot coordinates
                    y_start = self.grid_length - i1 - 1
                    y_end = self.grid_length - i1
                    plt.plot(
                        [x, x], [y_start, y_end], "k-", linewidth=inner_wall_thickness
                    )

    def get_rdf_graph(self):
        """Get current state as RDF graph."""
        g = Graph()

        # Add room connections
        for room_name, connections in self.room_connections.items():
            for direction, connected in connections.items():
                g.add((URIRef(room_name), URIRef(direction), URIRef(connected)))
        # Add object locations
        for obj_name, location in self.static_locations.items():
            g.add((URIRef(obj_name), URIRef("at_location"), URIRef(location)))

        for obj_name, location in self.moving_locations.items():
            g.add((URIRef(obj_name), URIRef("at_location"), URIRef(location)))

        # Add agent location
        g.add((URIRef("agent"), URIRef("at_location"), URIRef(self.agent_location)))

        return g
```

## Environment config generator



```python
"""Env Creator."""

import random
from itertools import combinations

from .utils import read_lines
from .utils import write_json_prod as write_json


class RoomCreator:
    def __init__(
        self,
        filename: str = "dev",
        grid_length: int = 3,
        num_static_objects: int = 3,
        num_moving_objects: int = 3,
        num_inner_walls: int = 3,
        seed: int = 42,
    ) -> None:
        """Create a simplified square room environment.

        Args:
            filename: Config filename to save
            grid_length: Size of square grid (must be odd)
            num_static_objects: Number of static objects
            num_moving_objects: Number of moving objects
            num_inner_walls: Number of inner walls to select
            seed: Random seed for reproducibility
        """
        assert grid_length % 2 == 1, "grid_length must be odd"

        self.filename = filename
        self.grid_length = grid_length
        self.num_rooms = grid_length**2
        self.num_static_objects = num_static_objects
        self.num_moving_objects = num_moving_objects
        self.num_inner_walls = num_inner_walls
        self.seed = seed

        # Define the 10 possible wall patterns
        self.wall_patterns = [
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
            [1, 0],
            [1, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 0, 1],
        ]

        random.seed(self.seed)

    def run(self):
        """Create and save the room configuration."""
        self._create_room_grid()
        self._create_room_connections()
        self._create_object_configs()
        self._create_wall_configs()
        self._create_question_list()
        self._save_config()

    def _create_room_grid(self):
        """Create square grid of rooms."""
        # Load names
        room_names = read_lines("data/room_names.txt")[: self.num_rooms]
        static_names = read_lines("data/static_objects.txt")[: self.num_static_objects]
        moving_names = read_lines("data/moving_objects.txt")[: self.num_moving_objects]

        self.room_names = room_names
        self.static_names = static_names
        self.moving_names = moving_names

        # Create grid positions
        self.room_positions = {}
        for i in range(self.grid_length):
            for j in range(self.grid_length):
                room_idx = i * self.grid_length + j
                self.room_positions[room_names[room_idx]] = (i, j)

    def _create_room_connections(self):
        """Create room connections with outer walls."""
        self.room_connections = {}

        for room_name, (i, j) in self.room_positions.items():
            connections = {
                "north": "wall",
                "east": "wall",
                "south": "wall",
                "west": "wall",
            }

            # North
            if i > 0:
                north_idx = (i - 1) * self.grid_length + j
                connections["north"] = self.room_names[north_idx]

            # East
            if j < self.grid_length - 1:
                east_idx = i * self.grid_length + (j + 1)
                connections["east"] = self.room_names[east_idx]

            # South
            if i < self.grid_length - 1:
                south_idx = (i + 1) * self.grid_length + j
                connections["south"] = self.room_names[south_idx]

            # West
            if j > 0:
                west_idx = i * self.grid_length + (j - 1)
                connections["west"] = self.room_names[west_idx]

            self.room_connections[room_name] = connections

    def _create_object_configs(self):
        """Create object initial locations and movement preferences."""
        # First, determine the center room where agent will start
        center_idx = self.grid_length // 2
        center_room = self.room_names[center_idx * self.grid_length + center_idx]
        self.agent_location = center_room

        # Available rooms exclude the center room (where agent starts)
        available_rooms = [room for room in self.room_names if room != center_room]

        # Static objects - ensure each is in a different room
        self.static_locations = {}
        for obj_name in self.static_names:
            room = random.choice(available_rooms)
            self.static_locations[obj_name] = room
            available_rooms.remove(room)

        # Moving objects - ensure each is in a different room from static objects and each other
        self.moving_locations = {}
        self.movement_preferences = {}
        directions = ["north", "east", "south", "west"]

        for obj_name in self.moving_names:
            room = random.choice(available_rooms)
            self.moving_locations[obj_name] = room
            available_rooms.remove(room)

            preferences = directions.copy()
            random.shuffle(preferences)
            self.movement_preferences[obj_name] = preferences

    def _set_agent_location(self):
        """Set agent starting location to center room."""
        # Calculate center position
        center_idx = self.grid_length // 2
        center_room = self.room_names[center_idx * self.grid_length + center_idx]
        self.agent_location = center_room

    def _create_wall_configs(self):
        """Create periodic inner wall configurations."""
        # Get all possible inner wall positions
        possible_walls = []

        # Horizontal walls (between vertically adjacent rooms)
        for i in range(self.grid_length - 1):
            for j in range(self.grid_length):
                room1 = i * self.grid_length + j
                room2 = (i + 1) * self.grid_length + j
                possible_walls.append(
                    (self.room_names[room1], self.room_names[room2], "horizontal")
                )

        # Vertical walls (between horizontally adjacent rooms)
        for i in range(self.grid_length):
            for j in range(self.grid_length - 1):
                room1 = i * self.grid_length + j
                room2 = i * self.grid_length + (j + 1)
                possible_walls.append(
                    (self.room_names[room1], self.room_names[room2], "vertical")
                )

        # Ensure we don't select more walls than possible
        max_walls = len(possible_walls)
        if self.num_inner_walls > max_walls:
            self.num_inner_walls = max_walls
            print(f"Warning: Reduced num_inner_walls to {max_walls} (maximum possible)")

        # Select walls that maintain connectivity
        self.selected_walls = []
        attempts = 0
        max_attempts = 1000

        while (
            len(self.selected_walls) < self.num_inner_walls and attempts < max_attempts
        ):
            # Randomly select remaining walls
            remaining_walls = [
                w for w in possible_walls if w not in self.selected_walls
            ]
            candidate_wall = random.choice(remaining_walls)

            # Test if adding this wall maintains connectivity
            test_walls = self.selected_walls + [candidate_wall]
            if self._test_connectivity_with_all_walls_on(test_walls):
                self.selected_walls.append(candidate_wall)

            attempts += 1

        if len(self.selected_walls) < self.num_inner_walls:
            print(
                f"Warning: Could only select {len(self.selected_walls)} walls that maintain connectivity"
            )

        # Assign patterns to selected walls - convert tuples to strings for JSON serialization
        self.wall_configs = {}
        for wall in self.selected_walls:
            pattern = random.choice(self.wall_patterns)
            # Convert tuple to string key for JSON compatibility
            wall_key = f"{wall[0]}|{wall[1]}|{wall[2]}"
            self.wall_configs[wall_key] = pattern

    def _test_connectivity_with_all_walls_on(self, walls):
        """Test if rooms remain connected when all specified walls are active."""
        # Create temporary room connections with all walls on
        temp_connections = {}
        for room_name in self.room_names:
            temp_connections[room_name] = self.room_connections[room_name].copy()

        # Apply all walls
        for room1, room2, wall_type in walls:
            if wall_type == "horizontal":
                temp_connections[room1]["south"] = "wall"
                temp_connections[room2]["north"] = "wall"
            else:  # vertical
                temp_connections[room1]["east"] = "wall"
                temp_connections[room2]["west"] = "wall"

        # Check connectivity using DFS
        visited = set()
        start_room = self.room_names[0]
        self._dfs(start_room, temp_connections, visited)

        return len(visited) == len(self.room_names)

    def _dfs(self, room, connections, visited):
        """Depth-first search for connectivity check."""
        visited.add(room)
        for direction in ["north", "east", "south", "west"]:
            neighbor = connections[room][direction]
            if neighbor != "wall" and neighbor not in visited:
                self._dfs(neighbor, connections, visited)

    def _create_question_list(self):
        """Create list of exactly 100 random questions."""
        all_objects = self.static_names + self.moving_names
        self.question_objects = [random.choice(all_objects) for _ in range(100)]

    def _save_config(self):
        """Save configuration to JSON file."""
        config = {
            "grid_length": self.grid_length,
            "room_names": self.room_names,
            "room_positions": self.room_positions,
            "room_connections": self.room_connections,
            "static_names": self.static_names,
            "moving_names": self.moving_names,
            "static_locations": self.static_locations,
            "moving_locations": self.moving_locations,
            "movement_preferences": self.movement_preferences,
            "agent_location": self.agent_location,
            "selected_walls": self.selected_walls,
            "wall_configs": self.wall_configs,
            "question_objects": self.question_objects,
            "seed": self.seed,
        }

        write_json(config, f"data/room-config-{self.filename}.json")
        print(f"Saved configuration to room-config-{self.filename}.json")
        print(f"Selected {len(self.selected_walls)} inner walls with periodic patterns")

```

## Env config used in the experiment

```json
{
    "grid_length": 7,
    "room_names": [
        "living",
        "kitchen",
        "bedroom",
        "bathroom",
        "office",
        "den",
        "study",
        "garage",
        "basement",
        "attic",
        "dining",
        "family",
        "guest",
        "master",
        "laundry",
        "pantry",
        "closet",
        "foyer",
        "hallway",
        "porch",
        "deck",
        "patio",
        "balcony",
        "sunroom",
        "library",
        "nursery",
        "playroom",
        "gameroom",
        "studio",
        "workshop",
        "gym",
        "spa",
        "sauna",
        "cellar",
        "storage",
        "utility",
        "mudroom",
        "powder",
        "wardrobe",
        "loft",
        "cabin",
        "lodge",
        "cottage",
        "suite",
        "parlor",
        "lounge",
        "bar",
        "cafe",
        "nook"
    ],
    "room_positions": {
        "living": [
            0,
            0
        ],
        "kitchen": [
            0,
            1
        ],
        "bedroom": [
            0,
            2
        ],
        "bathroom": [
            0,
            3
        ],
        "office": [
            0,
            4
        ],
        "den": [
            0,
            5
        ],
        "study": [
            0,
            6
        ],
        "garage": [
            1,
            0
        ],
        "basement": [
            1,
            1
        ],
        "attic": [
            1,
            2
        ],
        "dining": [
            1,
            3
        ],
        "family": [
            1,
            4
        ],
        "guest": [
            1,
            5
        ],
        "master": [
            1,
            6
        ],
        "laundry": [
            2,
            0
        ],
        "pantry": [
            2,
            1
        ],
        "closet": [
            2,
            2
        ],
        "foyer": [
            2,
            3
        ],
        "hallway": [
            2,
            4
        ],
        "porch": [
            2,
            5
        ],
        "deck": [
            2,
            6
        ],
        "patio": [
            3,
            0
        ],
        "balcony": [
            3,
            1
        ],
        "sunroom": [
            3,
            2
        ],
        "library": [
            3,
            3
        ],
        "nursery": [
            3,
            4
        ],
        "playroom": [
            3,
            5
        ],
        "gameroom": [
            3,
            6
        ],
        "studio": [
            4,
            0
        ],
        "workshop": [
            4,
            1
        ],
        "gym": [
            4,
            2
        ],
        "spa": [
            4,
            3
        ],
        "sauna": [
            4,
            4
        ],
        "cellar": [
            4,
            5
        ],
        "storage": [
            4,
            6
        ],
        "utility": [
            5,
            0
        ],
        "mudroom": [
            5,
            1
        ],
        "powder": [
            5,
            2
        ],
        "wardrobe": [
            5,
            3
        ],
        "loft": [
            5,
            4
        ],
        "cabin": [
            5,
            5
        ],
        "lodge": [
            5,
            6
        ],
        "cottage": [
            6,
            0
        ],
        "suite": [
            6,
            1
        ],
        "parlor": [
            6,
            2
        ],
        "lounge": [
            6,
            3
        ],
        "bar": [
            6,
            4
        ],
        "cafe": [
            6,
            5
        ],
        "nook": [
            6,
            6
        ]
    },
    "room_connections": {
        "living": {
            "north": "wall",
            "east": "kitchen",
            "south": "garage",
            "west": "wall"
        },
        "kitchen": {
            "north": "wall",
            "east": "bedroom",
            "south": "basement",
            "west": "living"
        },
        "bedroom": {
            "north": "wall",
            "east": "bathroom",
            "south": "attic",
            "west": "kitchen"
        },
        "bathroom": {
            "north": "wall",
            "east": "office",
            "south": "dining",
            "west": "bedroom"
        },
        "office": {
            "north": "wall",
            "east": "den",
            "south": "family",
            "west": "bathroom"
        },
        "den": {
            "north": "wall",
            "east": "study",
            "south": "guest",
            "west": "office"
        },
        "study": {
            "north": "wall",
            "east": "wall",
            "south": "master",
            "west": "den"
        },
        "garage": {
            "north": "living",
            "east": "basement",
            "south": "laundry",
            "west": "wall"
        },
        "basement": {
            "north": "kitchen",
            "east": "attic",
            "south": "pantry",
            "west": "garage"
        },
        "attic": {
            "north": "bedroom",
            "east": "dining",
            "south": "closet",
            "west": "basement"
        },
        "dining": {
            "north": "bathroom",
            "east": "family",
            "south": "foyer",
            "west": "attic"
        },
        "family": {
            "north": "office",
            "east": "guest",
            "south": "hallway",
            "west": "dining"
        },
        "guest": {
            "north": "den",
            "east": "master",
            "south": "porch",
            "west": "family"
        },
        "master": {
            "north": "study",
            "east": "wall",
            "south": "deck",
            "west": "guest"
        },
        "laundry": {
            "north": "garage",
            "east": "pantry",
            "south": "patio",
            "west": "wall"
        },
        "pantry": {
            "north": "basement",
            "east": "closet",
            "south": "balcony",
            "west": "laundry"
        },
        "closet": {
            "north": "attic",
            "east": "foyer",
            "south": "sunroom",
            "west": "pantry"
        },
        "foyer": {
            "north": "dining",
            "east": "hallway",
            "south": "library",
            "west": "closet"
        },
        "hallway": {
            "north": "family",
            "east": "porch",
            "south": "nursery",
            "west": "foyer"
        },
        "porch": {
            "north": "guest",
            "east": "deck",
            "south": "playroom",
            "west": "hallway"
        },
        "deck": {
            "north": "master",
            "east": "wall",
            "south": "gameroom",
            "west": "porch"
        },
        "patio": {
            "north": "laundry",
            "east": "balcony",
            "south": "studio",
            "west": "wall"
        },
        "balcony": {
            "north": "pantry",
            "east": "sunroom",
            "south": "workshop",
            "west": "patio"
        },
        "sunroom": {
            "north": "closet",
            "east": "library",
            "south": "gym",
            "west": "balcony"
        },
        "library": {
            "north": "foyer",
            "east": "nursery",
            "south": "spa",
            "west": "sunroom"
        },
        "nursery": {
            "north": "hallway",
            "east": "playroom",
            "south": "sauna",
            "west": "library"
        },
        "playroom": {
            "north": "porch",
            "east": "gameroom",
            "south": "cellar",
            "west": "nursery"
        },
        "gameroom": {
            "north": "deck",
            "east": "wall",
            "south": "storage",
            "west": "playroom"
        },
        "studio": {
            "north": "patio",
            "east": "workshop",
            "south": "utility",
            "west": "wall"
        },
        "workshop": {
            "north": "balcony",
            "east": "gym",
            "south": "mudroom",
            "west": "studio"
        },
        "gym": {
            "north": "sunroom",
            "east": "spa",
            "south": "powder",
            "west": "workshop"
        },
        "spa": {
            "north": "library",
            "east": "sauna",
            "south": "wardrobe",
            "west": "gym"
        },
        "sauna": {
            "north": "nursery",
            "east": "cellar",
            "south": "loft",
            "west": "spa"
        },
        "cellar": {
            "north": "playroom",
            "east": "storage",
            "south": "cabin",
            "west": "sauna"
        },
        "storage": {
            "north": "gameroom",
            "east": "wall",
            "south": "lodge",
            "west": "cellar"
        },
        "utility": {
            "north": "studio",
            "east": "mudroom",
            "south": "cottage",
            "west": "wall"
        },
        "mudroom": {
            "north": "workshop",
            "east": "powder",
            "south": "suite",
            "west": "utility"
        },
        "powder": {
            "north": "gym",
            "east": "wardrobe",
            "south": "parlor",
            "west": "mudroom"
        },
        "wardrobe": {
            "north": "spa",
            "east": "loft",
            "south": "lounge",
            "west": "powder"
        },
        "loft": {
            "north": "sauna",
            "east": "cabin",
            "south": "bar",
            "west": "wardrobe"
        },
        "cabin": {
            "north": "cellar",
            "east": "lodge",
            "south": "cafe",
            "west": "loft"
        },
        "lodge": {
            "north": "storage",
            "east": "wall",
            "south": "nook",
            "west": "cabin"
        },
        "cottage": {
            "north": "utility",
            "east": "suite",
            "south": "wall",
            "west": "wall"
        },
        "suite": {
            "north": "mudroom",
            "east": "parlor",
            "south": "wall",
            "west": "cottage"
        },
        "parlor": {
            "north": "powder",
            "east": "lounge",
            "south": "wall",
            "west": "suite"
        },
        "lounge": {
            "north": "wardrobe",
            "east": "bar",
            "south": "wall",
            "west": "parlor"
        },
        "bar": {
            "north": "loft",
            "east": "cafe",
            "south": "wall",
            "west": "lounge"
        },
        "cafe": {
            "north": "cabin",
            "east": "nook",
            "south": "wall",
            "west": "bar"
        },
        "nook": {
            "north": "lodge",
            "east": "wall",
            "south": "wall",
            "west": "cafe"
        }
    },
    "static_names": [
        "chair",
        "table",
        "bed",
        "dresser",
        "bookshelf",
        "sofa",
        "desk",
        "wardrobe",
        "nightstand",
        "coffeetable",
        "diningtable",
        "armchair",
        "cabinet",
        "floorlamp",
        "ceilingfan",
        "window",
        "door",
        "wall"
    ],
    "moving_names": [
        "john",
        "mary",
        "david",
        "sarah",
        "michael",
        "lisa",
        "robert",
        "jennifer",
        "william",
        "amanda",
        "james",
        "jessica",
        "christopher",
        "ashley",
        "daniel",
        "emily",
        "matthew",
        "stephanie"
    ],
    "static_locations": {
        "chair": "nursery",
        "table": "studio",
        "bed": "bedroom",
        "dresser": "foyer",
        "bookshelf": "powder",
        "sofa": "mudroom",
        "desk": "gym",
        "wardrobe": "patio",
        "nightstand": "loft",
        "coffeetable": "gameroom",
        "diningtable": "nook",
        "armchair": "laundry",
        "cabinet": "parlor",
        "floorlamp": "attic",
        "ceilingfan": "sunroom",
        "window": "dining",
        "door": "garage",
        "wall": "sauna"
    },
    "moving_locations": {
        "john": "cottage",
        "mary": "master",
        "david": "cafe",
        "sarah": "hallway",
        "michael": "cellar",
        "lisa": "kitchen",
        "robert": "lounge",
        "jennifer": "porch",
        "william": "office",
        "amanda": "pantry",
        "james": "den",
        "jessica": "basement",
        "christopher": "family",
        "ashley": "spa",
        "daniel": "utility",
        "emily": "guest",
        "matthew": "workshop",
        "stephanie": "study"
    },
    "movement_preferences": {
        "john": [
            "east",
            "north",
            "west",
            "south"
        ],
        "mary": [
            "east",
            "west",
            "south",
            "north"
        ],
        "david": [
            "west",
            "north",
            "east",
            "south"
        ],
        "sarah": [
            "south",
            "north",
            "east",
            "west"
        ],
        "michael": [
            "north",
            "south",
            "east",
            "west"
        ],
        "lisa": [
            "south",
            "east",
            "west",
            "north"
        ],
        "robert": [
            "west",
            "east",
            "south",
            "north"
        ],
        "jennifer": [
            "north",
            "west",
            "south",
            "east"
        ],
        "william": [
            "west",
            "north",
            "south",
            "east"
        ],
        "amanda": [
            "north",
            "west",
            "south",
            "east"
        ],
        "james": [
            "west",
            "south",
            "east",
            "north"
        ],
        "jessica": [
            "north",
            "east",
            "west",
            "south"
        ],
        "christopher": [
            "east",
            "north",
            "west",
            "south"
        ],
        "ashley": [
            "south",
            "east",
            "north",
            "west"
        ],
        "daniel": [
            "south",
            "north",
            "west",
            "east"
        ],
        "emily": [
            "south",
            "west",
            "north",
            "east"
        ],
        "matthew": [
            "east",
            "south",
            "west",
            "north"
        ],
        "stephanie": [
            "south",
            "east",
            "west",
            "north"
        ]
    },
    "agent_location": "library",
    "selected_walls": [
        [
            "workshop",
            "gym",
            "vertical"
        ],
        [
            "utility",
            "cottage",
            "horizontal"
        ],
        [
            "gym",
            "spa",
            "vertical"
        ],
        [
            "gym",
            "powder",
            "horizontal"
        ],
        [
            "gameroom",
            "storage",
            "horizontal"
        ],
        [
            "parlor",
            "lounge",
            "vertical"
        ],
        [
            "closet",
            "foyer",
            "vertical"
        ],
        [
            "lounge",
            "bar",
            "vertical"
        ],
        [
            "balcony",
            "sunroom",
            "vertical"
        ],
        [
            "sauna",
            "cellar",
            "vertical"
        ],
        [
            "garage",
            "basement",
            "vertical"
        ],
        [
            "dining",
            "foyer",
            "horizontal"
        ],
        [
            "bathroom",
            "office",
            "vertical"
        ],
        [
            "pantry",
            "balcony",
            "horizontal"
        ],
        [
            "powder",
            "wardrobe",
            "vertical"
        ],
        [
            "basement",
            "attic",
            "vertical"
        ],
        [
            "playroom",
            "cellar",
            "horizontal"
        ],
        [
            "powder",
            "parlor",
            "horizontal"
        ],
        [
            "bedroom",
            "attic",
            "horizontal"
        ],
        [
            "living",
            "kitchen",
            "vertical"
        ],
        [
            "foyer",
            "library",
            "horizontal"
        ],
        [
            "porch",
            "deck",
            "vertical"
        ],
        [
            "hallway",
            "nursery",
            "horizontal"
        ],
        [
            "spa",
            "wardrobe",
            "horizontal"
        ],
        [
            "cellar",
            "storage",
            "vertical"
        ],
        [
            "guest",
            "porch",
            "horizontal"
        ],
        [
            "workshop",
            "mudroom",
            "horizontal"
        ],
        [
            "cabin",
            "cafe",
            "horizontal"
        ],
        [
            "office",
            "family",
            "horizontal"
        ],
        [
            "basement",
            "pantry",
            "horizontal"
        ],
        [
            "spa",
            "sauna",
            "vertical"
        ],
        [
            "balcony",
            "workshop",
            "horizontal"
        ],
        [
            "hallway",
            "porch",
            "vertical"
        ],
        [
            "playroom",
            "gameroom",
            "vertical"
        ],
        [
            "study",
            "master",
            "horizontal"
        ],
        [
            "lodge",
            "nook",
            "horizontal"
        ]
    ],
    "wall_configs": {
        "workshop|gym|vertical": [
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0
        ],
        "utility|cottage|horizontal": [
            1,
            1,
            1,
            0,
            1
        ],
        "gym|spa|vertical": [
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1
        ],
        "gym|powder|horizontal": [
            1,
            1,
            1,
            0,
            1
        ],
        "gameroom|storage|horizontal": [
            1,
            1,
            0,
            1,
            0,
            1
        ],
        "parlor|lounge|vertical": [
            1,
            0
        ],
        "closet|foyer|vertical": [
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0
        ],
        "lounge|bar|vertical": [
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1
        ],
        "balcony|sunroom|vertical": [
            1,
            1,
            0,
            1,
            0,
            1
        ],
        "sauna|cellar|vertical": [
            1,
            1,
            1,
            0,
            1
        ],
        "garage|basement|vertical": [
            1,
            1,
            1,
            0,
            1
        ],
        "dining|foyer|horizontal": [
            0,
            0,
            0,
            0,
            0,
            1,
            1
        ],
        "bathroom|office|vertical": [
            1,
            0,
            0
        ],
        "pantry|balcony|horizontal": [
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1
        ],
        "powder|wardrobe|vertical": [
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0
        ],
        "basement|attic|vertical": [
            1,
            1,
            0,
            0,
            1
        ],
        "playroom|cellar|horizontal": [
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0
        ],
        "powder|parlor|horizontal": [
            1,
            1,
            0,
            1,
            0,
            1
        ],
        "bedroom|attic|horizontal": [
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0
        ],
        "living|kitchen|vertical": [
            1,
            1,
            0,
            0,
            1
        ],
        "foyer|library|horizontal": [
            1,
            1,
            0,
            0
        ],
        "porch|deck|vertical": [
            1,
            0,
            0
        ],
        "hallway|nursery|horizontal": [
            1,
            0
        ],
        "spa|wardrobe|horizontal": [
            1,
            1,
            1,
            0,
            1
        ],
        "cellar|storage|vertical": [
            0,
            0,
            0,
            0,
            0,
            1,
            1
        ],
        "guest|porch|horizontal": [
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1
        ],
        "workshop|mudroom|horizontal": [
            1,
            1,
            1,
            0,
            1
        ],
        "cabin|cafe|horizontal": [
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1
        ],
        "office|family|horizontal": [
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0
        ],
        "basement|pantry|horizontal": [
            1,
            0,
            0
        ],
        "spa|sauna|vertical": [
            1,
            1,
            0,
            1,
            0,
            1
        ],
        "balcony|workshop|horizontal": [
            1,
            1,
            0,
            0,
            1
        ],
        "hallway|porch|vertical": [
            1,
            1,
            0,
            0
        ],
        "playroom|gameroom|vertical": [
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1
        ],
        "study|master|horizontal": [
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1
        ],
        "lodge|nook|horizontal": [
            1,
            0
        ]
    },
    "question_objects": [
        "floorlamp",
        "dresser",
        "diningtable",
        "diningtable",
        "sarah",
        "emily",
        "door",
        "wardrobe",
        "james",
        "armchair",
        "chair",
        "christopher",
        "william",
        "daniel",
        "mary",
        "michael",
        "robert",
        "door",
        "coffeetable",
        "stephanie",
        "chair",
        "jessica",
        "sofa",
        "sarah",
        "bed",
        "matthew",
        "wall",
        "nightstand",
        "window",
        "christopher",
        "michael",
        "john",
        "michael",
        "nightstand",
        "mary",
        "robert",
        "william",
        "sofa",
        "chair",
        "cabinet",
        "sarah",
        "diningtable",
        "window",
        "ceilingfan",
        "james",
        "robert",
        "william",
        "bed",
        "jennifer",
        "william",
        "bed",
        "diningtable",
        "james",
        "bookshelf",
        "door",
        "diningtable",
        "james",
        "emily",
        "ashley",
        "stephanie",
        "chair",
        "bed",
        "ashley",
        "david",
        "mary",
        "jessica",
        "dresser",
        "william",
        "cabinet",
        "stephanie",
        "sofa",
        "nightstand",
        "chair",
        "jennifer",
        "william",
        "david",
        "chair",
        "floorlamp",
        "chair",
        "chair",
        "emily",
        "desk",
        "cabinet",
        "wardrobe",
        "cabinet",
        "mary",
        "wall",
        "armchair",
        "desk",
        "christopher",
        "jennifer",
        "sofa",
        "table",
        "wall",
        "james",
        "wardrobe",
        "door",
        "nightstand",
        "emily",
        "michael"
    ],
    "seed": 0
}
```