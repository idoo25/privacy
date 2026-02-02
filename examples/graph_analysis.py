#!/usr/bin/env python3
"""
Example: Graph-Based Route Analysis

Demonstrates how to use the StationGraph for efficient route analysis
using DFS, BFS, and other graph algorithms.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import numpy as np

from src.station_graph import StationGraph
from src.trajectory import TrajectoryBuilder
from src.data_structures import DetectionEvent


def example_graph_construction():
    """Example: Build station graph from trajectories."""
    print("=" * 60)
    print("Example 1: Building Station Graph from Trajectories")
    print("=" * 60)
    
    # Create trajectory builder and add sample data
    builder = TrajectoryBuilder()
    base_time = datetime.now()
    
    # Simulate passenger journeys
    journeys = [
        ["Central", "North", "Airport"],  # 10 passengers
        ["Central", "South", "Beach"],     # 8 passengers
        ["Airport", "Central", "South"],   # 5 passengers
        ["North", "Central", "South"],     # 12 passengers
        ["Beach", "South", "Central"],     # 7 passengers
    ]
    
    passenger_counts = [10, 8, 5, 12, 7]
    
    for journey_idx, (journey, count) in enumerate(zip(journeys, passenger_counts)):
        for p in range(count):
            token = f"passenger_{journey_idx * 100 + p:05d}"
            for i, station in enumerate(journey):
                event = DetectionEvent(
                    token=token,
                    timestamp=base_time + timedelta(minutes=i*15, seconds=p),
                    station_id=station,
                    camera_id=f"cam_{station}",
                    confidence=0.95
                )
                builder.add_detection(event)
    
    # Build graph from trajectories
    graph = StationGraph()
    trajectories = builder.get_trajectories(min_length=2)
    graph.build_from_trajectories(trajectories)
    
    print(f"Stations: {graph.station_count}")
    print(f"Edges: {graph.edge_count}")
    print(f"Total flow: {graph.total_flow}")
    print()


def example_dfs_path_finding():
    """Example: Find all paths using DFS."""
    print("=" * 60)
    print("Example 2: Finding All Paths with DFS")
    print("=" * 60)
    
    graph = StationGraph()
    
    # Create a transit network
    #     North
    #      / \
    # West - Central - East
    #      \ /
    #     South
    
    edges = [
        ("Central", "North"), ("North", "Central"),
        ("Central", "South"), ("South", "Central"),
        ("Central", "East"), ("East", "Central"),
        ("Central", "West"), ("West", "Central"),
        ("North", "West"), ("West", "North"),
        ("South", "East"), ("East", "South"),
    ]
    
    for from_s, to_s in edges:
        graph.add_edge(from_s, to_s, weight=10)
    
    # Find all paths from West to East
    print("All paths from West to East (max depth 4):")
    paths = list(graph.dfs_all_paths("West", "East", max_depth=4))
    
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: {' -> '.join(path)}")
    
    print(f"\nTotal paths found: {len(paths)}")
    print()


def example_bfs_shortest_path():
    """Example: Find shortest path using BFS."""
    print("=" * 60)
    print("Example 3: Finding Shortest Path with BFS")
    print("=" * 60)
    
    graph = StationGraph()
    
    # Create a larger network
    # A -> B -> C -> D -> E (long path)
    # A -> E (direct connection)
    
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")
    graph.add_edge("D", "E")
    graph.add_edge("A", "E")  # Direct shortcut
    
    # Find shortest path
    path = graph.bfs_shortest_path("A", "E")
    print(f"Shortest path from A to E: {' -> '.join(path)}")
    print(f"Path length: {len(path) - 1} hops")
    
    # Find path in reverse (no direct connection)
    path_reverse = graph.bfs_shortest_path("E", "A")
    print(f"Shortest path from E to A: {path_reverse}")
    print()


def example_hub_detection():
    """Example: Detect hub stations."""
    print("=" * 60)
    print("Example 4: Hub Station Detection")
    print("=" * 60)
    
    graph = StationGraph()
    
    # Create a hub-and-spoke network
    hub = "Central_Hub"
    spokes = ["Branch_A", "Branch_B", "Branch_C", "Branch_D", "Branch_E"]
    
    for spoke in spokes:
        graph.add_edge(hub, spoke, weight=100)
        graph.add_edge(spoke, hub, weight=80)
    
    # Add some inter-branch connections
    graph.add_edge("Branch_A", "Branch_B", weight=10)
    graph.add_edge("Branch_B", "Branch_C", weight=10)
    
    # Calculate centrality
    centrality = graph.get_station_centrality()
    
    print("Station Centrality Scores:")
    for station in sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True):
        print(f"  {station}: {centrality[station]:.4f}")
    
    # Find hubs
    hubs = graph.find_hub_stations(threshold=0.01)
    print(f"\nDetected hub stations (threshold=0.01): {hubs}")
    print()


def example_reachability_analysis():
    """Example: Analyze station reachability."""
    print("=" * 60)
    print("Example 5: Reachability Analysis")
    print("=" * 60)
    
    graph = StationGraph()
    
    # Create a linear metro line
    stations = ["Terminal_A", "Station_1", "Station_2", "Central", "Station_3", "Station_4", "Terminal_B"]
    
    for i in range(len(stations) - 1):
        graph.add_edge(stations[i], stations[i + 1])
        graph.add_edge(stations[i + 1], stations[i])  # Bidirectional
    
    # Analyze reachability from Central
    for max_hops in [1, 2, 3]:
        reachable = graph.get_reachable_stations("Central", max_hops=max_hops)
        print(f"From Central with {max_hops} hop(s): {sorted(reachable)}")
    
    print()


def example_scc_analysis():
    """Example: Strongly connected component analysis."""
    print("=" * 60)
    print("Example 6: Strongly Connected Components")
    print("=" * 60)
    
    graph = StationGraph()
    
    # Create two separate transit zones
    # Zone 1: A <-> B <-> C (circular)
    graph.add_edge("Zone1_A", "Zone1_B")
    graph.add_edge("Zone1_B", "Zone1_C")
    graph.add_edge("Zone1_C", "Zone1_A")
    
    # Zone 2: X <-> Y <-> Z (circular)
    graph.add_edge("Zone2_X", "Zone2_Y")
    graph.add_edge("Zone2_Y", "Zone2_Z")
    graph.add_edge("Zone2_Z", "Zone2_X")
    
    # One-way connection between zones
    graph.add_edge("Zone1_A", "Zone2_X")
    
    # Find SCCs
    sccs = graph.get_strongly_connected_stations()
    
    print("Strongly Connected Components:")
    for i, scc in enumerate(sccs, 1):
        print(f"  Component {i}: {sorted(scc)}")
    
    print(f"\nTotal components: {len(sccs)}")
    print()


def example_top_routes():
    """Example: Find top routes by traffic."""
    print("=" * 60)
    print("Example 7: Top Routes Analysis")
    print("=" * 60)
    
    graph = StationGraph()
    
    # Add routes with varying traffic
    routes = [
        ("Central", "Airport", 500),
        ("Central", "Downtown", 450),
        ("Downtown", "Central", 420),
        ("Airport", "Central", 380),
        ("Central", "University", 300),
        ("University", "Central", 280),
        ("Downtown", "Beach", 150),
        ("Beach", "Downtown", 130),
    ]
    
    for from_s, to_s, traffic in routes:
        graph.add_edge(from_s, to_s, weight=traffic)
    
    # Get top routes
    top = graph.get_top_routes(k=5)
    
    print("Top 5 Routes by Traffic:")
    for i, (from_s, to_s, count) in enumerate(top, 1):
        print(f"  {i}. {from_s} -> {to_s}: {count} passengers")
    
    print()


def example_flow_matrix():
    """Example: Generate flow matrix with vectorization."""
    print("=" * 60)
    print("Example 8: Vectorized Flow Matrix")
    print("=" * 60)
    
    graph = StationGraph()
    
    # Simple 3-station network
    graph.add_edge("A", "B", weight=100)
    graph.add_edge("B", "A", weight=80)
    graph.add_edge("B", "C", weight=60)
    graph.add_edge("C", "B", weight=50)
    graph.add_edge("A", "C", weight=30)
    
    # Get flow matrix
    matrix, stations = graph.get_flow_matrix_numpy()
    
    print(f"Stations: {stations}")
    print(f"\nFlow Matrix:")
    print(f"     {' '.join(f'{s:>5}' for s in stations)}")
    for i, station in enumerate(stations):
        row = ' '.join(f'{matrix[i,j]:>5}' for j in range(len(stations)))
        print(f"{station:>5} {row}")
    
    print()


def main():
    """Run all graph analysis examples."""
    print("\n" + "=" * 60)
    print("PrivacyFlow - Graph-Based Route Analysis")
    print("Demonstrating DFS, BFS, and Vectorized Operations")
    print("=" * 60 + "\n")
    
    example_graph_construction()
    example_dfs_path_finding()
    example_bfs_shortest_path()
    example_hub_detection()
    example_reachability_analysis()
    example_scc_analysis()
    example_top_routes()
    example_flow_matrix()
    
    print("=" * 60)
    print("All graph analysis examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
