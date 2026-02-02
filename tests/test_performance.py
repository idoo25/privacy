"""
Performance Tests

Benchmarks to validate optimizations in data structures and algorithms.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.privacy_token import PrivacyTokenGenerator
from src.data_structures import DetectionEvent, DailyTrajectory
from src.trajectory import TrajectoryBuilder
from src.station_graph import StationGraph


def generate_test_events(n_events: int, n_stations: int = 10, n_tokens: int = 100):
    """Generate test detection events."""
    events = []
    base_time = datetime.now()
    stations = [f"station_{i}" for i in range(n_stations)]
    tokens = [f"token_{i:08d}" for i in range(n_tokens)]
    
    for i in range(n_events):
        event = DetectionEvent(
            token=tokens[i % n_tokens],
            timestamp=base_time + timedelta(seconds=i),
            station_id=stations[i % n_stations],
            camera_id=f"cam_{i % 5}",
            confidence=0.95
        )
        events.append(event)
    
    return events


class TestTrajectoryBisectInsertion:
    """Test optimized bisect-based insertion in DailyTrajectory."""
    
    def test_bisect_insertion_maintains_order(self):
        """Verify events remain sorted after bisect insertion."""
        traj = DailyTrajectory(token="test_token")
        base_time = datetime.now()
        
        # Add events out of order
        timestamps = [5, 1, 3, 8, 2, 6, 4, 7]
        for ts in timestamps:
            event = DetectionEvent(
                token="test_token",
                timestamp=base_time + timedelta(seconds=ts),
                station_id=f"station_{ts}",
                camera_id="cam",
                confidence=0.9
            )
            traj.add_event(event)
        
        # Verify sorted order
        for i in range(len(traj.events) - 1):
            assert traj.events[i].timestamp <= traj.events[i + 1].timestamp
    
    def test_bisect_insertion_performance(self):
        """Benchmark bisect insertion vs sorting all events."""
        n_events = 1000
        base_time = datetime.now()
        
        # Time bisect-based insertion
        traj = DailyTrajectory(token="test")
        start = time.time()
        for i in range(n_events):
            event = DetectionEvent(
                token="test",
                timestamp=base_time + timedelta(seconds=np.random.randint(0, 10000)),
                station_id=f"station_{i % 10}",
                camera_id="cam",
                confidence=0.9
            )
            traj.add_event(event)
        bisect_time = time.time() - start
        
        # The bisect approach should complete in reasonable time
        # (O(n²) due to list insertion, but much better constant factors)
        assert bisect_time < 2.0, f"Bisect insertion took {bisect_time:.2f}s for {n_events} events"
        print(f"\nBisect insertion: {bisect_time:.4f}s for {n_events} events")


class TestTrajectoryBuilderIndexes:
    """Test indexed lookups in TrajectoryBuilder."""
    
    def test_station_index_lookup(self):
        """Verify station index provides correct results."""
        builder = TrajectoryBuilder()
        events = generate_test_events(1000, n_stations=10)
        
        for event in events:
            builder.add_detection(event)
        
        # Get events for a specific station
        station_events = builder.get_station_events("station_5")
        
        # Verify all returned events are from the correct station
        for event in station_events:
            assert event.station_id == "station_5"
        
        # Verify count matches
        expected_count = sum(1 for e in events if e.station_id == "station_5")
        assert len(station_events) == expected_count
    
    def test_time_window_binary_search(self):
        """Verify time window queries use binary search correctly."""
        builder = TrajectoryBuilder()
        base_time = datetime.now()
        
        # Add events spread over 1000 seconds
        for i in range(1000):
            event = DetectionEvent(
                token=f"token_{i % 50}",
                timestamp=base_time + timedelta(seconds=i),
                station_id=f"station_{i % 10}",
                camera_id="cam",
                confidence=0.9
            )
            builder.add_detection(event)
        
        # Query a specific time window
        start_time = base_time + timedelta(seconds=100)
        end_time = base_time + timedelta(seconds=200)
        
        window_events = builder.get_events_in_window(start_time, end_time)
        
        # Verify all events are within the window
        for event in window_events:
            assert start_time <= event.timestamp <= end_time
        
        # Verify count (should be ~101 events: seconds 100-200)
        assert len(window_events) == 101
    
    def test_count_unique_tokens_optimized(self):
        """Benchmark unique token counting."""
        builder = TrajectoryBuilder()
        events = generate_test_events(10000, n_stations=20, n_tokens=500)
        
        for event in events:
            builder.add_detection(event)
        
        # Benchmark no-filter case (should be O(1))
        start = time.time()
        for _ in range(1000):
            count = builder.count_unique_tokens()
        no_filter_time = time.time() - start
        
        # Benchmark station filter
        start = time.time()
        for _ in range(100):
            count = builder.count_unique_tokens(station_id="station_5")
        station_filter_time = time.time() - start
        
        print(f"\nNo filter (1000x): {no_filter_time:.4f}s")
        print(f"Station filter (100x): {station_filter_time:.4f}s")
        
        # No-filter should be nearly instant (O(1) cached)
        assert no_filter_time < 0.001, f"No filter took {no_filter_time}s, expected <1ms"


class TestFlowMatrixVectorization:
    """Test vectorized flow matrix computation."""
    
    def test_flow_matrix_correctness(self):
        """Verify flow matrix is computed correctly."""
        builder = TrajectoryBuilder()
        base_time = datetime.now()
        
        # Create known trajectories: A->B->C, A->B->A, B->C
        trajectories = [
            ["A", "B", "C"],
            ["A", "B", "A"],
            ["B", "C"],
        ]
        
        for i, path in enumerate(trajectories):
            token = f"token_{i}"
            for j, station in enumerate(path):
                event = DetectionEvent(
                    token=token,
                    timestamp=base_time + timedelta(minutes=j, seconds=i*100),
                    station_id=station,
                    camera_id="cam",
                    confidence=0.9
                )
                builder.add_detection(event)
        
        # Get flow matrix
        matrix = builder.get_flow_matrix(stations=["A", "B", "C"])
        
        # Expected flows:
        # A->B: 2 (from trajectories 0 and 1)
        # B->C: 2 (from trajectories 0 and 2)
        # B->A: 1 (from trajectory 1)
        
        assert matrix[0, 1] == 2, "A->B should be 2"
        assert matrix[1, 2] == 2, "B->C should be 2"
        assert matrix[1, 0] == 1, "B->A should be 1"
    
    def test_flow_matrix_performance(self):
        """Benchmark vectorized flow matrix computation."""
        builder = TrajectoryBuilder()
        events = generate_test_events(50000, n_stations=50, n_tokens=2000)
        
        for event in events:
            builder.add_detection(event)
        
        # Benchmark flow matrix computation
        start = time.time()
        for _ in range(10):
            matrix = builder.get_flow_matrix()
        flow_time = time.time() - start
        
        print(f"\nFlow matrix (10x, 50k events): {flow_time:.4f}s")
        assert flow_time < 5.0, f"Flow matrix took {flow_time:.2f}s"


class TestStationGraph:
    """Test graph-based station analysis."""
    
    def test_dfs_all_paths(self):
        """Test DFS finds all paths."""
        graph = StationGraph()
        
        # Create a simple graph: A->B->C, A->C (direct)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("A", "C")
        
        # Find all paths from A to C
        paths = list(graph.dfs_all_paths("A", "C"))
        
        # Should find 2 paths: A->B->C and A->C
        assert len(paths) == 2
        assert ["A", "C"] in paths
        assert ["A", "B", "C"] in paths
    
    def test_bfs_shortest_path(self):
        """Test BFS finds shortest path."""
        graph = StationGraph()
        
        # Create graph with different path lengths
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        graph.add_edge("A", "D")  # Direct path
        
        # Shortest from A to D should be direct
        path = graph.bfs_shortest_path("A", "D")
        assert path == ["A", "D"]
    
    def test_strongly_connected_components(self):
        """Test SCC detection."""
        graph = StationGraph()
        
        # Create two SCCs: {A, B} and {C, D}
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")
        graph.add_edge("C", "D")
        graph.add_edge("D", "C")
        graph.add_edge("B", "C")  # Connection between components
        
        sccs = graph.get_strongly_connected_stations()
        
        # Should have 2 components
        assert len(sccs) == 2
    
    def test_graph_build_from_trajectories(self):
        """Test building graph from trajectories."""
        builder = TrajectoryBuilder()
        base_time = datetime.now()
        
        # Create trajectories
        for i in range(100):
            token = f"token_{i}"
            path = ["A", "B", "C"] if i % 2 == 0 else ["A", "C", "B"]
            for j, station in enumerate(path):
                event = DetectionEvent(
                    token=token,
                    timestamp=base_time + timedelta(minutes=j, seconds=i),
                    station_id=station,
                    camera_id="cam",
                    confidence=0.9
                )
                builder.add_detection(event)
        
        # Build graph
        graph = StationGraph()
        graph.build_from_trajectories(builder.get_trajectories(min_length=2))
        
        # Check flow counts
        neighbors_a = graph.get_neighbors("A")
        assert "B" in neighbors_a
        assert "C" in neighbors_a
    
    def test_hub_detection(self):
        """Test hub station detection."""
        graph = StationGraph()
        
        # Create a star topology with A as hub
        for i in range(10):
            station = f"station_{i}"
            graph.add_edge("A", station, weight=10)
            graph.add_edge(station, "A", weight=10)
        
        # A should be detected as hub
        centrality = graph.get_station_centrality()
        
        # A should have highest centrality
        assert max(centrality, key=centrality.get) == "A"
    
    def test_reachable_stations(self):
        """Test reachable stations within max hops."""
        graph = StationGraph()
        
        # Create linear chain: A->B->C->D->E
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        graph.add_edge("D", "E")
        
        # From A with 2 hops: should reach A, B, C
        reachable = graph.get_reachable_stations("A", max_hops=2)
        assert reachable == {"A", "B", "C"}
        
        # From A with 4 hops: should reach all
        reachable = graph.get_reachable_stations("A", max_hops=4)
        assert reachable == {"A", "B", "C", "D", "E"}


class TestRoundTripCache:
    """Test round trip caching in DailyTrajectory."""
    
    def test_round_trip_cached(self):
        """Verify round trip status is cached."""
        traj = DailyTrajectory(token="test")
        base_time = datetime.now()
        
        # Add round trip: A -> B -> A
        for i, station in enumerate(["A", "B", "A"]):
            event = DetectionEvent(
                token="test",
                timestamp=base_time + timedelta(minutes=i),
                station_id=station,
                camera_id="cam",
                confidence=0.9
            )
            traj.add_event(event)
        
        # First access computes
        assert traj.is_round_trip is True
        
        # Second access uses cache
        assert traj._round_trip_cached is True
        assert traj.is_round_trip is True


class TestOverallPerformance:
    """Integration performance tests."""
    
    def test_large_scale_processing(self):
        """Test processing large number of events."""
        builder = TrajectoryBuilder()
        
        # Generate 100k events
        n_events = 100000
        events = generate_test_events(n_events, n_stations=100, n_tokens=5000)
        
        # Time event insertion
        start = time.time()
        for event in events:
            builder.add_detection(event)
        insertion_time = time.time() - start
        
        # Time various queries
        start = time.time()
        _ = builder.count_unique_tokens()
        count_time = time.time() - start
        
        start = time.time()
        _ = builder.get_flow_matrix()
        matrix_time = time.time() - start
        
        start = time.time()
        _ = builder.get_trajectories(min_length=2)
        traj_time = time.time() - start
        
        print(f"\n=== Large Scale Performance ({n_events} events) ===")
        print(f"Event insertion: {insertion_time:.2f}s ({n_events/insertion_time:.0f} events/sec)")
        print(f"Count unique tokens: {count_time:.4f}s")
        print(f"Flow matrix: {matrix_time:.4f}s")
        print(f"Get trajectories: {traj_time:.4f}s")
        
        # Performance assertions
        assert insertion_time < 30, "Insertion should complete in <30s"
        assert count_time < 0.01, "Token count should be <10ms"
        assert matrix_time < 2, "Flow matrix should compute in <2s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
