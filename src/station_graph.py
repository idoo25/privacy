"""
Station Graph Module

Graph-based station network for efficient route analysis.

Features:
- Adjacency list representation for O(1) neighbor lookup
- DFS for path finding and route discovery
- BFS for shortest path analysis
- Vectorized graph operations using numpy
"""

from collections import defaultdict, deque
from typing import List, Dict, Set, Optional, Tuple, Generator
import numpy as np

from .data_structures import DetectionEvent, DailyTrajectory


class StationGraph:
    """
    Graph representation of station network for efficient route analysis.
    
    Uses adjacency list for O(1) neighbor access and supports:
    - DFS for finding all paths between stations
    - BFS for shortest path analysis
    - Flow-weighted edge analysis
    """
    
    def __init__(self):
        """Initialize empty station graph."""
        # Adjacency list: station -> {neighbor: weight}
        self._adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._stations: Set[str] = set()
        self._total_edges: int = 0
        
        # Cached analysis results
        self._betweenness_cache: Optional[Dict[str, float]] = None
        self._degree_cache: Optional[Dict[str, Tuple[int, int]]] = None
    
    def add_edge(self, from_station: str, to_station: str, weight: int = 1) -> None:
        """
        Add or update edge between stations.
        
        Args:
            from_station: Source station
            to_station: Destination station
            weight: Edge weight (flow count)
        """
        self._stations.add(from_station)
        self._stations.add(to_station)
        self._adjacency[from_station][to_station] += weight
        self._total_edges += weight
        
        # Invalidate caches
        self._betweenness_cache = None
        self._degree_cache = None
    
    def build_from_trajectories(self, trajectories: List[DailyTrajectory]) -> None:
        """
        Build graph from trajectory data.
        
        Args:
            trajectories: List of DailyTrajectory objects
        """
        for traj in trajectories:
            stations = traj.stations_visited
            for i in range(len(stations) - 1):
                self.add_edge(stations[i], stations[i + 1])
    
    def get_neighbors(self, station: str) -> Dict[str, int]:
        """
        Get neighbors of a station with edge weights - O(1).
        
        Args:
            station: Station ID
        
        Returns:
            Dict mapping neighbor stations to flow counts
        """
        return dict(self._adjacency.get(station, {}))
    
    def dfs_all_paths(self, start: str, end: str, 
                      max_depth: int = 10) -> Generator[List[str], None, None]:
        """
        Find all paths between stations using DFS.
        
        Args:
            start: Start station
            end: End station
            max_depth: Maximum path length
        
        Yields:
            Lists of stations representing paths
        """
        if start not in self._stations or end not in self._stations:
            return
        
        def dfs_helper(current: str, target: str, path: List[str], 
                      visited: Set[str], depth: int):
            if depth > max_depth:
                return
            
            if current == target:
                yield path.copy()
                return
            
            for neighbor in self._adjacency.get(current, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    yield from dfs_helper(neighbor, target, path, visited, depth + 1)
                    path.pop()
                    visited.remove(neighbor)
        
        visited = {start}
        yield from dfs_helper(start, end, [start], visited, 0)
    
    def bfs_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        Find shortest path between stations using BFS.
        
        Args:
            start: Start station
            end: End station
        
        Returns:
            List of stations in path, or None if no path exists
        """
        if start not in self._stations or end not in self._stations:
            return None
        
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self._adjacency.get(current, {}):
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_strongly_connected_stations(self) -> List[Set[str]]:
        """
        Find strongly connected components using Kosaraju's algorithm.
        
        Returns:
            List of sets, each containing stations in a connected component
        """
        # First DFS to get finish order
        visited = set()
        finish_order = []
        
        def dfs_forward(station: str):
            visited.add(station)
            for neighbor in self._adjacency.get(station, {}):
                if neighbor not in visited:
                    dfs_forward(neighbor)
            finish_order.append(station)
        
        for station in self._stations:
            if station not in visited:
                dfs_forward(station)
        
        # Build reverse graph
        reverse_adj = defaultdict(list)
        for station, neighbors in self._adjacency.items():
            for neighbor in neighbors:
                reverse_adj[neighbor].append(station)
        
        # Second DFS in reverse order
        visited.clear()
        components = []
        
        def dfs_reverse(station: str, component: Set[str]):
            visited.add(station)
            component.add(station)
            for neighbor in reverse_adj.get(station, []):
                if neighbor not in visited:
                    dfs_reverse(neighbor, component)
        
        for station in reversed(finish_order):
            if station not in visited:
                component = set()
                dfs_reverse(station, component)
                components.append(component)
        
        return components
    
    def get_station_centrality(self) -> Dict[str, float]:
        """
        Calculate degree centrality for all stations.
        
        Returns:
            Dict mapping stations to centrality scores
        """
        if not self._stations:
            return {}
        
        n = len(self._stations)
        if n <= 1:
            return {s: 0.0 for s in self._stations}
        
        centrality = {}
        for station in self._stations:
            out_degree = sum(self._adjacency.get(station, {}).values())
            in_degree = sum(
                neighbors.get(station, 0) 
                for neighbors in self._adjacency.values()
            )
            # Normalized degree centrality
            centrality[station] = (out_degree + in_degree) / (2 * (n - 1) * self._total_edges) \
                if self._total_edges > 0 else 0.0
        
        return centrality
    
    def get_flow_matrix_numpy(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get flow matrix as numpy array with vectorized computation.
        
        Returns:
            Tuple of (flow_matrix, station_list)
        """
        stations = sorted(self._stations)
        n = len(stations)
        
        if n == 0:
            return np.zeros((0, 0), dtype=int), []
        
        station_idx = {s: i for i, s in enumerate(stations)}
        
        # Vectorized matrix construction
        rows = []
        cols = []
        data = []
        
        for from_station, neighbors in self._adjacency.items():
            from_idx = station_idx[from_station]
            for to_station, weight in neighbors.items():
                rows.append(from_idx)
                cols.append(station_idx[to_station])
                data.append(weight)
        
        # Create dense matrix using vectorized operations
        matrix = np.zeros((n, n), dtype=int)
        if rows:
            np.add.at(matrix, (rows, cols), data)
        
        return matrix, stations
    
    def get_top_routes(self, k: int = 10) -> List[Tuple[str, str, int]]:
        """
        Get top k routes by flow count.
        
        Args:
            k: Number of top routes to return
        
        Returns:
            List of (from_station, to_station, count) tuples
        """
        routes = []
        for from_station, neighbors in self._adjacency.items():
            for to_station, count in neighbors.items():
                routes.append((from_station, to_station, count))
        
        # Sort by count descending and take top k
        routes.sort(key=lambda x: x[2], reverse=True)
        return routes[:k]
    
    def find_hub_stations(self, threshold: float = 0.1) -> List[str]:
        """
        Find hub stations (high connectivity).
        
        Args:
            threshold: Minimum centrality threshold
        
        Returns:
            List of hub station IDs
        """
        centrality = self.get_station_centrality()
        return [s for s, c in centrality.items() if c >= threshold]
    
    def get_reachable_stations(self, start: str, max_hops: int = 3) -> Set[str]:
        """
        Get all stations reachable within max_hops using BFS.
        
        Args:
            start: Starting station
            max_hops: Maximum number of hops
        
        Returns:
            Set of reachable station IDs
        """
        if start not in self._stations:
            return set()
        
        reachable = {start}
        current_level = {start}
        
        for _ in range(max_hops):
            next_level = set()
            for station in current_level:
                for neighbor in self._adjacency.get(station, {}):
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
            if not current_level:
                break
        
        return reachable
    
    def clear(self) -> None:
        """Clear all graph data."""
        self._adjacency.clear()
        self._stations.clear()
        self._total_edges = 0
        self._betweenness_cache = None
        self._degree_cache = None
    
    @property
    def station_count(self) -> int:
        """Get number of stations."""
        return len(self._stations)
    
    @property
    def edge_count(self) -> int:
        """Get number of unique edges."""
        return sum(len(neighbors) for neighbors in self._adjacency.values())
    
    @property
    def total_flow(self) -> int:
        """Get total flow across all edges."""
        return self._total_edges
