#!/usr/bin/env python3
"""
Example: Basic PrivacyFlow Usage

Demonstrates how to use the PrivacyFlow system for privacy-preserving
person detection and analytics.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import numpy as np

from src.privacyflow import PrivacyFlow
from src.privacy_token import PrivacyTokenGenerator
from src.analytics import Analytics, OccupancyMonitor, PlanningReport
from src.trajectory import TrajectoryBuilder
from src.database import EphemeralDatabase
from src.data_structures import DetectionEvent


def example_token_generation():
    """Example: Generate anonymous tokens from features."""
    print("=" * 60)
    print("Example 1: Token Generation")
    print("=" * 60)
    
    generator = PrivacyTokenGenerator()
    
    # Simulate feature extraction results
    face_embedding = np.random.rand(64).astype(np.float32)
    body_proportions = np.random.rand(8).astype(np.float32)
    appearance_features = np.random.rand(32).astype(np.float32)
    
    # Generate anonymous token
    token = generator.generate_token(
        face_embedding,
        body_proportions,
        appearance_features
    )
    
    print(f"Generated token: {token}")
    print(f"Token length: {len(token)} characters")
    print("Note: This token is irreversible - cannot recover original features")
    print()


def example_trajectory_building():
    """Example: Build and analyze trajectories."""
    print("=" * 60)
    print("Example 2: Trajectory Building")
    print("=" * 60)
    
    builder = TrajectoryBuilder()
    generator = PrivacyTokenGenerator()
    
    # Simulate a person traveling through stations
    person_features = (
        np.random.rand(64).astype(np.float32),  # face
        np.random.rand(8).astype(np.float32),   # body
        np.random.rand(32).astype(np.float32)   # appearance
    )
    
    token = generator.generate_token(*person_features)
    
    # Add detection events
    from datetime import timedelta
    base_time = datetime.now()
    
    events = [
        ("station_A", base_time),
        ("station_B", base_time + timedelta(minutes=15)),
        ("station_C", base_time + timedelta(minutes=30)),
        ("station_A", base_time + timedelta(hours=1))  # Round trip!
    ]
    
    for station_id, timestamp in events:
        event = DetectionEvent(
            token=token,
            timestamp=timestamp,
            station_id=station_id,
            camera_id=f"cam_{station_id}",
            confidence=0.95
        )
        builder.add_detection(event)
    
    # Analyze trajectories
    trajectories = builder.get_trajectories(min_length=2)
    print(f"Total trajectories: {len(trajectories)}")
    
    for traj in trajectories:
        print(f"  Token: {traj.token[:8]}...")
        print(f"  Stations visited: {' -> '.join(traj.stations_visited)}")
        print(f"  Is round trip: {traj.is_round_trip}")
    
    print(f"\nRound trip percentage: {builder.get_round_trip_percentage():.1f}%")
    print()


def example_occupancy_monitoring():
    """Example: Real-time occupancy monitoring."""
    print("=" * 60)
    print("Example 3: Occupancy Monitoring")
    print("=" * 60)
    
    monitor = OccupancyMonitor(
        station_id="platform_3",
        max_capacity=200,
        alert_threshold=0.8
    )
    
    @monitor.on_threshold_exceeded
    def handle_crowding(current_count, capacity):
        print(f"⚠️  ALERT: Platform 3 at {current_count/capacity*100:.0f}% capacity!")
    
    # Simulate people entering
    print("Simulating people entering platform...")
    for i in range(170):
        monitor.increment()
        if i == 150:
            print(f"Current count: {monitor.current_count}")
        if i == 160:
            print(f"Current count: {monitor.current_count}")
    
    print(f"Final count: {monitor.current_count}")
    print(f"Occupancy: {monitor.occupancy_ratio*100:.1f}%")
    print()


def example_aggregate_analytics():
    """Example: Aggregate analytics (privacy-preserving)."""
    print("=" * 60)
    print("Example 4: Aggregate Analytics")
    print("=" * 60)
    
    # Peak hour analysis
    peak_data = Analytics.find_peak_hours(
        station_id="central_station",
        date_range=("2024-01-01", "2024-01-31")
    )
    
    print("Peak Hours Analysis:")
    print(f"  Station: {peak_data['station_id']}")
    print(f"  Peak hour: {peak_data['peak_hour']}:00")
    print(f"  Average passengers: {peak_data['avg_count']}")
    
    # Note: This returns ONLY aggregate data, no individual tracking
    print("\nNote: All analytics contain ONLY aggregate data.")
    print("No individual tracking information is ever exposed.")
    print()


def example_planning_report():
    """Example: Generate planning reports."""
    print("=" * 60)
    print("Example 5: Planning Report Generation")
    print("=" * 60)
    
    report = PlanningReport.generate(
        stations=["A", "B", "C", "D"],
        date_range=("2024-01-01", "2024-03-31"),
        metrics=[
            "daily_unique_visitors",
            "hourly_distribution",
            "route_popularity",
            "round_trip_percentage",
            "average_journey_time"
        ]
    )
    
    print("Report generated with metrics:")
    for metric in report.data['metrics']:
        print(f"  - {metric}")
    
    # Export to file
    output_file = "/tmp/quarterly_report.json"
    report.export(output_file)
    print(f"\nReport exported to: {output_file}")
    print()


def example_daily_purge():
    """Example: Daily data purge (privacy mechanism)."""
    print("=" * 60)
    print("Example 6: Daily Purge Mechanism")
    print("=" * 60)
    
    db = EphemeralDatabase()
    
    # Insert some detections
    for i in range(10):
        event = DetectionEvent(
            token=f"token_{i:04d}",
            timestamp=datetime.now(),
            station_id="station_001",
            camera_id="camera_001",
            confidence=0.95
        )
        db.insert_detection(event)
    
    count_before = db.count_unique_tokens()
    print(f"Detections before purge: {count_before}")
    
    # Simulate midnight purge
    deleted = db.purge_daily_data()
    print(f"Records deleted: {deleted}")
    
    count_after = db.count_unique_tokens()
    print(f"Detections after purge: {count_after}")
    
    print("\n✓ This is the PRIMARY privacy mechanism.")
    print("✓ All raw detection data is deleted at midnight.")
    print("✓ Only aggregate statistics persist.")
    
    db.close()
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PrivacyFlow - Privacy-Preserving Person Detection")
    print("Example Demonstrations")
    print("=" * 60 + "\n")
    
    example_token_generation()
    example_trajectory_building()
    example_occupancy_monitoring()
    example_aggregate_analytics()
    example_planning_report()
    example_daily_purge()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
