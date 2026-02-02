"""
Scheduler Module

Handles scheduled tasks including the critical daily purge.
"""

import logging
import threading
from datetime import datetime, time
from typing import Any, Callable, Optional

try:
    import schedule
except ImportError:
    schedule = None


# Configure logging
logging.basicConfig(level=logging.INFO)
audit_log = logging.getLogger('privacyflow.audit')


class DailyPurgeScheduler:
    """
    Scheduler for the daily data purge.
    
    This is a CRITICAL privacy component.
    If the purge fails, the system MUST halt.
    """
    
    def __init__(self, purge_callback: Callable[[], int],
                 halt_callback: Optional[Callable[[], None]] = None,
                 purge_time: str = "00:00"):
        """
        Initialize purge scheduler.
        
        Args:
            purge_callback: Function to call for purging data.
                           Should return count of deleted records.
            halt_callback: Function to call if purge fails.
            purge_time: Time to run purge (HH:MM format)
        """
        self.purge_callback = purge_callback
        self.halt_callback = halt_callback
        self.purge_time = purge_time
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def purge_daily_data(self) -> None:
        """
        Execute the daily purge.
        
        This is the PRIMARY privacy mechanism.
        If this fails, the system MUST halt.
        """
        try:
            # Execute purge
            deleted_count = self.purge_callback()
            
            # Log for audit
            audit_log.info("Daily purge completed", extra={
                "records_deleted": deleted_count,
                "new_salt_generated": True,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            # CRITICAL: If purge fails, halt system
            audit_log.critical("PURGE FAILED - HALTING SYSTEM", extra={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            if self.halt_callback:
                self.halt_callback()
            else:
                self.emergency_halt()
    
    def emergency_halt(self) -> None:
        """Emergency halt when purge fails."""
        audit_log.critical("EMERGENCY HALT INITIATED")
        self._running = False
        # In production, this would trigger alerts and stop all processing
        raise SystemExit("CRITICAL: Daily purge failed - system halted for privacy protection")
    
    def start(self) -> None:
        """Start the scheduler."""
        if schedule is None:
            audit_log.warning("Schedule library not available, manual purge required")
            return
        
        # Schedule for configured time (default midnight)
        schedule.every().day.at(self.purge_time).do(self.purge_daily_data)
        
        self._running = True
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        
        audit_log.info(f"Daily purge scheduled for {self.purge_time}")
    
    def _run_scheduler(self) -> None:
        """Run the scheduler loop."""
        import time as time_module
        
        while self._running:
            schedule.run_pending()
            time_module.sleep(60)  # Check every minute
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def run_purge_now(self) -> None:
        """Manually trigger purge (for testing or maintenance)."""
        audit_log.info("Manual purge triggered")
        self.purge_daily_data()


class AuditLogger:
    """
    Audit logger for compliance tracking.
    
    All system actions are logged for compliance audits.
    Logs contain NO personal data.
    """
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize audit logger.
        
        Args:
            log_path: Path to audit log file (optional)
        """
        self.logger = logging.getLogger('privacyflow.audit')
        
        if log_path:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s - %(extra)s'
            ))
            self.logger.addHandler(handler)
    
    def log_detection(self, count: int, station: str) -> None:
        """Log detection event (no personal data)."""
        self.logger.info("Detection event", extra={
            "event_type": "detection",
            "count": count,
            "station": station
        })
    
    def log_purge(self, records_deleted: int) -> None:
        """Log purge event."""
        self.logger.info("Purge event", extra={
            "event_type": "purge",
            "records_deleted": records_deleted
        })
    
    def log_export(self, export_type: str, destination: str) -> None:
        """Log export event."""
        self.logger.info("Export event", extra={
            "event_type": "export",
            "type": export_type,
            "destination": destination
        })
    
    def log_config_change(self, parameter: str, old_value: Any, new_value: Any) -> None:
        """Log configuration change."""
        self.logger.info("Config change", extra={
            "event_type": "config_change",
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value
        })
