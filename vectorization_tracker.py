"""
SQLite-based vectorization tracking system.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VectorizationTracker:
    
    def __init__(self, db_path: str = "./vectorization_tracker.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS vectorized_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_key TEXT UNIQUE NOT NULL,
                        event_name TEXT NOT NULL,
                        location TEXT NOT NULL,
                        year TEXT NOT NULL,
                        month TEXT NOT NULL,
                        vectorized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        bounty_count INTEGER NOT NULL,
                        vector_count INTEGER NOT NULL,
                        status TEXT DEFAULT 'active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS vectorized_bounties (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_key TEXT NOT NULL,
                        bounty_id TEXT UNIQUE NOT NULL,
                        company TEXT NOT NULL,
                        title TEXT NOT NULL,
                        vectorized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'active',
                        FOREIGN KEY (event_key) REFERENCES vectorized_events(event_key)
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_key ON vectorized_events(event_key)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bounty_id ON vectorized_bounties(bounty_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_bounties ON vectorized_bounties(event_key)
                """)
                
                conn.commit()
                logger.info("✅ Vectorization tracker database initialized")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {str(e)}")
            raise
    
    def is_event_vectorized(self, event_key: str) -> bool:
        """Check if an event has been vectorized."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM vectorized_events 
                    WHERE event_key = ? AND status = 'active'
                """, (event_key,))
                
                count = cursor.fetchone()[0]
                return count > 0
                
        except Exception as e:
            logger.error(f"❌ Error checking event vectorization status: {str(e)}")
            return False
    
    def get_event_info(self, event_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a vectorized event."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT event_key, event_name, location, year, month, 
                           vectorized_at, bounty_count, vector_count, status
                    FROM vectorized_events 
                    WHERE event_key = ? AND status = 'active'
                """, (event_key,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'event_key': result[0],
                        'event_name': result[1],
                        'location': result[2],
                        'year': result[3],
                        'month': result[4],
                        'vectorized_at': result[5],
                        'bounty_count': result[6],
                        'vector_count': result[7],
                        'status': result[8]
                    }
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting event info: {str(e)}")
            return None
    
    def mark_event_vectorized(self, event_key: str, event_name: str, location: str, 
                            year: str, month: str, bounty_count: int, vector_count: int) -> bool:
        """Mark an event as vectorized."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or update event record
                cursor.execute("""
                    INSERT OR REPLACE INTO vectorized_events 
                    (event_key, event_name, location, year, month, bounty_count, vector_count, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'active', CURRENT_TIMESTAMP)
                """, (event_key, event_name, location, year, month, bounty_count, vector_count))
                
                conn.commit()
                logger.info(f"✅ Marked event {event_key} as vectorized with {bounty_count} bounties, {vector_count} vectors")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error marking event as vectorized: {str(e)}")
            return False
    
    def mark_bounties_vectorized(self, event_key: str, bounties: List[Dict[str, Any]]) -> bool:
        """Mark individual bounties as vectorized."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert bounty records
                bounty_data = []
                for bounty in bounties:
                    bounty_data.append((
                        event_key,
                        bounty['bounty_id'],
                        bounty['company'],
                        bounty['title']
                    ))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO vectorized_bounties 
                    (event_key, bounty_id, company, title, status)
                    VALUES (?, ?, ?, ?, 'active')
                """, bounty_data)
                
                conn.commit()
                logger.info(f"✅ Marked {len(bounties)} bounties as vectorized for event {event_key}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error marking bounties as vectorized: {str(e)}")
            return False
    
    def get_vectorized_events(self) -> List[Dict[str, Any]]:
        """Get all vectorized events."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT event_key, event_name, location, year, month, 
                           vectorized_at, bounty_count, vector_count, status
                    FROM vectorized_events 
                    WHERE status = 'active'
                    ORDER BY vectorized_at DESC
                """)
                
                results = cursor.fetchall()
                events = []
                for result in results:
                    events.append({
                        'event_key': result[0],
                        'event_name': result[1],
                        'location': result[2],
                        'year': result[3],
                        'month': result[4],
                        'vectorized_at': result[5],
                        'bounty_count': result[6],
                        'vector_count': result[7],
                        'status': result[8]
                    })
                
                return events
                
        except Exception as e:
            logger.error(f"❌ Error getting vectorized events: {str(e)}")
            return []
    
    def get_event_bounty_count(self, event_key: str) -> int:
        """Get the number of bounties vectorized for an event."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM vectorized_bounties 
                    WHERE event_key = ? AND status = 'active'
                """, (event_key,))
                
                count = cursor.fetchone()[0]
                return count
                
        except Exception as e:
            logger.error(f"❌ Error getting bounty count: {str(e)}")
            return 0
    
    def clear_event(self, event_key: str) -> bool:
        """Clear vectorization records for an event (mark as inactive)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Mark event as inactive
                cursor.execute("""
                    UPDATE vectorized_events 
                    SET status = 'inactive'
                    WHERE event_key = ?
                """, (event_key,))
                
                # Mark bounties as inactive
                cursor.execute("""
                    UPDATE vectorized_bounties 
                    SET status = 'inactive'
                    WHERE event_key = ?
                """, (event_key,))
                
                conn.commit()
                logger.info(f"✅ Cleared vectorization records for event {event_key}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error clearing event: {str(e)}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all vectorization records."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Mark all events as inactive
                cursor.execute("""
                    UPDATE vectorized_events 
                    SET status = 'inactive'
                """)
                
                # Mark all bounties as inactive
                cursor.execute("""
                    UPDATE vectorized_bounties 
                    SET status = 'inactive'
                """)
                
                conn.commit()
                logger.info("✅ Cleared all vectorization records")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error clearing all records: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total events
                cursor.execute("SELECT COUNT(*) FROM vectorized_events WHERE status = 'active'")
                total_events = cursor.fetchone()[0]
                
                # Get total bounties
                cursor.execute("SELECT COUNT(*) FROM vectorized_bounties WHERE status = 'active'")
                total_bounties = cursor.fetchone()[0]
                
                # Get total vectors (sum of vector_count from events)
                cursor.execute("SELECT SUM(vector_count) FROM vectorized_events WHERE status = 'active'")
                result = cursor.fetchone()[0]
                total_vectors = result if result else 0
                
                return {
                    'total_events': total_events,
                    'total_bounties': total_bounties,
                    'total_vectors': total_vectors,
                    'database_path': self.db_path
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting stats: {str(e)}")
            return {'total_events': 0, 'total_bounties': 0, 'total_vectors': 0, 'database_path': self.db_path}
    
    def get_event_summary(self) -> str:
        """Get a summary of vectorized events for display."""
        events = self.get_vectorized_events()
        if not events:
            return "No events vectorized"
        
        summary = []
        for event in events:
            summary.append(
                f"✅ {event['event_name']}: {event['bounty_count']} bounties, {event['vector_count']} vectors"
            )
        
        return "\n".join(summary)
