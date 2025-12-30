"""
Price alerts - notify users when event probability reaches certain threshold
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

ALERTS_FILE = "price_alerts.json"

class PriceAlert:
    def __init__(self, event_slug: str, condition: str, threshold: float, outcome_index: int = 0):
        self.event_slug = event_slug
        self.condition = condition  # ">" or "<"
        self.threshold = threshold
        self.outcome_index = outcome_index
        self.triggered = False

    def to_dict(self):
        return {
            "event_slug": self.event_slug,
            "condition": self.condition,
            "threshold": self.threshold,
            "outcome_index": self.outcome_index,
            "triggered": self.triggered
        }

    @staticmethod
    def from_dict(data):
        alert = PriceAlert(
            data["event_slug"],
            data["condition"],
            data["threshold"],
            data.get("outcome_index", 0)
        )
        alert.triggered = data.get("triggered", False)
        return alert

class Alerts:
    def __init__(self):
        self.user_alerts: Dict[int, List[PriceAlert]] = {}
        self.load()

    def load(self):
        """Load alerts from file"""
        if Path(ALERTS_FILE).exists():
            try:
                with open(ALERTS_FILE, 'r') as f:
                    data = json.load(f)
                    for user_id, alerts in data.items():
                        self.user_alerts[int(user_id)] = [
                            PriceAlert.from_dict(a) for a in alerts
                        ]
            except Exception as e:
                print(f"Error loading alerts: {e}")

    def save(self):
        """Save alerts to file"""
        try:
            data = {}
            for user_id, alerts in self.user_alerts.items():
                data[user_id] = [a.to_dict() for a in alerts]

            with open(ALERTS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving alerts: {e}")

    def add_alert(self, user_id: int, event_slug: str, condition: str, threshold: float, outcome_index: int = 0) -> bool:
        """Add a price alert for user"""
        if user_id not in self.user_alerts:
            self.user_alerts[user_id] = []

        # Check if alert already exists
        for alert in self.user_alerts[user_id]:
            if (alert.event_slug == event_slug and
                alert.condition == condition and
                alert.threshold == threshold and
                alert.outcome_index == outcome_index):
                return False  # Alert already exists

        alert = PriceAlert(event_slug, condition, threshold, outcome_index)
        self.user_alerts[user_id].append(alert)
        self.save()
        return True

    def remove_alert(self, user_id: int, index: int) -> bool:
        """Remove alert by index"""
        if user_id not in self.user_alerts:
            return False

        if index < 0 or index >= len(self.user_alerts[user_id]):
            return False

        del self.user_alerts[user_id][index]
        self.save()
        return True

    def get_alerts(self, user_id: int) -> List[PriceAlert]:
        """Get user's alerts"""
        return self.user_alerts.get(user_id, [])

    def clear_alerts(self, user_id: int):
        """Clear all user's alerts"""
        if user_id in self.user_alerts:
            del self.user_alerts[user_id]
            self.save()

    def check_alert(self, alert: PriceAlert, current_price: float) -> bool:
        """Check if alert should trigger"""
        if alert.triggered:
            return False

        if alert.condition == ">" and current_price > alert.threshold:
            return True
        elif alert.condition == "<" and current_price < alert.threshold:
            return True

        return False

    def mark_triggered(self, user_id: int, index: int):
        """Mark alert as triggered"""
        if user_id in self.user_alerts and index < len(self.user_alerts[user_id]):
            self.user_alerts[user_id][index].triggered = True
            self.save()
