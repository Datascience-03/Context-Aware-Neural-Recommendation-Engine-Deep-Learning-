import json
import os
from typing import Any, Dict, Optional

import redis


class RedisUserProfileStore:
    """Redis-backed feature store for user profiles."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        key_prefix: str = "user:",
        ttl_seconds: Optional[int] = None,
    ):
        # Get the Redis connection URL from the environment.
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0",
        )

        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

        # Create the Redis client.
        self.client = redis.Redis.from_url(
            self.redis_url,
            decode_responses=True,
        )

    def _make_key(self, customer_id: Any) -> str:
        """Create a Redis key for a customer."""
        return f"{self.key_prefix}{customer_id}"

    def ping(self) -> bool:
        """Check whether Redis is reachable."""
        return bool(self.client.ping())

    def store_user_profile(
        self,
        customer_id: Any,
        profile: Dict[str, Any],
    ) -> bool:
        """Store a user profile in Redis."""
        if not profile:
            raise ValueError("User profile cannot be empty.")

        key = self._make_key(customer_id)

        profile_data = {
            str(field): json.dumps(value)
            for field, value in profile.items()
        }

        self.client.hset(key, mapping=profile_data)

        if self.ttl_seconds is not None:
            self.client.expire(key, self.ttl_seconds)

        return True

    def get_user_profile(
        self,
        customer_id: Any,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a user profile from Redis."""
        key = self._make_key(customer_id)

        data = self.client.hgetall(key)

        if not data:
            return None

        return {
            field: json.loads(value)
            for field, value in data.items()
        }

    def update_user_profile(
        self,
        customer_id: Any,
        updates: Dict[str, Any],
    ) -> bool:
        """Update selected fields in a user profile."""
        if not updates:
            raise ValueError("Updates cannot be empty.")

        key = self._make_key(customer_id)

        update_data = {
            str(field): json.dumps(value)
            for field, value in updates.items()
        }

        self.client.hset(key, mapping=update_data)

        if self.ttl_seconds is not None:
            self.client.expire(key, self.ttl_seconds)

        return True

    def delete_user_profile(self, customer_id: Any) -> bool:
        """Delete a user profile from Redis."""
        key = self._make_key(customer_id)
        return bool(self.client.delete(key))

    def user_exists(self, customer_id: Any) -> bool:
        """Check whether a user profile exists."""
        return bool(self.client.exists(self._make_key(customer_id)))

    def get_query_features(
        self,
        customer_id: Any,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve features required by the QueryTower."""
        profile = self.get_user_profile(customer_id)

        if profile is None:
            return None

        required_features = [
            "customer_id_idx",
            "month_sin",
            "month_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "is_weekend",
            "days_since_last_purchase",
            "purchase_sequence",
        ]

        return {
            feature: profile[feature]
            for feature in required_features
            if feature in profile
        }

    def close(self) -> None:
        """Close the Redis connection."""
        self.client.close()