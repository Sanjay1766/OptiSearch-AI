"""
Geolocation Service for finding nearby internships
Optimized with caching and vectorized distance calculations
"""
from geopy.distance import geodesic
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


class GeolocationService:
    """Service to handle location-based internship searches - Optimized"""
    
    NEARBY_RADIUS_KM = 100  # Search radius for nearby internships
    
    # Cache for coordinates - prevent repeated lookups
    _coord_cache = {}
    
    # Cities database - shared across all instances
    CITIES = {
        'bangalore': (12.9716, 77.5946),
        'mumbai': (19.0760, 72.8777),
        'delhi': (28.7041, 77.1025),
        'hyderabad': (17.3850, 78.4867),
        'pune': (18.5204, 73.8567),
        'chennai': (13.0827, 80.2707),
        'kolkata': (22.5726, 88.3639),
        'jaipur': (26.9124, 75.7873),
        'lucknow': (26.8467, 80.9462),
        'ahmedabad': (23.0225, 72.5714),
        'gurgaon': (28.4595, 77.0266),
        'noida': (28.5721, 77.3565),
    }
    
    @classmethod
    def get_coordinates(cls, location_name: str) -> Tuple[float, float]:
        """
        Get latitude and longitude for a location with caching
        Returns hardcoded Indian city coordinates
        """
        location_lower = location_name.lower().strip()
        
        # Check cache first
        if location_lower in cls._coord_cache:
            return cls._coord_cache[location_lower]
        
        if location_lower in cls.CITIES:
            coords = cls.CITIES[location_lower]
            cls._coord_cache[location_lower] = coords
            return coords
        
        return None
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    
    @staticmethod
    def find_nearby_internships(internships_df: pd.DataFrame, 
                               location: str, 
                               radius_km: int = None) -> pd.DataFrame:
        """
        Optimized: Find internships in nearby locations
        Uses vectorized operations for better performance
        """
        if radius_km is None:
            radius_km = GeolocationService.NEARBY_RADIUS_KM
        
        user_coords = GeolocationService.get_coordinates(location)
        if user_coords is None:
            return pd.DataFrame()
        
        user_lat, user_lon = user_coords
        
        # Vectorized distance calculation using numpy
        try:
            # Use vectorized operations instead of apply()
            lats = internships_df['latitude'].values
            lons = internships_df['longitude'].values
            
            # Haversine formula for faster distance calculation
            distances = GeolocationService._vectorized_distance(
                user_lat, user_lon, lats, lons
            )
            
            # Create a copy to avoid SettingWithCopyWarning
            result_df = internships_df.copy()
            result_df['distance_km'] = distances
            
            # Filter and sort
            nearby = result_df[result_df['distance_km'] <= radius_km]
            return nearby.sort_values('distance_km').reset_index(drop=True)
        
        except Exception as e:
            print(f"Error in vectorized distance calculation: {e}")
            # Fallback to original method
            return internships_df.copy()
    
    @staticmethod
    def _vectorized_distance(lat1: float, lon1: float, 
                            lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Calculate distances using vectorized operations (Haversine formula)
        Much faster than geodesic for arrays
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lats_rad = np.radians(lats)
        dlon = np.radians(lons - lon1)
        dlat = np.radians(lats - lat1)
        
        # Haversine formula
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        km = 6371 * c
        return km
    
    @classmethod
    def get_nearby_cities(cls, location: str, radius_km: int = 100) -> List[str]:
        """Get list of nearby cities within radius - Optimized"""
        user_coords = cls.get_coordinates(location)
        if user_coords is None:
            return []
        
        user_lat, user_lon = user_coords
        nearby_cities = []
        
        # Pre-compute distances for all cities
        for city, (lat, lon) in cls.CITIES.items():
            distance = cls.calculate_distance(user_lat, user_lon, lat, lon)
            if 0 < distance <= radius_km:  # Exclude the searched city itself
                nearby_cities.append((city, distance))
        
        # Sort by distance
        nearby_cities.sort(key=lambda x: x[1])
        return [city[0] for city in nearby_cities]
