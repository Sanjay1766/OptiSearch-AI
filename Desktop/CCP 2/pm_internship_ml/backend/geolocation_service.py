"""
Geolocation Service for finding nearby internships
Uses geopy for distance calculations
"""
from geopy.distance import geodesic
from typing import List, Dict, Tuple
import pandas as pd


class GeolocationService:
    """Service to handle location-based internship searches"""
    
    NEARBY_RADIUS_KM = 100  # Search radius for nearby internships
    
    @staticmethod
    def get_coordinates(location_name: str) -> Tuple[float, float]:
        """
        Get latitude and longitude for a location
        Returns hardcoded Indian city coordinates
        """
        cities = {
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
            'pune': (18.5204, 73.8567),
        }
        
        location_lower = location_name.lower().strip()
        if location_lower in cities:
            return cities[location_lower]
        # Return None if location not found
        return None
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        coord1 = (lat1, lon1)
        coord2 = (lat2, lon2)
        return geodesic(coord1, coord2).kilometers
    
    @staticmethod
    def find_nearby_internships(internships_df: pd.DataFrame, 
                               location: str, 
                               radius_km: int = None) -> pd.DataFrame:
        """
        Find internships in nearby locations if not found in exact location
        Returns sorted by distance
        """
        if radius_km is None:
            radius_km = GeolocationService.NEARBY_RADIUS_KM
        
        user_coords = GeolocationService.get_coordinates(location)
        if user_coords is None:
            return pd.DataFrame()  # Return empty if location not found
        
        user_lat, user_lon = user_coords
        
        # Calculate distances
        internships_df['distance_km'] = internships_df.apply(
            lambda row: GeolocationService.calculate_distance(
                user_lat, user_lon, row['latitude'], row['longitude']
            ),
            axis=1
        )
        
        # Filter by radius
        nearby = internships_df[internships_df['distance_km'] <= radius_km]
        
        # Sort by distance
        return nearby.sort_values('distance_km').reset_index(drop=True)
    
    @staticmethod
    def get_nearby_cities(location: str, radius_km: int = 100) -> List[str]:
        """Get list of nearby cities within radius"""
        user_coords = GeolocationService.get_coordinates(location)
        if user_coords is None:
            return []
        
        cities = {
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
        
        user_lat, user_lon = user_coords
        nearby_cities = []
        
        for city, (lat, lon) in cities.items():
            distance = GeolocationService.calculate_distance(user_lat, user_lon, lat, lon)
            if distance <= radius_km and distance > 0:  # Exclude the searched city itself
                nearby_cities.append((city, distance))
        
        # Sort by distance
        nearby_cities.sort(key=lambda x: x[1])
        return [city[0] for city in nearby_cities]
