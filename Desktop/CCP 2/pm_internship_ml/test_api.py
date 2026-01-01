"""
Test script for PM Internship ML Search API
"""
import requests
import json
from typing import Dict, List

BASE_URL = "http://localhost:5000/api"


class InternshipSearchTester:
    """Test different search functionalities"""
    
    @staticmethod
    def test_semantic_search(query: str, location: str = None):
        """Test semantic search"""
        print(f"\n{'='*60}")
        print(f"TEST: Semantic Search")
        print(f"Query: {query}")
        if location:
            print(f"Location: {location}")
        print('='*60)
        
        payload = {
            "query": query,
            "search_type": "semantic",
            "top_k": 5
        }
        if location:
            payload["location"] = location
        
        response = requests.post(f"{BASE_URL}/search", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nFound {data['total_results']} results:\n")
            for i, result in enumerate(data['results'], 1):
                print(f"{i}. {result['title']} at {result['company']} ({result['location']})")
                print(f"   Relevance: {result['relevance_score']:.2%}")
                print(f"   Skills: {result['skills_required'][:50]}...")
                print()
        else:
            print(f"Error: {response.text}")
    
    @staticmethod
    def test_location_search(location: str):
        """Test location-based search with nearby fallback"""
        print(f"\n{'='*60}")
        print(f"TEST: Location-Based Search")
        print(f"Location: {location}")
        print('='*60)
        
        payload = {"location": location, "top_k": 5}
        response = requests.post(f"{BASE_URL}/search-by-location", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nFound {data['total_results']} results:\n")
            for i, result in enumerate(data['results'], 1):
                print(f"{i}. {result['title']} at {result['company']}")
                print(f"   Location: {result['location']}")
                print()
        else:
            print(f"Error: {response.text}")
    
    @staticmethod
    def test_skill_search(skills: List[str]):
        """Test skill-based search"""
        print(f"\n{'='*60}")
        print(f"TEST: Skill-Based Search")
        print(f"Skills: {', '.join(skills)}")
        print('='*60)
        
        payload = {"skills": skills, "top_k": 5}
        response = requests.post(f"{BASE_URL}/search-by-skills", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nFound {data['total_results']} results:\n")
            for i, result in enumerate(data['results'], 1):
                print(f"{i}. {result['title']} at {result['company']}")
                print(f"   Relevance: {result['relevance_score']:.2%}")
                print()
        else:
            print(f"Error: {response.text}")
    
    @staticmethod
    def test_category_search(category: str, query: str = None):
        """Test category-based search"""
        print(f"\n{'='*60}")
        print(f"TEST: Category Search")
        print(f"Category: {category}")
        if query:
            print(f"Query: {query}")
        print('='*60)
        
        payload = {"category": category, "top_k": 5}
        if query:
            payload["query"] = query
        
        response = requests.post(f"{BASE_URL}/search-by-category", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nFound {data['total_results']} results:\n")
            for i, result in enumerate(data['results'], 1):
                print(f"{i}. {result['title']} at {result['company']}")
                if 'relevance_score' in result:
                    print(f"   Relevance: {result['relevance_score']:.2%}")
                print()
        else:
            print(f"Error: {response.text}")
    
    @staticmethod
    def test_get_metadata():
        """Test getting available locations and categories"""
        print(f"\n{'='*60}")
        print("TEST: Get Metadata")
        print('='*60)
        
        # Get locations
        response = requests.get(f"{BASE_URL}/locations")
        if response.status_code == 200:
            locations = response.json()['locations']
            print(f"\nAvailable Locations ({len(locations)}):")
            print(", ".join(locations))
        
        # Get categories
        response = requests.get(f"{BASE_URL}/categories")
        if response.status_code == 200:
            categories = response.json()['categories']
            print(f"\nAvailable Categories ({len(categories)}):")
            print(", ".join(categories))
    
    @staticmethod
    def test_health_check():
        """Test API health"""
        print(f"\n{'='*60}")
        print("TEST: Health Check")
        print('='*60)
        
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"\nStatus: {data['status']}")
            print(f"Total Internships: {data['total_internships']}")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print("PM INTERNSHIP ML SEARCH - API TESTS")
    print("="*60)
    
    # Test health
    tester = InternshipSearchTester()
    tester.test_health_check()
    tester.test_get_metadata()
    
    # Test semantic search
    tester.test_semantic_search("Python Django web development")
    tester.test_semantic_search("machine learning data analysis", "Bangalore")
    tester.test_semantic_search("cloud computing AWS")
    
    # Test location search
    tester.test_location_search("Mumbai")
    tester.test_location_search("Delhi")
    
    # Test skill search
    tester.test_skill_search(["Python", "Machine Learning"])
    tester.test_skill_search(["Java", "Spring", "SQL"])
    
    # Test category search
    tester.test_category_search("Technology")
    tester.test_category_search("Data Science", "machine learning")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\nMake sure the Flask server is running (python backend/app.py)")
    print("Starting tests in 3 seconds...\n")
    
    import time
    time.sleep(3)
    
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server")
        print("Please start the Flask server with: cd backend && python app.py")
    except Exception as e:
        print(f"ERROR: {e}")
