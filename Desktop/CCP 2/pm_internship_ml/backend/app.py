"""
Flask application for PM Internship ML Search System
Optimized with caching, lazy loading, and efficient data handling
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys
from functools import lru_cache
import time

sys.path.insert(0, os.path.dirname(__file__))
from geolocation_service import GeolocationService
from ml_search_engine import MLSearchEngine

app = Flask(__name__)
CORS(app)

# Global caching system
class Cache:
    search_cache = {}
    location_cache = {}
    categories_cache = None
    locations_cache = None
    cache_ttl = 3600  # 1 hour
    last_update = {}
    
    @classmethod
    def is_valid(cls, key):
        if key not in cls.last_update:
            return False
        return time.time() - cls.last_update[key] < cls.cache_ttl
    
    @classmethod
    def get(cls, cache_dict, key):
        if key in cache_dict and cls.is_valid(key):
            return cache_dict[key]
        return None
    
    @classmethod
    def set(cls, cache_dict, key, value):
        cache_dict[key] = value
        cls.last_update[key] = time.time()

# Lazy loading for better startup
internships_df = None
ml_engine = None

def get_internships_df():
    global internships_df
    if internships_df is None:
        DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/sample_internships.csv')
        internships_df = pd.read_csv(DATA_PATH)
        internships_df['location_lower'] = internships_df['location'].str.lower()
    return internships_df

def get_ml_engine():
    global ml_engine
    if ml_engine is None:
        ml_engine = MLSearchEngine()
        ml_engine.create_embeddings(get_internships_df())
    return ml_engine


@app.route('/api/search', methods=['POST'])
def search():
    """
    Main search endpoint with combined geolocation and ML search
    Optimized with caching and efficient filtering
    """
    data = request.json or {}
    query = data.get('query', '').strip()
    location = data.get('location', '').strip()
    search_type = data.get('search_type', 'semantic')
    top_k = min(data.get('top_k', 10), 100)  # Limit to 100 max
    
    if not query:
        return jsonify({'success': False, 'error': 'Query is required', 'results': []}), 200
    
    # Check cache first
    cache_key = f"{query}|{location}|{search_type}|{top_k}"
    cached = Cache.get(Cache.search_cache, cache_key)
    if cached is not None:
        return jsonify(cached)
    
    try:
        df = get_internships_df()
        engine = get_ml_engine()
        
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No data available', 'results': []}), 200
        
        # Perform semantic search - get extra results for location filtering
        results = []
        try:
            if search_type == 'semantic':
                results = engine.search(query, top_k=top_k * 2)
            elif search_type == 'multi-field':
                results = engine.multi_field_search(query, top_k=top_k * 2)
            else:
                results = engine.search(query, top_k=top_k * 2)
        except Exception as search_err:
            print(f"Search engine error: {search_err}")
            import traceback
            traceback.print_exc()
            results = []
        
        # Apply location filter if provided
        if location and results:
            location_lower = location.lower()
            
            # First try exact match (faster)
            exact_matches = [r for r in results if r.get('location', '').lower() == location_lower]
            if exact_matches:
                results = exact_matches
            else:
                # Find nearby internships with caching
                try:
                    location_key = f"{location}|100"
                    nearby_df = Cache.get(Cache.location_cache, location_key)
                    if nearby_df is None:
                        nearby_df = GeolocationService.find_nearby_internships(df, location, radius_km=100)
                        Cache.set(Cache.location_cache, location_key, nearby_df)
                    
                    if nearby_df is not None and not nearby_df.empty:
                        nearby_ids_set = set(nearby_df['id'].astype(int).tolist())
                        results = [r for r in results if int(r.get('id', -1)) in nearby_ids_set]
                except Exception as loc_err:
                    print(f"Location filter error: {loc_err}")
                    # Continue with unfiltered results
        
        # Clean and format results
        cleaned_results = []
        for result in results:
            try:
                clean_result = {
                    'id': int(result.get('id', 0)),
                    'title': str(result.get('title', '')),
                    'company': str(result.get('company', '')),
                    'location': str(result.get('location', '')),
                    'description': str(result.get('description', '')),
                    'skills_required': str(result.get('skills_required', '')),
                    'stipend': str(result.get('stipend', '')),
                    'duration_months': int(result.get('duration_months', 0)),
                    'category': str(result.get('category', '')),
                    'relevance_score': float(result.get('relevance_score', 0.0))
                }
                cleaned_results.append(clean_result)
            except Exception as clean_err:
                print(f"Result cleaning error: {clean_err}")
                continue
        
        # Sort by relevance score (highest first)
        cleaned_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        response = {
            'success': True,
            'total_results': len(cleaned_results),
            'results': cleaned_results[:top_k]
        }
        
        Cache.set(Cache.search_cache, cache_key, response)
        return jsonify(response)
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'results': []}), 200


@app.route('/api/search-by-location', methods=['POST'])
def search_by_location():
    """
    Search internships by location with nearby fallback
    """
    data = request.json
    location = data.get('location', '')
    top_k = data.get('top_k', 10)
    
    if not location:
        return jsonify({'error': 'Location is required'}), 400
    
    try:
        # Find internships in exact location
        exact_match = internships_df[
            internships_df['location'].str.lower() == location.lower()
        ]
        
        results = exact_match.head(top_k).to_dict('records')
        
        if len(results) < top_k:
            # Find nearby internships
            nearby = GeolocationService.find_nearby_internships(
                internships_df, location, radius_km=100
            )
            
            if not nearby.empty:
                # Exclude already found results
                found_ids = exact_match['id'].tolist()
                nearby_results = nearby[~nearby['id'].isin(found_ids)].head(top_k - len(results))
                results.extend(nearby_results.to_dict('records'))
        
        return jsonify({
            'success': True,
            'location': location,
            'total_results': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-by-skills', methods=['POST'])
def search_by_skills():
    """
    Search internships by required skills
    """
    data = request.json
    skills = data.get('skills', [])
    top_k = data.get('top_k', 10)
    
    if not skills:
        return jsonify({'error': 'Skills list is required'}), 400
    
    try:
        results = ml_engine.skill_based_search(skills, top_k=top_k)
        
        return jsonify({
            'success': True,
            'skills': skills,
            'total_results': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-by-category', methods=['POST'])
def search_by_category():
    """
    Search internships by category
    """
    data = request.json
    category = data.get('category', '')
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    try:
        if query:
            results = ml_engine.category_filter_search(query, category, top_k=top_k)
        else:
            # Just filter by category
            category_results = internships_df[
                internships_df['category'].str.lower() == category.lower()
            ]
            results = category_results.head(top_k).to_dict('records')
        
        return jsonify({
            'success': True,
            'category': category,
            'total_results': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/locations', methods=['GET'])
def get_locations():
    """
    Get list of all available locations (cached)
    """
    if Cache.locations_cache is not None and Cache.is_valid('locations'):
        return jsonify({
            'success': True,
            'locations': Cache.locations_cache
        })
    
    df = get_internships_df()
    locations = df['location'].unique().tolist()
    Cache.locations_cache = locations
    Cache.last_update['locations'] = time.time()
    
    return jsonify({
        'success': True,
        'locations': locations
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """
    Get list of all internship categories (cached)
    """
    if Cache.categories_cache is not None and Cache.is_valid('categories'):
        return jsonify({
            'success': True,
            'categories': Cache.categories_cache
        })
    
    df = get_internships_df()
    categories = df['category'].unique().tolist()
    Cache.categories_cache = categories
    Cache.last_update['categories'] = time.time()
    
    return jsonify({
        'success': True,
        'categories': categories
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        df = get_internships_df()
        return jsonify({
            'status': 'healthy',
            'total_internships': len(df),
            'ml_engine_loaded': ml_engine is not None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Use threaded mode for better concurrency and disable reloader
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
