"""
Flask application for PM Internship ML Search System
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from geolocation_service import GeolocationService
from ml_search_engine import MLSearchEngine

app = Flask(__name__)
CORS(app)

# Initialize services
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/sample_internships.csv')
internships_df = pd.read_csv(DATA_PATH)
ml_engine = MLSearchEngine()
ml_engine.create_embeddings(internships_df)


@app.route('/api/search', methods=['POST'])
def search():
    """
    Main search endpoint with combined geolocation and ML search
    """
    data = request.json
    query = data.get('query', '')
    location = data.get('location', '')
    search_type = data.get('search_type', 'semantic')  # semantic or keyword
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        # Perform semantic search
        if search_type == 'semantic':
            results = ml_engine.search(query, top_k=top_k)
        elif search_type == 'multi-field':
            results = ml_engine.multi_field_search(query, top_k=top_k)
        else:
            results = ml_engine.search(query, top_k=top_k)
        
        # Apply location filter if provided
        if location:
            # First try to find in exact location
            exact_location_results = [
                r for r in results if r['location'].lower() == location.lower()
            ]
            
            if exact_location_results:
                results = exact_location_results
            else:
                # Find nearby internships
                nearby = GeolocationService.find_nearby_internships(
                    internships_df, location, radius_km=100
                )
                
                if not nearby.empty:
                    nearby_ids = nearby['id'].tolist()
                    results = [r for r in results if r['id'] in nearby_ids]
                    
                    # Sort by distance
                    for r in results:
                        r['distance_km'] = nearby[nearby['id'] == r['id']]['distance_km'].values[0]
                    results.sort(key=lambda x: x.get('distance_km', float('inf')))
        
        # Clean results
        for result in results:
            result.pop('distance_km', None)
        
        return jsonify({
            'success': True,
            'total_results': len(results),
            'results': results[:top_k]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    Get list of all available locations
    """
    locations = internships_df['location'].unique().tolist()
    return jsonify({
        'success': True,
        'locations': locations
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """
    Get list of all internship categories
    """
    categories = internships_df['category'].unique().tolist()
    return jsonify({
        'success': True,
        'categories': categories
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'total_internships': len(internships_df)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
