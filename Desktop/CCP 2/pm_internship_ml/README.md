# PM Internship Scheme - ML Enhanced Search System

## Project Overview

This project improves the PM Internship scheme website by addressing two key issues:

1. **Geolocation-Based Search**: When no internships are found in a specified region, the system now shows internships from nearby locations sorted by distance.

2. **ML-Powered Semantic Search**: Replaced traditional keyword/word-based search with advanced machine learning using sentence embeddings and cosine similarity for more intelligent and relevant results.

## Features

### 1. Semantic Search

- Uses pre-trained sentence transformers (all-MiniLM-L6-v2)
- Understands context and meaning beyond exact keyword matching
- Returns results ranked by relevance score

### 2. Geolocation-Based Nearby Search

- Automatically finds internships in nearby cities when exact location has no results
- Uses geodesic distance calculations
- Configurable search radius (default: 100 km)
- Shows distance from requested location

### 3. Multi-Field Search

- Weighted search across title, skills, and description
- Customizable field weights for different search priorities

### 4. Category-Based Search

- Filter internships by type (Technology, Data Science, Design, etc.)
- Combined with semantic search for better results

### 5. Skill-Based Search

- Find internships matching specific skill requirements
- Useful for skill-focused career planning

## Project Structure

```
pm_internship_ml/
├── backend/
│   ├── app.py                      # Flask application
│   ├── ml_search_engine.py        # ML search implementation
│   └── geolocation_service.py     # Geolocation logic
├── data/
│   └── sample_internships.csv     # Sample internship dataset
├── models/
│   └── embeddings.pkl             # Pre-computed embeddings cache
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### 1. Clone/Setup the project

```bash
cd pm_internship_ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Note: The first time you run the application, it will download the sentence-transformers model (~22 MB). This is normal and happens automatically.

## Usage

### Starting the Server

```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### 1. Semantic Search

**Endpoint**: `POST /api/search`

```json
{
  "query": "python web development",
  "location": "Bangalore",
  "search_type": "semantic",
  "top_k": 10
}
```

**Response**:

```json
{
  "success": true,
  "total_results": 8,
  "results": [
    {
      "id": 1,
      "title": "Software Development Intern",
      "company": "TCS",
      "location": "Bangalore",
      "description": "Develop web applications using Python and Django",
      "relevance_score": 0.89,
      ...
    }
  ]
}
```

#### 2. Location-Based Search

**Endpoint**: `POST /api/search-by-location`

```json
{
  "location": "Mumbai",
  "top_k": 10
}
```

Returns internships in Mumbai and nearby cities if not enough results in Mumbai.

#### 3. Skill-Based Search

**Endpoint**: `POST /api/search-by-skills`

```json
{
  "skills": ["Python", "Machine Learning", "TensorFlow"],
  "top_k": 10
}
```

#### 4. Category Search

**Endpoint**: `POST /api/search-by-category`

```json
{
  "category": "Technology",
  "query": "backend development",
  "top_k": 10
}
```

#### 5. Get Available Locations

**Endpoint**: `GET /api/locations`

#### 6. Get Available Categories

**Endpoint**: `GET /api/categories`

#### 7. Health Check

**Endpoint**: `GET /api/health`

## How ML Search Works

### 1. Embedding Generation

- Each internship is converted to a text representation combining: title, company, description, skills, and category
- Text is encoded into 384-dimensional embeddings using a pre-trained model
- Embeddings are cached for faster subsequent searches

### 2. Query Processing

- User query is encoded to the same embedding space
- Cosine similarity is calculated between query and all internship embeddings
- Results are ranked by similarity score (0-1)

### 3. Benefits Over Keyword Search

- **Context Understanding**: "Python development" will match with "Django web applications"
- **Typo Tolerance**: Similar words are matched even with minor spelling differences
- **Semantic Relevance**: Results based on meaning, not just keyword presence
- **Better Ranking**: More relevant results appear first

## How Geolocation Search Works

### 1. Location Matching

- User's requested location is matched to a database of Indian cities with coordinates
- Current supported cities: Bangalore, Mumbai, Delhi, Hyderabad, Pune, Chennai, Kolkata, Jaipur, Lucknow, Ahmedabad, Gurgaon, Noida

### 2. Nearby City Finding

- Geodesic distance is calculated between user location and all other cities
- Cities within 100 km radius are identified
- Results from these nearby cities are included if exact location has no results

### 3. Distance Calculation

- Uses the `geopy.distance.geodesic` function for accurate distance calculation on Earth's surface
- Results are sorted by distance from the user's requested location

## Sample Data

The system comes with 20 sample internships across various fields:

- Technology (Software Development, Full Stack, DevOps, etc.)
- Data Science & AI/ML
- Cloud Computing
- Design & UI/UX
- Business & Marketing
- Hardware & IoT
- And more...

To use your own data:

1. Replace `data/sample_internships.csv` with your dataset
2. Ensure it has columns: `id, title, company, location, latitude, longitude, description, skills_required, stipend, duration_months, category`
3. Run `python backend/app.py` - embeddings will be regenerated automatically

## Performance Optimization

### Embedding Caching

- Embeddings are cached in `models/embeddings.pkl`
- First run will take ~10-20 seconds to generate embeddings
- Subsequent runs load cached embeddings instantly

### Scalability

- Current system handles up to ~10,000 internships comfortably
- For larger datasets, consider:
  - Using vector databases (e.g., Faiss, Milvus)
  - Implementing batch processing
  - Using approximate nearest neighbor search

## Future Enhancements

1. **User Profiles**: Save user preferences and search history
2. **Recommendations**: Recommend internships based on previous searches
3. **Salary Predictions**: ML model to predict salary based on skills and location
4. **Review System**: User reviews and ratings for internships
5. **Application Tracking**: Track application status for each internship
6. **Advanced Filtering**: Filter by stipend, duration, company type, etc.
7. **Web Interface**: Create a user-friendly web UI
8. **Real Data Integration**: Connect to actual PM Internship scheme database

## Technologies Used

- **Flask**: Web framework for REST API
- **Sentence-Transformers**: Pre-trained models for text embeddings
- **Scikit-learn**: Cosine similarity calculations
- **Pandas**: Data manipulation and analysis
- **Geopy**: Geolocation distance calculations
- **NumPy**: Numerical computations

## License

This is a college project for educational purposes.

## Author

Created as part of the PM Internship Scheme improvement project.
