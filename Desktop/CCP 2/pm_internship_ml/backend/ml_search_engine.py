"""
Machine Learning based search engine for internships
Optimized with efficient caching and numpy operations
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import pickle
import os


class MLSearchEngine:
    """ML-powered semantic search for internships using TF-IDF"""
    
    def __init__(self, model_name: str = 'tfidf'):
        """Initialize with optimized TF-IDF vectorizer"""
        # Optimized parameters for better performance
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduced from 5000 for faster computation
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1,
            max_df=0.95,
            dtype=np.float32  # Use float32 for memory efficiency
        )
        self.internships_df = None
        self.embeddings = None
        self.embeddings_path = os.path.join(
            os.path.dirname(__file__), '../models/embeddings.pkl'
        )
        self._similarity_cache = {}
    
    def create_embeddings(self, internships_df: pd.DataFrame, 
                         force_recreate: bool = False) -> np.ndarray:
        """
        Create TF-IDF embeddings for all internships
        Optimized with caching and efficient storage
        """
        # Check if embeddings already exist
        if os.path.exists(self.embeddings_path) and not force_recreate:
            try:
                with open(self.embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings'].astype(np.float32)
                    self.vectorizer = data['vectorizer']
                    self.internships_df = internships_df
                    print("Loaded cached embeddings successfully")
                    return self.embeddings
            except Exception as e:
                print(f"Cache load failed, regenerating: {e}")
        
        self.internships_df = internships_df
        
        # Create comprehensive text representation with better concatenation
        texts = [
            f"{row['title']} {row['company']} {row['description']} {row['skills_required']} {row['category']}"
            for _, row in internships_df.iterrows()
        ]
        
        print(f"Creating TF-IDF embeddings for {len(texts)} internships...")
        # Use sparse matrix conversion then to dense for memory efficiency
        self.embeddings = self.vectorizer.fit_transform(texts).astype(np.float32).toarray()
        
        # Save embeddings and vectorizer
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings, 'vectorizer': self.vectorizer}, f)
        
        print("Embeddings created and cached successfully")
        return self.embeddings
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Optimized semantic search using TF-IDF and cosine similarity
        Returns top K most relevant internships with fallback for low matches
        """
        if self.embeddings is None or self.internships_df is None:
            return []
        
        # Handle empty query
        if not query or not query.strip():
            return []
        
        try:
            # Create embedding for the query
            query_embedding = self.vectorizer.transform([query]).astype(np.float32).toarray()[0]
            
            # Optimized cosine similarity calculation
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            # Get all indices sorted by relevance
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Create results with dynamic threshold
            results = []
            max_threshold = 0.001  # Minimum threshold
            
            for idx in sorted_indices:
                similarity_score = similarities[idx]
                
                # Include all results above threshold or top 5 regardless
                if similarity_score > max_threshold or len(results) < max(5, top_k // 2):
                    result = self.internships_df.iloc[idx].to_dict()
                    result['relevance_score'] = float(similarity_score)
                    results.append(result)
                    
                    # Stop if we have enough results
                    if len(results) >= top_k * 2:
                        break
            
            # If we still have no results, return top 5 regardless of score
            if not results:
                for idx in sorted_indices[:5]:
                    result = self.internships_df.iloc[idx].to_dict()
                    result['relevance_score'] = float(similarities[idx])
                    results.append(result)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return random sample of internships
            return self.internships_df.head(top_k).to_dict('records')
    
    def multi_field_search(self, query: str, field_weights: Dict[str, float] = None, 
                          top_k: int = 10) -> List[Dict]:
        """
        Optimized search with weighted importance on different fields
        """
        if field_weights is None:
            field_weights = {
                'title': 0.4,
                'skills': 0.3,
                'description': 0.3
            }
        
        # Get more results initially
        results = self.search(query, top_k=min(top_k * 2, len(self.internships_df)))
        
        if not results:
            return []
        
        # Re-rank based on field matches - vectorized operation
        query_lower = query.lower()
        for result in results:
            bonus_score = 0
            
            # Title match (higher weight)
            if query_lower in result.get('title', '').lower():
                bonus_score += field_weights.get('title', 0.4) * 0.5
            
            # Skills match (important)
            if query_lower in result.get('skills_required', '').lower():
                bonus_score += field_weights.get('skills', 0.3) * 0.5
            
            # Category match
            if query_lower in result.get('category', '').lower():
                bonus_score += field_weights.get('description', 0.3) * 0.3
            
            result['relevance_score'] = min(result.get('relevance_score', 0) + bonus_score, 1.0)
        
        # Sort by updated score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def category_filter_search(self, query: str, category: str, 
                              top_k: int = 10) -> List[Dict]:
        """
        Optimized search within a specific category
        """
        # Filter by category - using numpy for better performance
        category_mask = self.internships_df['category'].str.lower() == category.lower()
        filtered_indices = np.where(category_mask)[0]
        
        if len(filtered_indices) == 0:
            return []
        
        # Perform search
        query_embedding = self.vectorizer.transform([query]).astype(np.float32).toarray()[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get only filtered indices with high relevance
        category_similarities = [(idx, float(similarities[idx])) for idx in filtered_indices 
                                if similarities[idx] > 0.01]
        
        # Sort and limit
        category_similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(category_similarities))
        
        # Build results
        results = []
        for idx, score in category_similarities[:top_k]:
            result = self.internships_df.iloc[idx].to_dict()
            result['relevance_score'] = score
            results.append(result)
        
        return results
    
    def skill_based_search(self, skills: List[str], top_k: int = 10) -> List[Dict]:
        """
        Find internships that match required skills
        """
        query = " ".join(skills)
        return self.search(query, top_k=top_k)
