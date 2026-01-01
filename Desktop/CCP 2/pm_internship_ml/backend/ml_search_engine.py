"""
Machine Learning based search engine for internships
Uses TF-IDF for semantic search and cosine similarity
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
        """Initialize with TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)
        self.internships_df = None
        self.embeddings = None
        self.embeddings_path = os.path.join(
            os.path.dirname(__file__), '../models/embeddings.pkl'
        )
    
    def create_embeddings(self, internships_df: pd.DataFrame, 
                         force_recreate: bool = False) -> np.ndarray:
        """
        Create TF-IDF embeddings for all internships
        Combines title, description, and skills for comprehensive embeddings
        """
        # Check if embeddings already exist
        if os.path.exists(self.embeddings_path) and not force_recreate:
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.vectorizer = data['vectorizer']
                self.internships_df = internships_df
                return self.embeddings
        
        self.internships_df = internships_df
        
        # Create comprehensive text representation
        texts = []
        for _, row in internships_df.iterrows():
            combined_text = f"{row['title']} {row['company']} {row['description']} {row['skills_required']} {row['category']}"
            texts.append(combined_text)
        
        print(f"Creating TF-IDF embeddings for {len(texts)} internships...")
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
        
        # Save embeddings and vectorizer
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings, 'vectorizer': self.vectorizer}, f)
        
        return self.embeddings
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Semantic search using TF-IDF and cosine similarity
        Returns top K most relevant internships
        """
        if self.embeddings is None or self.internships_df is None:
            return []
        
        # Create embedding for the query
        query_embedding = self.vectorizer.transform([query]).toarray()[0]
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create results with scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.0:  # Only include if there's some similarity
                result = self.internships_df.iloc[idx].to_dict()
                result['relevance_score'] = float(similarities[idx])
                results.append(result)
        
        return results
    
    def multi_field_search(self, query: str, field_weights: Dict[str, float] = None, 
                          top_k: int = 10) -> List[Dict]:
        """
        Search with weighted importance on different fields
        Useful for filtering by specific criteria
        """
        if field_weights is None:
            field_weights = {
                'title': 0.4,
                'skills': 0.3,
                'description': 0.3
            }
        
        results = self.search(query, top_k=top_k*3)  # Get more results to filter
        
        # Re-rank based on field matches
        for result in results:
            score = 0
            query_lower = query.lower()
            
            # Title match weight
            if query_lower in result['title'].lower():
                score += field_weights.get('title', 0.4)
            
            # Skills match weight
            if query_lower in result['skills_required'].lower():
                score += field_weights.get('skills', 0.3)
            
            # Add to relevance score
            result['relevance_score'] = result.get('relevance_score', 0) + score
        
        # Sort by updated score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def category_filter_search(self, query: str, category: str, 
                              top_k: int = 10) -> List[Dict]:
        """
        Search within a specific category
        Useful for filtering by internship type
        """
        # Filter by category
        category_df = self.internships_df[
            self.internships_df['category'].str.lower() == category.lower()
        ]
        
        if category_df.empty:
            return []
        
        # Get indices of filtered results
        filtered_indices = category_df.index.tolist()
        
        # Search and filter
        query_embedding = self.vectorizer.transform([query]).toarray()[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Only consider similarities for filtered category
        category_similarities = [(idx, similarities[idx]) for idx in filtered_indices]
        category_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for idx, score in category_similarities[:top_k]:
            result = self.internships_df.iloc[idx].to_dict()
            result['relevance_score'] = float(score)
            results.append(result)
        
        return results
    
    def skill_based_search(self, skills: List[str], top_k: int = 10) -> List[Dict]:
        """
        Find internships that match required skills
        """
        query = " ".join(skills)
        return self.search(query, top_k=top_k)
