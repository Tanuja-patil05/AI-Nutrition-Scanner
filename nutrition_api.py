import requests
import os

class NutritionService:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("CALORIE_NINJAS_API_KEY")
        self.base_url = "https://api.calorieninjas.com/v1/nutrition?query="

    def get_nutrition_data(self, food_query):
        """
        Fetches nutritional data from CalorieNinjas API.
        If no API key is provided, it returns mock data.
        """
        if not self.api_key:
            return self._get_mock_data(food_query)

        try:
            headers = {'X-Api-Key': self.api_key}
            response = requests.get(self.base_url + food_query, headers=headers)
            
            if response.status_code == requests.codes.ok:
                data = response.json()
                if data.get('items') and len(data['items']) > 0:
                    return data['items'][0]
            return self._get_mock_data(food_query) # Fallback to mock if API fails
        except Exception:
            return self._get_mock_data(food_query)

    def _get_mock_data(self, food_query):
        """
        Provides realistic mock data for UI testing.
        """
        import random
        return {
            "name": food_query,
            "calories": random.uniform(200, 600),
            "protein_g": random.uniform(5, 30),
            "fat_total_g": random.uniform(5, 25),
            "carbohydrates_total_g": random.uniform(10, 50),
            "fiber_g": random.uniform(1, 8),
            "sugar_g": random.uniform(2, 15)
        }

    def get_health_score(self, nutrition_data):
        """
        Calculates a proprietary health score (0-100) based on macros.
        Simple heuristic: Higher protein/fiber = better, Higher sugar/fat = worse.
        """
        protein = nutrition_data.get('protein_g', 0)
        fiber = nutrition_data.get('fiber_g', 0)
        fat = nutrition_data.get('fat_total_g', 0)
        sugar = nutrition_data.get('sugar_g', 0)
        
        # Simple weighted score
        score = 50 + (protein * 2) + (fiber * 3) - (fat * 1.5) - (sugar * 1.2)
        return min(max(int(score), 5), 100)

if __name__ == "__main__":
    service = NutritionService()
    data = service.get_nutrition_data("pizza")
    print(f"Nutrition for pizza: {data}")
    print(f"Health Score: {service.get_health_score(data)}")
