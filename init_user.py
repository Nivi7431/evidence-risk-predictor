from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['risk_prediction_db']
users_collection = db['users']

# Add a test user
users_collection.insert_one({
    'username': 'nivi',
    'password': 'test123'
})

print("✅ User 'nivi' with password 'test123' inserted into MongoDB!")
